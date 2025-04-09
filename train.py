# external
import torch
import imageio
import comet_ml
import torchvision
from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm
from typing import List, Union
from omegaconf import OmegaConf
from pytorch3d.structures import Meshes

# local
from src import common, utils, model, constants, parser
from src.losses.loss import Loss
from src.renderer.render import render
from src.datasets.ho_dataset import HumanObjectData
from src.renderer.render_manager import RenderManager, Render


class GaussianTrainer:

    def __init__(
        self,
        cfg: OmegaConf,
        exp: Union[comet_ml.BaseExperiment, None],
    ) -> None:

        self.device = constants.device
        self.cfg = cfg
        self.log_dir: Path = Path(self.cfg.log_dir)
        self.render_mode = self.cfg.mode
        self.verbose = self.cfg.train.verbose
        self.renders: RenderManager = RenderManager()
        self.comet_experiment: Union[comet_ml.BaseExperiment, None] = exp
        self.train_dataset = HumanObjectData(cfg=self.cfg)
        self.random_sample = utils.general.RandomIndexIterator(len(self.train_dataset))

        if self.cfg.train.anim_frames is None:
            self.cfg.train.anim_frames = self.train_dataset.num_frames

        self.bg_color = 0.498 * torch.ones(3, dtype=torch.float32, device=self.device)

        # import gc
        # ts = [x for x in gc.get_objects() if isinstance(x, torch.Tensor)]
        # for i in ts:
        #     if str(i.device) == "cpu":
        #         print(i.shape)

        # init gaussian models
        self.gaussian_models: List[model.gaussian_model.GaussianModel] = []
        self.object_gs, self.human_gs = None, None
        if "o" in self.cfg.mode:  # mode = o/ho
            self.object_gs = model.object_gs.ObjectGS(
                cfg=self.cfg, data=self.train_dataset
            )
            self.gaussian_models.append(self.object_gs)
        if "h" in self.cfg.mode:  # mode = h/ho
            self.human_gs = model.human_gs.HumanGS(
                cfg=self.cfg, data=self.train_dataset
            )
            self.gaussian_models.append(self.human_gs)

        # init loss
        self.loss: Loss = Loss(cfg=self.cfg.loss, verbose=False)  # self.verbose)
        self.loss.compute_o_loss = True if self.render_mode == "o" else False
        self.loss.compute_h_loss = True if self.render_mode == "h" else False
        self.loss.compute_ho_loss = True if self.render_mode == "ho" else False

    def get_sample(self) -> int:
        """Sample `valid` frame from training data."""

        # HumanGS: use all frames!
        training_sample = next(self.random_sample)

        # ObjectGS: only use frames where COLMAP converged
        if self.render_mode == "o":
            valid = self.train_dataset[training_sample]["obj_valid"].item()
            while not valid:
                # unlikely to get stuck in while-loop...otherwise use try-catch to be safe
                logger.warning(
                    f"Re-sampling. Missing object pose in frame {training_sample}."
                )
                training_sample = next(self.random_sample)
                valid = self.train_dataset[training_sample]["obj_valid"].item()

        return training_sample

    def train(self):
        pbar = tqdm(
            range(self.cfg.train.iterations), desc=f"Training {self.render_mode}"
        )

        for i in range(self.cfg.train.iterations):
            # torch.cuda.empty_cache()
            self.renders.reset()

            # NOTE: use random data samples/bg colors when learning
            #       appearance models (except for "debug" iterations)
            if i % self.cfg.train.dbg_interval == 0:
                sample_idx = self.cfg.train.sample
                self.bg_color = 0.498 * torch.ones(
                    3, dtype=torch.float32, device=self.device
                )
            else:
                sample_idx = self.get_sample()
                self.bg_color = torch.rand(3, dtype=torch.float32, device=self.device)

            data = self.train_dataset[sample_idx]
            self.renders.add_gt_renders(data=data, bg_color=self.bg_color)

            for gs in self.gaussian_models:
                if gs.render_mode == self.render_mode:
                    gs.update_learning_rate(i)
                    if i % 1000 == 0:
                        gs.oneupSHdegree()

                # deform gaussians
                gs.forward(sample_idx)

            # render gaussians
            render_pkg = self.render_gaussians(data)

            # loss computation
            loss, loss_dict = self.loss(
                renders=self.renders,
                human_gs=self.human_gs,
                object_gs=self.object_gs,
                rnd_idx=sample_idx,
                iter=i,
                visibility_filter_h=render_pkg["visibility_filter"],
            )

            # backward pass
            loss.backward()

            with torch.no_grad():
                self.object_densification(render_pkg=render_pkg, i=i)
                if i % 10 == 0:
                    postfix_dict = {}
                    for k, v in loss_dict.items():
                        postfix_dict["l_" + k] = f"{v.item():.4f}"
                    pbar.set_postfix(postfix_dict)
                    pbar.update(10)

                if i % self.cfg.train.dbg_interval == 0:
                    self.renders.save(
                        run_mode="train",
                        render_mode=self.render_mode,
                        out_p=f"{self.cfg.train_dir}/{i:06d}.png",
                    )

                    common.comet_utils.log_dict(
                        self.comet_experiment,
                        loss_dict,
                        step=i,
                        epoch=0,
                    )

                if i % self.cfg.train.anim_interval == 0 and i > 0:
                    self.animate(iter=i)

            # Optimizer step
            for gs in self.gaussian_models:
                if gs.render_mode == self.render_mode:
                    gs.optimizer.step()
                    gs.optimizer.zero_grad(set_to_none=True)

        # wrap up training
        pbar.close()

        if self.cfg.train.save_ckpt:
            self.save_ckpt()

        if self.cfg.train.save_animation:
            self.animate()

        if self.cfg.train.save_training:
            _video_p = (
                self.log_dir
                / f"train_{self.render_mode}_{self.cfg.train.iterations:06d}.mp4"
            )
            utils.general.create_video(
                imgs_p=Path(self.cfg.train_dir),
                video_p=_video_p,
                reset_folder=self.cfg.train.remove_frames,
                v=self.verbose,
            )
            if self.comet_experiment is not None:
                self.comet_experiment.log_video(_video_p, name="train")

    def render_gaussians(self, data) -> dict:
        if self.render_mode == "h":
            # need to render hand in both h/ho modes, s.t. ∃
            # visibility_filter_h for loss.compute_hand_losses()
            render_pkg = render(
                gaussians=self.human_gs.out,
                bg_color=self.bg_color,
                viewpoint_cam=self.train_dataset.cam,
            )

            _pred_img_h = render_pkg["render"]

            # NOTE: remove object-occluded areas from predicted hand (to prevent these gaussians being pruned)
            obj_msk = self.renders.get("o", "gt_msk")  # TODO: use gt_msk from `data`
            pred_img_w_mask = _pred_img_h * (1 - obj_msk)
            _pred_img_h_cropped = pred_img_w_mask + (
                obj_msk * self.bg_color[:, None, None]
            )

            self.renders.append(
                Render(mode=self.render_mode, name="pred_img_full", img=_pred_img_h),
                Render(mode=self.render_mode, name="pred_img", img=_pred_img_h_cropped),
                Render(
                    mode=self.render_mode,
                    name="pred_msk",
                    img=self.get_pred_msk(
                        render_pkg["alphas"],
                        mask=(data["gt_msk"] == constants.SEGM_IDS["object"]).float(),
                    ),
                ),
            )

        if self.render_mode == "o":
            render_pkg = render(
                gaussians=self.object_gs.out,
                bg_color=self.bg_color,
                viewpoint_cam=self.train_dataset.cam,
            )

            _pred_img_o = render_pkg["render"]
            self.renders.append(
                Render(mode=self.render_mode, name="pred_img", img=_pred_img_o),
                Render(
                    mode=self.render_mode,
                    name="pred_msk",
                    img=self.get_pred_msk(
                        render_pkg["alphas"],
                        mask=(data["gt_msk"] == constants.SEGM_IDS["human"]).float(),
                    ),  # NOTE: use hand gt mask here?
                ),
            )

        if self.render_mode == "ho":
            combined_gaussians = {
                k: torch.cat((self.human_gs.out[k], self.object_gs.out[k]), axis=0)
                for k in self.object_gs.out.keys()
                if k not in ["active_sh_degree", "normals", "joints"]
            }

            # NOTE: HumanGS SH are NOTE deformed -> use lower SH-degree for joint rasterization
            combined_gaussians["active_sh_degree"] = self.human_gs.out[
                "active_sh_degree"
            ]

            render_pkg = render(
                gaussians=combined_gaussians,
                viewpoint_cam=self.train_dataset.cam,
                bg_color=self.bg_color,
            )

            self.renders.append(
                Render(
                    mode=self.render_mode, name="pred_img", img=render_pkg["render"]
                ),
                Render(
                    mode=self.render_mode,
                    name="pred_msk",  # NOTE: segmentation mask, not binary
                    img=self.get_gs_segmentation(
                        data=data,
                        gaussians=combined_gaussians,
                    ),  # NOTE: this mask is NOT binary; it's a segmentation mask
                ),
            )

        self.renders.append(
            Render(
                mode=self.render_mode, name="distortions", img=render_pkg["distortions"]
            ),
            Render(
                mode=self.render_mode,
                name="pred_normal",
                img=render_pkg["normals_render"],
            ),
            Render(mode=self.render_mode, name="normals", img=render_pkg["normals"]),
            Render(
                mode=self.render_mode,
                name="normals_from_depth",
                img=render_pkg["normals_from_depth"],
            ),
        )

        return render_pkg

    ################################################################################################################
    # visualization methods ########################################################################################
    ################################################################################################################
    @torch.no_grad()
    def get_pred_msk(self, pred_img: torch.Tensor, mask=None) -> torch.Tensor:
        """Compute mask of pixels occupied by gaussians. Return img tensor."""
        pred_msk = pred_img.detach().clone()
        # clean up alpha blended image a bit (make binary)
        pred_msk[pred_msk != 0] = 1

        if mask is not None:
            # remove occluded area from pred_msk
            pred_msk = pred_msk * (1 - mask)

        if pred_msk.shape[0] == 1:
            pred_msk = pred_msk.repeat_interleave(
                3, dim=0
            )  # ensure out-shape [3, H, W]

        return pred_msk

    @torch.no_grad()
    def get_gs_segmentation(
        self, data, gaussians: model.gaussian_model.GaussianModel
    ) -> torch.Tensor:
        """Render segmentation mask of combined renderer."""
        n_h_gs = len(self.human_gs.get_xyz)
        # colors correspond to normalized segmentation colors as defined in src.constants.SEGM_IDS
        colors = torch.zeros(len(gaussians["xyz"]), 3)
        target_values = torch.tensor([0.0000, 0.1961, 0.5882], device=self.device)
        colors[:n_h_gs] = torch.tensor([0.5882, 0.5882, 0.5882])
        colors[n_h_gs:] = torch.tensor([0.1961, 0.1961, 0.1961])

        _gaussians = {
            "xyz": gaussians["xyz"],
            "shs": colors.to(self.device),
            "opacity": gaussians["opacity"],
            "scales": gaussians["scales"],
            "rotq": gaussians["rotq"],
            "active_sh_degree": gaussians["active_sh_degree"],
        }
        render_pkg = render(
            gaussians=_gaussians,
            viewpoint_cam=self.train_dataset.cam,
            bg_color=torch.zeros(
                3, dtype=torch.float32, device=self.device
            ),  # bg should be black
            precomp_colors=True,  # this just means, that we pass the colors of the gaussians we want...
        )

        gs_sem_msk = render_pkg["render"]
        diffs = torch.abs(gs_sem_msk.unsqueeze(3) - target_values.view(1, 1, 1, -1))
        min_indices = torch.argmin(diffs, dim=-1)
        gs_sem_msk_rounded = target_values[min_indices]
        # FIXME: add tolerance; otherwise code breaks here due to floating point errors!
        # assert gs_sem_msk_rounded.unique().shape == torch.Size([3]), logger.error(
        # "Rendered segmentation mask contains invalid pixel values!"
        # )

        return gs_sem_msk_rounded

    @torch.no_grad()
    def animate(
        self, iter: Union[int, None] = None, viz_alignment: bool = False
    ) -> None:
        """Animate full sequence with current state."""
        iter_s = "final" if iter is None else f"{iter:06d}"
        vid_p: Path = self.log_dir / f"anim_{self.render_mode}_{iter_s}.mp4"
        writer = imageio.get_writer(
            vid_p,
            fps=10,
            mode="I",
            format="FFMPEG",
            macro_block_size=1,
        )

        n_anim_frames = self.cfg.train.anim_frames
        # self.bg_color = 0.498 * torch.ones(3, dtype=torch.float32, device=self.device)
        self.bg_color = torch.ones(3, dtype=torch.float32, device=self.device)

        if viz_alignment:
            v3d_deform = {}
            for gs in self.gaussian_models:
                v3d_deform[gs.type] = []
            hand_meshes: List[Meshes] = []

        for idx in tqdm(
            range(0, self.train_dataset.num_frames),
            desc=f"Generating Full Sequence Animation ({self.render_mode})",
        ):
            if n_anim_frames is not None and idx > n_anim_frames:
                break

            self.renders.reset()
            data = self.train_dataset[idx]
            self.renders.add_gt_renders(data=data)
            for gs in self.gaussian_models:
                gs.forward(idx)

                if viz_alignment:
                    v3d_deform[gs.type].append(gs.out["xyz"].detach().cpu().numpy())
                    if gs.type == "human":
                        hand_meshes.append(gs.current_mesh)

            self.render_gaussians(data)
            frame = self.renders.save(
                run_mode="anim",
                render_mode=self.render_mode,
                out_p=None,
            )

            frame_hwc = (frame.permute(1, 2, 0) * 255).clamp(0, 255).byte()
            writer.append_data(frame_hwc.cpu().numpy())

        writer.close()
        logger.success(f"Find results at: {vid_p}")

        if self.comet_experiment is not None:
            self.comet_experiment.log_video(file=vid_p)

        # TODO: replace AITVIEWER w/ GLOSS!
        # see:  ./datapipeline/src/visualize_fits_gloss.py/visualize()
        if viz_alignment:
            logger.warning(f"Re-implement alignment visualization w/ GLOSS!")

        # TODO: refactor these methods at some point...
        # use the orbiting camera & centerd/static gaussians like GaussianExtractor

        # works for now but is ugly/duplicated code/...
        # if self.render_mode == "h":
        #     self.animate_canonical_h(iter=iter)

        # if self.render_mode == "o":
        #     self.animate_canonical_o(iter=iter)

        return

    @torch.no_grad()
    def animate_canonical_o(self, iter=None, animate_orbit: bool = False) -> None:
        # from src.losses.loss_utils import get_l1_loss
        # from src.utils.image import add_caption

        iter_s = "final" if iter is None else f"{iter:06d}"
        vid_p: Path = self.log_dir / f"anim_o_canon_{iter_s}.mp4"
        writer = imageio.get_writer(
            vid_p,
            fps=10,
            mode="I",
            format="FFMPEG",
            macro_block_size=1,
        )

        self.bg_color = 0.498 * torch.ones(3, dtype=torch.float32, device=self.device)

        n_frames_per_axis: int = 60  # rotation discretization
        # sample = self.cfg.train.sample
        # data = self.train_dataset[sample]
        # msk_object = (data["gt_msk"]== constants.SEGM_IDS["object"]).float()
        # gt_img = (
        #     self.bg_color[:, None, None] * (1.0 - msk_object)
        #     + data["gt_img"] * msk_object
        # )

        # setup
        init_transl = self.object_gs.obj_trans[self.cfg.train.sample].detach()
        init_rot = utils.rotations.rotation_6d_to_matrix(
            self.object_gs.obj_rot[self.cfg.train.sample].detach()
        )
        center = torch.mean(self.object_gs._xyz, dim=0)

        translation_to_origin = torch.eye(4, device=init_rot.device)
        translation_to_origin[:3, 3] = -center

        translation_back = torch.eye(4, device=init_rot.device)
        translation_back[:3, 3] = center

        initial_transform = torch.eye(4, device=init_rot.device)
        initial_transform[:3, :3] = init_rot
        initial_transform[:3, 3] = init_transl

        frame_idx = 0  # counter variable for output frames

        _rotations = [(["y"], "y")]

        if animate_orbit:
            # also rotate around these axes!
            _rotations.extend(
                [
                    (["x"], "x"),
                    (["z"], "z"),
                    (["x", "y"], "xy"),
                    (["x", "z"], "xz"),
                    (["y", "z"], "yz"),
                    (["x", "y", "z"], "xyz"),
                ]
            )

        # best_loss = float("inf")
        # _pose_cache = None

        for axes, desc in _rotations:
            for idx in tqdm(range(n_frames_per_axis), desc=f"Canonical: {desc}"):
                angle = torch.tensor(
                    (2 * torch.pi * idx) / n_frames_per_axis, device=self.device
                )

                # compose rotation
                rotation = torch.eye(4, device=self.device)
                for axis in axes:
                    rotation = rotation @ utils.rotations.create_rotation_matrix(
                        angle, axis, self.device
                    )

                combined_transform = (
                    initial_transform  # initially estimated pose
                    @ translation_back  # world origin -> center
                    @ rotation  # per-frame rotations
                    @ translation_to_origin  # center -> world origin
                )

                self.object_gs.forward(
                    obj_rot=combined_transform[:3, :3],
                    obj_trans=combined_transform[:3, 3],
                )

                render_pkg = render(
                    gaussians=self.object_gs.out,
                    bg_color=self.bg_color,
                    viewpoint_cam=self.train_dataset.cam,
                )
                pred_img = render_pkg["render"]
                normal_img = render_pkg["normals_render"]

                # remove hand mask
                # msk_object = (data["gt_msk"]== constants.SEGM_IDS["human"]).float()
                # pred_img = (1 - msk_human) * pred_img + data[
                #     "msk_human"
                # ] * self.bg_color[:, None, None]

                # l1 = get_l1_loss(pred_img=pred_img, gt_img=gt_img).item()

                # if l1 < best_loss:
                #     best_loss = l1
                #     _pose_cache = {
                #         "transform": combined_transform.clone(),
                #         "pred_img": pred_img.clone(),
                #         "angle": angle.item(),
                #         "axes": desc,
                #         "loss": l1,
                #     }

                img_grid = torchvision.utils.make_grid(
                    [normal_img, pred_img],
                    normalize=True,
                    nrow=2,
                    padding=0,
                )

                # img_grid = add_caption(
                #     img_grid, f"L1 Loss: {l1:.4f}; angle: {angle:.4f}, axes: {desc}"
                # )

                frame_hwc = (img_grid.permute(1, 2, 0) * 255).clamp(0, 255).byte()
                writer.append_data(frame_hwc.cpu().numpy())
                frame_idx += 1

        writer.close()

        # if _pose_cache is not None:
        #     _img_p: Path = self.log_dir / f"best_pose_{iter_s}.png"
        #     _img_grid = torchvision.utils.make_grid(
        #         [gt_img, _pose_cache["pred_img"]],
        #         normalize=True,
        #         nrow=2,
        #         padding=0,
        #     )

        #     _img_grid = add_caption(
        #         _img_grid,
        #         f"Pose w/ lowest L1 Loss: {_pose_cache['loss']:.4f}; angle: {_pose_cache['angle']:.4f}, axes: {_pose_cache['axes']}",
        #     )

        #     torchvision.utils.save_image(_img_grid, _img_p)

        if self.comet_experiment is not None:
            self.comet_experiment.log_video(file=vid_p)

    @torch.no_grad()
    def animate_canonical_h(self, iter=None):
        """Render canonical sequence w/ pre-defined hand pose."""
        iter_s = "final" if iter is None else f"{iter:06d}"
        vid_p: Path = self.log_dir / f"anim_h_canon_{iter_s}.mp4"

        writer = imageio.get_writer(
            vid_p,
            fps=10,
            mode="I",
            format="FFMPEG",
            macro_block_size=1,
        )

        self.bg_color = 0.498 * torch.ones(3, dtype=torch.float32, device=self.device)

        n_frames = self.cfg.train.anim_frames

        # initial hand rotation
        global_orient_y = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        global_orient_z = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        initial_orient = torch.mm(global_orient_y, global_orient_z)

        # per-frame rotation
        global_orientations = []
        for i in range(n_frames):
            angle = torch.tensor(2 * torch.pi * i / n_frames, dtype=torch.float32)

            # rotate around y-axis
            cos_t = torch.cos(angle)
            sin_t = torch.sin(angle)
            rot_y = torch.tensor(
                [[cos_t, 0.0, sin_t], [0.0, 1.0, 0.0], [-sin_t, 0.0, cos_t]]
            )

            frame_orient = torch.mm(rot_y, initial_orient)
            frame_orient_aa = utils.rotations.matrix_to_axis_angle(frame_orient)
            global_orientations.append(frame_orient_aa.cuda())

        for idx in tqdm(
            range(n_frames), desc=f"Generating Canonical Animation ({self.render_mode})"
        ):
            # TODO: fix
            transl = torch.tensor(
                [-0.075, 0.075, 0.35], dtype=torch.float32, device=self.device
            )
            hand_pose = torch.zeros(
                self.human_gs.smplx.NUM_HAND_JOINTS * 3,
                dtype=torch.float32,
                device=self.device,
            )
            body_pose = torch.zeros(
                self.human_gs.smplx.NUM_BODY_JOINTS * 3,
                dtype=torch.float32,
                device=self.device,
            )

            self.human_gs.forward(
                smplx_global_orient=global_orientations[idx],
                smplx_transl=transl,
                smplx_body_pose=body_pose,
                smplx_left_hand_pose=hand_pose,
                smplx_right_hand_pose=hand_pose,
            )
            render_pkg = render(
                gaussians=self.human_gs.out,
                bg_color=self.bg_color,
                viewpoint_cam=self.train_dataset.cam,
            )
            img_grid = torchvision.utils.make_grid(
                [render_pkg["render"], render_pkg["normals_render"]],
                normalize=True,
                nrow=2,
                padding=0,
            )
            frame_hwc = (img_grid.permute(1, 2, 0) * 255).clamp(0, 255).byte()
            writer.append_data(frame_hwc.cpu().numpy())

        writer.close()

        if self.comet_experiment is not None:
            self.comet_experiment.log_video(file=vid_p)

    def object_densification(self, render_pkg, i: int) -> None:
        _densify: bool = (
            self.render_mode == "o"  # densify when learning object appearance only
            and i >= self.cfg.object.densify.start  # densify phase started
            and i < self.cfg.object.densify.end  # densify phase not over
            and self.object_gs.n_gs < self.cfg.object.max_n_gs  # cap N gaussians
        )

        if not _densify:
            return

        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        viewspace_point_tensor = render_pkg["viewspace_points"]

        self.object_gs.max_radii2D[visibility_filter] = torch.max(
            self.object_gs.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        self.object_gs.add_densification_stats(
            viewspace_point_tensor, visibility_filter
        )

        if i % self.cfg.object.densify.interval == 0:
            size_threshold = 20 if i > self.cfg.object.opacity_reset_interval else None
            self.object_gs.densify_and_prune(
                self.cfg.object.densify.grad_thresh,
                min_opacity=self.cfg.object.prune_min_opacity,
                extent=3.0,
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.object.max_n_gs,
            )

        return

    def save_ckpt(self, iter=None):
        ckpt_p: Path = self.log_dir / "ckpt"
        if not ckpt_p.exists():
            ckpt_p.mkdir(parents=True, exist_ok=True)

        iter_s = "final" if iter is None else f"{iter:06d}"

        for gs in self.gaussian_models:
            torch.save(gs.capture(), f"{ckpt_p}/{gs.type}_{self.cfg.seq}_{iter_s}.pth")

        if self.verbose:
            logger.info(f"Saved checkpoint @ iteration {iter_s}")


def main():
    default_cfg, training_cfgs = parser.parse_args()
    logger.info(f"Running {len(training_cfgs)} experiments")

    for exp_id, training_cfg in enumerate(training_cfgs):
        logger.info(f"Running experiment #{exp_id}")

        cfg, exp = parser.init_experiment(
            default_cfg=default_cfg, training_cfg=training_cfg
        )

        common.torch_utils.reset_all_seeds(cfg.seed)
        
        # use try-catch to ensure that all experiments are executed
        try:
            logger.info("Stage 0: Initialize HO-dataset, HumanGS & ObjectGS")
            gs_trainer = GaussianTrainer(cfg=cfg, exp=exp)

            logger.info("Stage 1: Learning Appearance")
            gs_trainer.stage = "appearance"
            gs_trainer.loss.compute_ho_loss = False

            logger.info("--- 1.1: Human Appearance")
            gs_trainer.render_mode = "h"
            gs_trainer.gaussian_models = [gs_trainer.human_gs]
            gs_trainer.loss.compute_o_loss = False
            gs_trainer.loss.compute_h_loss = True
            gs_trainer.train()

            logger.info("--- 1.2: Object Appearance")
            gs_trainer.render_mode = "o"
            gs_trainer.gaussian_models = [gs_trainer.object_gs]
            gs_trainer.loss.compute_o_loss = True
            gs_trainer.loss.compute_h_loss = False
            gs_trainer.train()

            extractor = utils.mesh.GaussianExtractor(
                seq_id=cfg.seq,
                gaussians=gs_trainer.object_gs,
                cam=gs_trainer.train_dataset.cam,
                bg_color=gs_trainer.bg_color,
            )
            extractor.reconstruction(out_p=Path(cfg.log_dir))

            logger.success(f"Appearance Stage Done")
            gs_trainer.render_mode = "ho"
            gs_trainer.gaussian_models = [gs_trainer.object_gs, gs_trainer.human_gs]
            gs_trainer.animate()

            # --- evaluate appearance stage
            if cfg.eval:
                logger.error("Evaluation Not Yet Implemented!")

            # POSE OPTIMIZATION STAGE
            logger.info("Stage 2: Optimizing Poses")
            """
            gs_trainer.stage = "pose"
            gs_trainer.render_mode = "ho"  # render h/o jointly to optimize pose!
            gs_trainer.cfg.train.iterations = 30_000
            gs_trainer.loss.compute_ho_loss = True
            gs_trainer.loss.compute_h_loss = False
            gs_trainer.loss.compute_o_loss = False
            for _gs in gs_trainer.gaussian_models:
                _gs.freeze_all()
                # optimize only for object pose
                if _gs.type == "object":
                    _gs.defrost("obj_trans", v=args.train.verbose)
                    _gs.defrost("obj_rot", v=args.train.verbose)
                    _gs.defrost("obj_scale", v=args.train.verbose)
                    # set learning rates for object pose params
                    for g in _gs.optimizer.param_groups:
                        if g["name"].startswith("obj_"):
                            g["lr"] = 0.01

            overwrite poor init poses w/ reasonable pose priors (prev. frame)
            with torch.no_grad():
                for i in gs_trainer.train_dataset.frames_w_bad_init_poses:
                    # TODO: handle case if i == 0
                    gs_trainer.object_gs.obj_rot[i].copy_(
                        gs_trainer.object_gs.obj_rot[i - 1]
                    )
                    gs_trainer.object_gs.obj_trans[i].copy_(
                        gs_trainer.object_gs.obj_trans[i - 1]
                    )

            gs_trainer.animate(iter=1)
            gs_trainer.train()
            """
            logger.error("Pose Optimization Not Yet Implemented!")
            # --- evaluate HO optimization stage
            if cfg.eval:
                logger.error("Evaluation Not Yet Implemented!")

            gs_trainer = None
            utils.general.clean(Path(cfg.log_dir), v=cfg.train.verbose)
        except Exception as e:
            import traceback

            logger.error(e)
            print(traceback.format_exc())
            logger.info("Continuing with next experiment...")


if __name__ == "__main__":
    main()
