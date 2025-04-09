import torch
from typing import Union
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
    mesh_normal_consistency,  # TODO: maybe add as well to replace current normal loss
    chamfer_distance,
)

from pytorch3d.structures import Pointclouds
from loguru import logger

from src.utils import rotations
from src.losses.loss_utils import (
    get_constant_distance_loss,
    get_ssim_loss,
    get_l1_loss,
    get_lpips_loss,
    get_3D_smoothness_loss,
    get_pseudo_depth_loss,
    get_sem_loss,
    get_emd_loss,
)

from src.renderer.render_manager import RenderManager
from src.model.gaussian_model import GaussianModel


class Loss(torch.nn.Module):
    def __init__(self, cfg: dict, verbose: bool = False):
        super(Loss, self).__init__()
        self.verbose: bool = verbose
        for k, v in cfg.items():
            setattr(self, k, v)

        self.compute_h_loss: bool = False
        self.compute_o_loss: bool = False
        self.compute_ho_loss: bool = False

    def forward(
        self,
        renders: RenderManager,
        human_gs: Union[GaussianModel, None],
        object_gs: Union[GaussianModel, None],
        rnd_idx: int,  # current frame index
        iter: int,  # current iteration index
        visibility_filter_h: Union[torch.Tensor, None],
    ):
        torch.cuda.empty_cache()

        loss_dict = {}

        # extract relevant renders for loss computation from render_pkg
        render_data = {
            mode: {
                key: renders.get(mode, key)
                for key in [
                    "gt_img",
                    "pred_img",
                    "gt_msk",
                    "pred_msk",
                    # "gt_normal",
                    # "pred_normal",
                ]
            }
            for mode in ["o", "h", "ho"]
            if getattr(self, f"compute_{mode}_loss")
        }

        # NOTE: ho/gt_msk_seg -> segmentation mask (silhouette loss)
        #       ho/gt_msk     -> binary mask (mask/RGB losses)
        if self.compute_ho_loss:
            ho_seg_msk = render_data["ho"]["gt_msk"]
            render_data["ho"]["gt_msk_seg"] = ho_seg_msk
            render_data["ho"]["gt_msk"] = torch.where(ho_seg_msk != 0, 1.0, 0.0)

        # general losses (L1, SSIM, LPIPS, GS Mask)
        general_losses = self.compute_pixel_losses(render_data)
        loss_dict.update(general_losses)

        if self.compute_h_loss:
            hand_losses = self.compute_hand_losses(human_gs, visibility_filter_h)
            loss_dict.update(hand_losses)

            twoD_losses = self.compute_2DGS_losses(renders, iter, mode="h")
            loss_dict.update(twoD_losses)

        if self.compute_o_loss:
            twoD_losses = self.compute_2DGS_losses(renders, iter, mode="o")
            loss_dict.update(twoD_losses)

            # ensure object gaussians to be isotropic
            if hasattr(self, "isotropic_reg_w") and self.isotropic_reg_w > 0:
                scale_condition_nr: float = 0.4
                max_scale = torch.max(object_gs.get_scaling, dim=1)[0]
                min_scale = torch.min(object_gs.get_scaling, dim=1)[0]
                isotropic_loss = torch.mean(
                    (min_scale / (max_scale + 1e-8) - scale_condition_nr) ** 2
                )
                loss_dict["isotropic_reg"] = isotropic_loss * self.isotropic_reg_w

        if False:  # self.compute_ho_loss:
            v3d = None
            # HOLD ho loss (ensure, 3D vertices are smooth)
            if hasattr(self, "smoothness_w") and self.smoothness_w > 0:
                ho_smoothness_loss = get_3D_smoothness_loss(v3d=v3d)
                loss_dict["smoothness"] = ho_smoothness_loss * self.smoothness_w

            if hasattr(self, "approx_depth_w") and self.approx_depth_w > 0:
                # Since we're looking at a combined H/O scene from a monocular view,
                # both H/O should be in the same depth range.
                # raise NotImplementedError("Approximate depth loss not implemented yet.")
                loss_dict["approx_depth"] = self.approx_depth_w * get_pseudo_depth_loss(
                    v3d=v3d, thresh=self.threshold_depth
                )

            if hasattr(self, "gap") and self.gap_w > 0:
                # since the object is mostly tightly grasped, the gap between the H/O centroids
                # should be constant for all frames
                loss_dict["constant_gap"] = self.gap_w * get_constant_distance_loss(
                    v3d=v3d, thresh=self.threshold_gap
                )

            if hasattr(self, "contact_w") and self.contact_w > 0:
                # compute 1-directional chamfer distance between finger tips and object
                # point cloud; this is a very crude approximation of contact loss

                v3d_obj = Pointclouds(points=v3d[rnd_idx]["object"].unsqueeze(0))
                # v3d_hnd = Pointclouds(v3d[rnd_idx]["hand"].unsqueeze(0))
                v3d_finger_tips = human_gs.current_mesh.verts_list()[0][
                    human_gs.finger_tips
                ].unsqueeze(0)

                finger_tips = Pointclouds(points=v3d_finger_tips)

                loss_dict["contact"] = (
                    self.contact_w
                    * chamfer_distance(
                        x=finger_tips, y=v3d_obj, single_directional=True
                    )[0]
                )

        # total loss
        total_loss = sum(loss_dict.values())
        loss_dict["loss"] = total_loss

        return total_loss, loss_dict

    def compute_pixel_losses(self, render_data: dict):
        """Compute pixel based losses: L1, SSIM, LPIPS, (Mask)
        NOTE: ensure that the function signature follows: function(pred_img, gt_img, gt_msk)
        NOTE: ensure that the weight attribute in the cfg file is named like: {loss_name}_w
        """

        loss_dict = {}
        loss_types = [
            ("l1", get_l1_loss),
            ("mask", get_l1_loss),  # l1 loss between gs_mask and gt_mask
            ("ssim", get_ssim_loss),
            ("lpips", get_lpips_loss),
        ]

        for loss_name, loss_func in loss_types:
            weight: str = f"{loss_name}_w"
            if hasattr(self, weight) and getattr(self, weight) > 0.0:
                loss = 0.0
                for mode, data in render_data.items():
                    if self.verbose:
                        logger.info(f"Computing {loss_name} loss for mode: {mode}")

                    if loss_name == "mask":
                        # TODO: fix mask loss. DONT use L1 loss here & consider background as well!
                        loss += loss_func(
                            data["pred_msk"],
                            data["gt_msk_seg"] if mode == "ho" else data["gt_msk"],
                        )
                    else:
                        loss += loss_func(
                            data["pred_img"], data["gt_img"], data["gt_msk"]
                        )

                loss_dict[loss_name] = getattr(self, weight) * loss

        return loss_dict

    def compute_hand_losses(self, human_gs, visibility_filter_h):
        _loss_dict = {}

        if hasattr(self, "xyz_w") and self.xyz_w > 0:
            if self.verbose:
                logger.info(f"Computing xyz hand-loss.")
            _loss_dict["xyz"] = (
                self.xyz_w
                * torch.nn.functional.relu(
                    human_gs._xyz[visibility_filter_h].norm(dim=1) - self.threshold_xyz
                ).mean()
            )

        if hasattr(self, "scale_w") and self.scale_w > 0:
            if self.verbose:
                logger.info(f"Computing scale hand-loss.")
            _loss_dict["scale"] = (
                self.scale_w
                * torch.nn.functional.relu(
                    torch.exp(human_gs._scaling[visibility_filter_h])
                    - self.threshold_scale
                )
                .norm(dim=1)
                .mean()
            )

        if hasattr(self, "normal_align_w") and self.normal_align_w > 0:
            if self.verbose:
                logger.info(f"Computing normal hand-loss.")
            rot_q = human_gs.out["rotq"]
            rotmat = rotations.quaternion_to_matrix(rot_q)
            gaussian_normals = rotmat[:, :, 2]
            face_normals = human_gs.out["normals"]
            _loss_dict["normal"] = self.normal_align_w * (
                1 - torch.cosine_similarity(gaussian_normals, face_normals).mean()
            )

        if hasattr(self, "vert_offset_w") and self.vert_offset_w > 0:
            if self.verbose:
                logger.info(f"Computing vertex-offset hand-loss.")
            if human_gs.cfg.normal_displacement:
                _loss_dict["vertex_offset"] = self.vert_offset_w * torch.sum(
                    human_gs.vert_offsets**2.0
                )
            else:
                _loss_dict["vertex_offset"] = self.vert_offset_w * torch.sum(
                    torch.norm(human_gs.vert_offsets, dim=1) ** 2.0
                )

        if hasattr(self, "laplacian_w") and self.laplacian_w > 0:
            if self.verbose:
                logger.info(f"Computing smoothness hand-loss (laplacian).")
            _loss_dict["laplacian"] = self.laplacian_w * mesh_laplacian_smoothing(
                human_gs.current_mesh
            )

        return _loss_dict

    def compute_2DGS_losses(self, renders: RenderManager, iter: int, mode: str):
        """Compute normal & distortion (from depth) loss following 2DGS.
        - normal consisteny loss: https://github.com/nerfstudio-project/gsplat/blob/bd64a47414e182dc105c1a2fdb6691068518d060/examples/simple_trainer_2dgs.py#L616
        """
        _loss_dict = {}
        if (
            hasattr(self, "normal_w")
            and self.normal_w > 0
            and self.normal_start_iter <= iter
        ):
            normal_error = (
                1
                - (
                    renders.get(mode, "normals")
                    * renders.get(mode, "normals_from_depth")
                ).sum(dim=0)
            )[None]
            _loss_dict["normal"] = self.normal_w * normal_error.mean()

        if (
            hasattr(self, "distortion_w")
            and self.distortion_w > 0
            and self.distortion_start_iter <= iter
        ):
            _loss_dict["distortions"] = (
                self.distortion_w * renders.get(mode, "distortions").mean()
            )

        if (
            hasattr(self, "dn_normal_w")
            and self.dn_normal_w > 0
            and self.dn_normal_start_iter <= iter
        ):
            # https://github.com/maturk/dn-splatter/blob/249d52c4bb14b7bf6dd18d7d66099a36eac2ee78/dn_splatter/regularization_strategy.py#L140
            # https://github.com/maturk/dn-splatter/blob/249d52c4bb14b7bf6dd18d7d66099a36eac2ee78/dn_splatter/losses.py#L166

            # remap normals from [0,1] -> [-1,1]
            gt = renders.get(mode, "gt_normal") * 2 - 1
            pred = renders.get(mode, "normals_from_depth")

            # L1 loss
            # import torch.nn.functional as F
            # _loss = F.l1_loss(pred, gt)  # torch.abs(pred - gt).mean()
            _loss = (1 - (pred * gt).sum(dim=0))[None]

            # weighted loss
            _loss_dict["dn_normal"] = self.dn_normal_w * _loss.mean()

        if (
            hasattr(self, "dn_normal_smooth_w")
            and self.dn_normal_w > 0
            and self.dn_normal_smooth_start_iter <= iter
        ):
            # normal smooth loss: https://github.com/maturk/dn-splatter/blob/249d52c4bb14b7bf6dd18d7d66099a36eac2ee78/dn_splatter/regularization_strategy.py#L143
            # https://github.com/maturk/dn-splatter/blob/249d52c4bb14b7bf6dd18d7d66099a36eac2ee78/dn_splatter/losses.py#L279

            pred = renders.get(mode, "pred_normal")

            h_diff = pred[:, :-1, :] - pred[:, 1:, :]
            w_diff = pred[:-1, :, :] - pred[1:, :, :]
            _normal_smooth_loss = torch.mean(torch.abs(h_diff)) + torch.mean(
                torch.abs(w_diff)
            )

            _loss_dict["dn_normal_smooth"] = (
                self.dn_normal_smooth_w * _normal_smooth_loss
            )

        return _loss_dict
