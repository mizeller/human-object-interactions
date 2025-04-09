import cv2
import torch
import joblib
import argparse
import numpy as np
from smplx import SMPLX
from pathlib import Path
from loguru import logger
import pytorch_lightning as pl
from omegaconf import OmegaConf

# local
from visualize_fits_gloss import visualize
from utils import constants, video, debug
from alignment.pl_module.ho import HOModule

torch.set_float32_matmul_precision("medium")


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, num_iter):
        self.num_iter = num_iter

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        return idx


def project2d(v3d, cam_extrinsics, cam_focal, cam_center) -> np.ndarray:
    """Perspective Projection: 3D vertices in world coords. -> 2D points in pixel coords.

    Args:
        v3d: [N, V, 3] array of 3D vertices (N=num frames, V=num vertices)
        cam_extrinsics: [4, 4] or [N, 4, 4] camera extrinsic matrix
        cam_focal: float or [N,] focal length
        cam_center: [2,] or [N, 2] camera center

    Returns:
        j2d: [N, J, 2] array of 2D joint positions in pixels
    """
    if isinstance(v3d, torch.Tensor):
        v3d = v3d.cpu().detach().numpy()

    # [N, V, 4]
    v3d_homo = np.pad(v3d, ((0, 0), (0, 0), (0, 1)), constant_values=1)

    # [N, 4, 4]
    N, _, _ = v3d.shape
    if cam_extrinsics.ndim == 2:
        cam_extrinsics = np.tile(cam_extrinsics[None], (N, 1, 1))

    # to camera coords.
    v3d_cam = np.einsum("nij,nkj->nki", cam_extrinsics, v3d_homo)
    v3d_cam = v3d_cam[:, :, :3]  # [N, V, 3]

    # perspective projection
    x = v3d_cam[:, :, 0] / v3d_cam[:, :, 2]  # [N, V]
    y = v3d_cam[:, :, 1] / v3d_cam[:, :, 2]  # [N, V]

    # Handle broadcasting of camera parameters
    if np.isscalar(cam_focal):
        cam_focal = np.full(N, cam_focal)

    if isinstance(cam_center, list):
        cam_center = np.array(cam_center)
    if cam_center.ndim == 1:
        cam_center = np.tile(cam_center[None], (N, 1))

    # cam intrinsics [N, V]
    px = cam_focal[:, None] * x + cam_center[:, 0, None]
    py = cam_focal[:, None] * y + cam_center[:, 1, None]

    # [N, V, 2]
    v2d = np.stack([px, py], axis=2)

    return v2d


def main(args):
    logger.info("Starting Human-Object Alignment Procedure.")

    # set paths
    data_p = Path.cwd() / "data" / args.seq_name
    video_p = data_p / "video.mp4"
    combined_data_p = data_p / "scratch" / "alignment" / "human_object_camera.pt"
    aligned_data_p = data_p / "data.pt"
    out_p = data_p / "scratch" / "alignment"

    # mfv stuff
    mfv_data_p = data_p / "scratch" / "mfv" / "results.pkl"
    # load camera intrinsics for full frame!
    obj_data_p: Path = data_p / "scratch" / "colmap" / "obj_data.npy"

    # visualize data pre-alignment
    logger.info("--- Merge human & object data")

    # load initial human/object poses, configs, frames,...
    mfv_data = joblib.load(mfv_data_p)
    alignment_cfg = OmegaConf.load(args.cfg_p)
    obj_data = np.load(obj_data_p, allow_pickle=True).item()
    frames = video.video2frames(video_p=video_p, return_images=True)

    if args.use_cache:
        logger.info("Loading cached data for visualization...")

        # Load the cached aligned data
        aligned_data = torch.load(aligned_data_p, map_location="cpu")
        combined_data = torch.load(combined_data_p, map_location="cpu")

        camera_extrinsics = combined_data["camera_extrinsics"].numpy()
        camera_intrinsics = combined_data["camera_intrinsics"].numpy()

        # logger.info("Visualizing cached data PRE-alignment...")
        # visualize(
        #     camera_extrinsics=camera_extrinsics,
        #     camera_intrinsics=camera_intrinsics,
        #     v3d_h=combined_data["smplx_v3d"],
        #     j3d_h=combined_data["smplx_j3d"],
        #     v3d_o=combined_data["object_v3d"],
        #     frames=frames,
        #     out_p=out_p / "vis_pre_align.mp4",
        #     side_view=args.side_view,
        #     debug=args.debug,
        # )

        logger.info("Visualizing cached data POST-alignment...")
        visualize(
            camera_extrinsics=camera_extrinsics,
            camera_intrinsics=camera_intrinsics,
            v3d_h=aligned_data["human"]["v3d"],
            j3d_h=aligned_data["human"]["j3d"],
            v3d_o=aligned_data["object"]["j3d"],
            frames=frames,
            out_p=out_p / "vis_post_align.mp4",
            side_view=True,
            debug=args.debug,
        )

        logger.success("Visualization complete!")
        return

    # CAMERA: extrinsics & intrinsics; static
    pred_cam = mfv_data["camera_world"]

    camera_extrinsics = np.eye(4)
    camera_extrinsics[:3, :3] = pred_cam["Rcw"][0]
    camera_extrinsics[:3, 3] = pred_cam["Tcw"][0]

    # NOTE: these camera intrinsics correspond to the resized video used in the MfV step
    # cam_focal = pred_cam["img_focal"]
    # cam_center = pred_cam["img_center"]
    # camera_intrinsics = np.eye(3)
    # camera_intrinsics[0,0] = cam_focal
    # camera_intrinsics[1,1] = cam_focal
    # camera_intrinsics[0,2] = cam_center[0]
    # camera_intrinsics[1,2] = cam_center[1]

    # NOTE: these camera intrinsics correspond to the original video dimension
    camera_intrinsics = np.array(obj_data["K"])
    cam_focal = camera_intrinsics[0, 0]
    cam_center = [camera_intrinsics[0, 2], camera_intrinsics[1, 2]]

    skip: bool = args.skip_combine and combined_data_p.exists()
    if not skip:
        # OBJECT: COLMAP -> world camera & projection to pixel space
        logger.info("Transforming COLMAP output to GLOSS world cam.")
        v3d_o_c = obj_data["v3d"]  # canonical obj. vertices
        w2c_mat = obj_data["w2c_mats"].numpy()
        N = w2c_mat.shape[0]
        v3d_homo = np.pad(v3d_o_c, ((0, 0), (0, 1)), constant_values=1)  # [V, 4]
        v3d_homo = np.tile(v3d_homo[None], (N, 1, 1))  # [N, V, 4]
        v3d_cam_homo = np.einsum("nij,nvj->nvi", w2c_mat, v3d_homo)  # [N, V, 4]
        v3d_o = v3d_cam_homo[:, :, :3].astype(np.float32)  # [N, V, 3]

        # compute 2D kpts; [N, V_o, 2]
        logger.info("Projecting 3D object vertices to 2D.")
        v2d_o = project2d(
            v3d=v3d_o,
            cam_extrinsics=camera_extrinsics,
            cam_focal=cam_focal,
            cam_center=cam_center,
        )

        # verify project2D is correct
        if args.debug:
            vis_frame = frames[0][..., ::-1].copy()
            for x, y in v2d_o[0]:
                x = int(np.clip(x, 0, vis_frame.shape[1] - 1))
                y = int(np.clip(y, 0, vis_frame.shape[0] - 1))
                cv2.circle(vis_frame, (x, y), radius=8, color=(0, 255, 0), thickness=-1)
            cv2.imwrite("tmp/debug_v2d_o.png", vis_frame)

        # HUMAN - Assumptions: (i) single person, (ii) person visible ∀ frames
        person = mfv_data["people"][1]

        # get verts and joints in world coords
        logger.info("Applying body parameters & pose to SMPL-X.")
        smplx = SMPLX(
            constants.SMPLX_NEUTRAL_P,
            batch_size=person["frames"].shape[0],
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
        )

        b = torch.from_numpy(person[f"smplx_world"]["shape"]).float()
        t = torch.from_numpy(person[f"smplx_world"]["trans"]).float()
        p = torch.from_numpy(person[f"smplx_world"]["pose"]).float()
        p = p.reshape(-1, 55 * 3)
        bp = p[:, 3 : 3 + 21 * 3]
        lhp = p[:, 75:120]
        rhp = p[:, 120:]
        go = p[:, :3]

        s = smplx(
            body_pose=bp,
            global_orient=go,
            betas=b,
            left_hand_pose=lhp,
            right_hand_pose=rhp,
            transl=t,
            pose2rot=True,
        )

        v3d_h = s.vertices  # N x V x 3

        # only use body + hand joints for now; for more details see:
        # https://github.com/vchoutas/smplx/issues/14 or
        # https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
        j3d_h = s.joints  # N x J x 3, J = 127
        j2d_h = project2d(
            v3d=j3d_h,
            cam_extrinsics=camera_extrinsics,
            cam_focal=cam_focal,
            cam_center=cam_center,
        )

        if args.debug:
            vis_frame = frames[0][..., ::-1].copy()
            for x, y in j2d_h[0]:
                x = int(np.clip(x, 0, vis_frame.shape[1] - 1))
                y = int(np.clip(y, 0, vis_frame.shape[0] - 1))
                cv2.circle(vis_frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
            cv2.imwrite("tmp/debug_j2d_h.png", vis_frame)

        # logger.info("Visualizing sequence in GLOSS.")
        # visualize(
        #     camera_extrinsics=camera_extrinsics,
        #     camera_intrinsics=camera_intrinsics,
        #     v3d_h=v3d_h,
        #     j3d_h=j3d_h,
        #     v3d_o=v3d_o,
        #     frames=frames,
        #     out_p=out_p / "vis_pre_align.mp4",
        #     debug=args.debug,
        #     side_view=args.side_view,
        # )

        # merge data for alignment step
        _combined_data = {
            # camera
            "camera_extrinsics": torch.Tensor(camera_extrinsics),
            # w.r.t original video dimensions!
            "camera_intrinsics": torch.Tensor(camera_intrinsics),
            # object
            "object_poses": obj_data["w2c_mats"],
            "object_converged": obj_data["converged"],
            "object_v3d_cano": torch.Tensor(obj_data["v3d"]),
            "object_v3d_colors": torch.Tensor(obj_data["v3d_rgb"]),
            "object_v3d": torch.from_numpy(v3d_o).float(),
            "object_v2d": torch.from_numpy(v2d_o).float(),
            # human
            "smplx_betas": b,
            "smplx_transl": t,
            "smplx_body_pose": bp,
            "smplx_left_hand_pose": lhp,
            "smplx_right_hand_pose": rhp,
            "smplx_global_orient": go,
            "smplx_v3d": torch.Tensor(v3d_h),
            "smplx_j3d": j3d_h,
            "smplx_j2d": torch.Tensor(j2d_h),
        }

        # combined_data = {k: v.cuda() for k, v in _combined_data.items()}

        # NOTE: for some reason, PL-training only works if I save to disk & load from disk
        torch.save(
            {k: v.cpu() for k, v in _combined_data.items()},
            combined_data_p,
        )
        logger.success(f"Saved complete scene data to: {combined_data_p}")
    logger.warning(f"Aligning scene data from: {combined_data_p}")
    combined_data = torch.load(combined_data_p, map_location="cuda")

    combined_data["frames"] = torch.from_numpy(np.array(frames))

    # set up PL module
    pl_model = HOModule(data=combined_data, conf=alignment_cfg, out_p=out_p)

    trainset = torch.utils.data.DataLoader(
        FakeDataset(alignment_cfg["num_iters"]),
        batch_size=1,
        shuffle=False,
        num_workers=111,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        gradient_clip_val=0.5,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(pl_model, trainset)

    logger.success(f"HO Alignment fitting worked")

    aligned_data = {}
    aligned_data["camera"] = {
        "extrinsics": combined_data["camera_extrinsics"].cpu(),
        "intrinsics": combined_data["camera_intrinsics"].cpu(),
    }

    pl_model = pl_model.cuda()  # after fit(), some tensors are on cpu?
    for key in pl_model.models.keys():
        out = pl_model.models[key](mask=combined_data["object_converged"])
        aligned_data[key] = {k: v.cpu() for k, v in out.items()}

    if "object" in aligned_data:
        aligned_data["object"]["v3d_cano"] = combined_data["object_v3d_cano"].cpu()
        aligned_data["object"]["v3d_rgb"] = combined_data["object_v3d_colors"].cpu()
        aligned_data["object"]["converged"] = combined_data["object_converged"].cpu()

    # save
    torch.save(aligned_data, aligned_data_p)

    logger.success(f"Saved aligned data to {aligned_data_p}")

    logger.info("Visualizing Aligned Data...")

    # camera_extrinsics = aligned_data["camera"]["extrinsics"].numpy().astype(np.float32)
    # camera_intrinsics = aligned_data["camera"]["intrinsics"].numpy().astype(np.float32)

    if args.debug:
        logger.info("Aligned `data.pt` contains:")
        debug.log(aligned_data)

        vis_frame = frames[0].copy()
        w = vis_frame.shape[1]
        h = vis_frame.shape[0]
        for x, y in aligned_data["object"]["j2d"][0]:  # red
            x = int(torch.clip(x, 0, w - 1).item())
            y = int(torch.clip(y, 0, h - 1).item())
            cv2.circle(vis_frame, (x, y), radius=8, color=(255, 0, 0), thickness=-1)
        for x, y in aligned_data["human"]["j2d"][0]:  # green
            x = int(torch.clip(x, 0, w - 1).item())
            y = int(torch.clip(y, 0, h - 1).item())
            cv2.circle(vis_frame, (x, y), radius=8, color=(0, 255, 0), thickness=-1)
        for x, y in aligned_data["human"]["v2d"][0]:  # blue
            x = int(torch.clip(x, 0, w - 1).item())
            y = int(torch.clip(y, 0, h - 1).item())
            cv2.circle(vis_frame, (x, y), radius=4, color=(0, 0, 255), thickness=-1)

        cv2.imwrite("./tmp/debug_alignemt.png", vis_frame[..., ::-1])
        return

    visualize(
        camera_extrinsics=camera_extrinsics,
        camera_intrinsics=camera_intrinsics,
        v3d_h=aligned_data["human"]["v3d"],
        j3d_h=aligned_data["human"]["j3d"],
        v3d_o=aligned_data["object"]["j3d"],
        frames=frames,
        out_p=out_p / "vis_post_align.mp4",
        side_view=True,  # visualize side-view per default!
        debug=args.debug,
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--cfg_p",
        type=str,
        default="configs/generic.yaml",
        help="Path to the config file for the HO alignment.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cached data for visualization only",
    )
    parser.add_argument(
        "--skip_combine",
        action="store_true",
        help="Don't combine HO data anew - just re-run alignment!",
    )
    parser.add_argument(
        "--side_view",
        action="store_true",
        help="Render side-view as well.",
    )

    args = parser.parse_args()
    main(args)
