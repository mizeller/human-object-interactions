import torch
import trimesh
from tqdm import tqdm
from typing import List
from pathlib import Path
from loguru import logger

from src import constants
from src.utils import camera, video, debug


class HumanObjectData(torch.utils.data.Dataset):
    """DataLoader for HumanObject videos."""

    def __init__(self, cfg):
        self.device = constants.device

        # set paths
        seq_p: Path = Path.cwd() / "data" / cfg.seq
        ps = {
            "masks": seq_p / "masks.mp4",
            "video": seq_p / "video.mp4",
            # "normals": seq_p / "normals.mp4",
            "data": seq_p / "data.pt",
        }

        # sanity check
        for v in ps.values():
            assert v.exists(), v

        # load frames & masks
        logger.info(f"Loading frames & masks")
        self.frames: torch.Tensor = video.video2frames(
            video_p=ps["video"], return_images=True, to_tensor=True
        )
        self.masks: torch.Tensor = video.video2frames(
            video_p=ps["masks"], return_images=True, to_tensor=True
        )
        # self.normals: torch.Tensor = video.video2frames(
        #     video_p=ps["normals"], return_images=True, to_tensor=True
        # )

        # ensure valid
        assert len(
            torch.unique(self.masks[0]) == 3
        ), "Loaded mask with more than 3 unique values!?"
        assert len(self.frames) == len(
            self.masks
        ), "video.mp4 & masks.mp4 have different number of frames?!"

        self.num_frames = len(self.frames)

        # load data
        data = torch.load(ps["data"], map_location="cuda")

        if cfg.train.verbose:
            debug.log(data)

        # ensure data is valid
        for k, v in data["human"].items():
            assert (
                v.shape[0] == self.num_frames
            ), f"Expected {k} to have {self.num_frames} entries; found {v.shape[0]}"

        for k, v in data["object"].items():
            if k not in ["obj_scale", "v3d_cano", "v3d_rgb"]:
                assert (
                    v.shape[0] == self.num_frames
                ), f"Expected {k} to have {self.num_frames} entries; found {v.shape[0]}"

        self.init_cam(data=data["camera"])
        self.init_smplx(data=data["human"])
        self.init_object(data=data["object"])
        self.load_data_to_cuda(sample=cfg.sample)

        # HOLD dataset compatibility...
        self.frames_w_bad_init_poses: List[int] = []

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int):
        if self.cached_data is not None:
            return self.cached_data[idx]
        return self.get(idx)

    def load_data_to_cuda(self, n: int = None, sample: int = None) -> None:
        """Load data to CUDA. Only load dataset partially for memory/efficeny reasons (i.e when debugging.)"""

        # sample every Nth frame
        s = 1 if sample is None else sample
        frame_indices = range(0, self.__len__(), s)
        if n:
            logger.warning(
                f"Loading only the first {n}/{self.__len__()} frames to CUDA!"
            )
            # Take only first n=15 frames
            selected_indices = list(frame_indices)[:n]
        else:
            selected_indices = list(frame_indices)

        # NOTE: "invalid" frames are loaded here as well; should potentially handle these here though
        self.cached_data: List[dict] = []
        for i in tqdm(selected_indices, desc="Loading data to GPU"):
            data = self.get(i)
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to("cuda")
            self.cached_data.append(data)

        self.num_frames = len(self.cached_data)

        return

    def get(self, idx: int) -> dict:
        """Does not load from cached data (i.e GPU)"""
        out_dict = {}
        out_dict["frame_idx"] = idx

        # smplx
        out_dict["smplx_transl"] = self.smplx_transl[idx]
        out_dict["smplx_global_orient"] = self.smplx_global_orient[idx]
        out_dict["smplx_betas"] = self.smplx_betas[idx]
        out_dict["smplx_body_pose"] = self.smplx_body_pose[idx]
        out_dict["smplx_left_hand_pose"] = self.smplx_left_hand_pose[idx]
        out_dict["smplx_right_hand_pose"] = self.smplx_right_hand_pose[idx]

        # object
        out_dict["obj_valid"] = self.obj_valid[idx]  # ∃ valid object pose?
        out_dict["obj_pose"] = self.obj_w2c[idx]
        out_dict["obj_scale"] = self.obj_scale

        # images/masks
        _img = self.frames[idx]  # gt pixels incl. bg
        msk = self.masks[idx]  # segmentation mask
        # _normal = self.normals[idx]  # normals
        # msk = msk.repeat(3, 1, 1)  # ensure three channels

        foreground_mask = (msk != 0).bool()
        img = _img * foreground_mask / 255  # gt pixels excl. bg, normalized
        # normal = _normal * foreground_mask / 255  # gt pixels excl. bg, normalized

        out_dict["gt_img"] = img  # normalized; h/o pixels only
        out_dict["gt_msk"] = msk  # NOT normalized! uint better for seg. mask!
        # out_dict["gt_normal"] = normal

        return out_dict

    def init_object(self, data) -> None:
        logger.info("Init Object")
        # SfM model from COLMAP
        self.init_pcd: trimesh.Pointcloud = trimesh.PointCloud(
            vertices=data["v3d_cano"].detach().cpu().numpy(),
            colors=data["v3d_rgb"].detach().cpu().numpy(),
        )

        # frames for which COLMAP converged
        self.obj_valid: torch.Tensor = data["converged"]

        # object params after alignment
        self.obj_scale: float = data["obj_scale"].item()
        self.obj_w2c: torch.Tensor = data["w2c_mats"]  # (N, 4, 4)

        return

    def init_cam(self, data) -> None:
        logger.info("Init Camera")
        H = self.frames[0].shape[1]
        W = self.frames[0].shape[2]
        logger.info(f"HxW: {H}x{W}")
        self.cam = camera.Camera(
            cam_id=0,
            image_size=(H, W),
            cam_intrinsics=data["intrinsics"],
            world_view_transform=data["extrinsics"],  # world2cam
        )
        return

    def init_smplx(self, data) -> None:
        logger.info("Init SMPLX")

        self.smplx_betas = data["betas"]  # b, (N, 10)
        self.smplx_transl = data["smplx_transl"]  # t, (N, 3)
        self.smplx_body_pose = data["smplx_body_pose"]  # bp, (N, 63)
        self.smplx_left_hand_pose = data["smplx_left_hand_pose"]  # lhp, (N, 45)
        self.smplx_right_hand_pose = data["smplx_right_hand_pose"]  # rhp, (N, 45)
        self.smplx_global_orient = data["smplx_global_orient"]  # go, (N, 3)

        # NOTE: the .smplx contains the same information as the aligned data
        # from smplcodec import SMPLCodec

        # smplx_p = "smplx": seq_p / "scratch" / "mfv" / "mfv_sub-1.smpl",
        # smplx_data = SMPLCodec.from_file(smplx_p)
        # print(
        #     torch.allclose(
        #         torch.tensor(smplx_data.body_pose.reshape(self.num_frames, -1)),
        #         torch.cat([self.smplx_global_orient, self.smplx_body_pose], dim=1),
        #     )
        # )
        # print(
        #     torch.allclose(
        #         torch.tensor(smplx_data.shape_parameters),
        #         self.smplx_betas[0],
        #         rtol=1e-5,
        #         atol=1e-8,
        #     )
        # )
        # print(
        #     torch.allclose(
        #         torch.tensor(smplx_data.right_hand_pose.reshape(self.num_frames, -1)),
        #         self.smplx_right_hand_pose,
        #     )
        # )
        # print(
        #     torch.allclose(
        #         torch.tensor(smplx_data.left_hand_pose.reshape(self.num_frames, -1)),
        #         self.smplx_left_hand_pose,
        #     )
        # )
        # print(
        #     torch.allclose(torch.tensor(smplx_data.body_translation), self.smplx_transl)
        # )

        return
