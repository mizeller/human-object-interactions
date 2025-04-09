import cv2
import torch
import trimesh
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from loguru import logger
from typing import List, Tuple
from easydict import EasyDict as edict
from torchvision.ops import masks_to_boxes

to_tensor = torchvision.transforms.ToTensor()

# local
from src.utils import graphics, image, camera
from src.constants import SEGM_IDS


def get_data_splits(scene_length: int) -> Tuple[List[int], List[int], List[int]]:
    """Return three list of indeces, for training, validation and testing."""
    num_val = scene_length // 5
    length = int(1 / (num_val) * scene_length)
    offset = length // 2
    val_list = list(range(scene_length))[offset::length]
    train_list = list(set(range(scene_length)) - set(val_list))
    test_list = val_list[: len(val_list) // 2]
    val_list = val_list[len(val_list) // 2 :]
    assert len(train_list) > 0
    assert len(test_list) > 0
    assert len(val_list) > 0
    return train_list, val_list, test_list


class HOLDataset(torch.utils.data.Dataset):

    def __init__(self, seq: str, split: str = "train", dbg_frames: int = None):
        self.verbose: bool = False
        seq_p: Path = Path(f"./data/{seq}")
        # the following paths are expected to exist
        self.data_p: Path = seq_p / "data.npy"
        # original v3d of object (unscaled, incl. color)
        self.obj_ply_p: Path = seq_p / "v3d_orig_obj.ply"
        self.img_dir: Path = seq_p / "images"
        self.msk_dir: Path = seq_p / "masks"
        # double check paths to make sure they exist
        assert seq_p.exists(), logger.error(f"Path {seq_p} does not exist")
        assert self.data_p.exists(), logger.error(f"Path {self.data_p} does not exist")
        assert self.obj_ply_p.exists(), logger.error(
            f"Path {self.obj_ply_p} does not exist"
        )
        assert self.img_dir.exists(), logger.error(
            f"Path {self.img_dir} does not exist"
        )
        assert self.msk_dir.exists(), logger.error(
            f"Path {self.msk_dir} does not exist"
        )

        self.img_list = sorted(self.img_dir.glob("*"))
        self.msk_list = sorted(self.msk_dir.glob("*"))
        assert len(self.img_list) == len(
            self.msk_list
        ), "Mismatch in img & msk lists length! Check pre-processing..."

        seq_data: edict = edict(np.load(self.data_p, allow_pickle=True).item())
        self.set_cam_properties(seq_data.cam_info)

        self.num_frames: int = dbg_frames if dbg_frames else len(self.img_list)
        self.split: str = split
        self.train_split = list(range(self.num_frames))
        # if self.split != "anim":
        #     self.train_split, self.val_split, self.test_split = get_data_splits(
        #         scene_length=self.num_frames
        #     )

        self.entities = list(seq_data.entities.keys())

        for entity in self.entities:
            if entity == "object":
                self.set_obj_properties(seq_data.entities[entity])
            elif entity == "right":
                self.set_hnd_properties_r(seq_data.entities[entity])
            elif entity == "left":
                self.set_hnd_properties_l(seq_data.entities[entity])

        # filter the COLMAP point-cloud based on reprojection-inlier count!
        bad_frames, obj_pc_cano = self.analyze_preprocessing(reproj_threshold=0.45)

        # skip these frames when learning appearance model
        self.frames_w_bad_init_poses: List[int] = bad_frames

        # convert trimesh.Pointcloud to BasicPointCloud
        # TODO: change this, s.t. obj_pc_cano is used directly & deprecate BasicPointCloud
        self.init_pcd = obj_pc_cano
        # self.init_pcd: graphics.BasicPointCloud = graphics.BasicPointCloud(
        #     points=torch.tensor(obj_pc_cano.vertices, dtype=torch.float32),
        #     colors=torch.tensor(obj_pc_cano.colors[:, :3]) / 255,
        #     normals=np.zeros_like(obj_pc_cano.vertices),
        #     faces=None,
        # )

        if not hasattr(self, "cached_data"):
            self.load_data_to_cuda()

    def __len__(self):
        if self.split == "train":
            return len(self.train_split)
        elif self.split == "val":
            return len(self.val_split)
        elif self.split == "anim":
            return self.num_frames

    def __getitem__(self, idx):
        if self.cached_data is None:
            return self.get_single_item(idx, is_src=True)
        else:
            return self.cached_data[idx]

    def analyze_preprocessing(
        self,
        inlier_threshold: float = 0.3,  # find good frames
        reproj_threshold: float = 0.5,  # find good v3d
        save_animation: bool = False,
    ) -> Tuple[List[int], trimesh.PointCloud]:
        """
        Analyze the pre-processed data from COLMAP.
        Find frames with poor initial poses & track point reliability across frames.
        A frame is considered good if more than inlier_threshold % of re-projected vertices are
        inside the object segmentation mask.

        A 3D vertex is considered good if it reprojects onto the segemtation mask for more than 50% of frames.

        Returns:
            Tuple containing:
            - List of bad frame indices
            - Canonical Object Model containing only reliable 3D vertices
        """
        bad_frames: List[int] = []

        # noisy canonical pcloud -> scale w/ obj_scale
        obj_pc_cano: trimesh.Pointcloud = trimesh.load(self.obj_ply_p).apply_scale(
            self.obj_scale
        )

        v3d = torch.Tensor(obj_pc_cano.vertices)

        P = self.K @ self.obj_w2c[:, :3, :]  # projection matrix

        v3d_homo = torch.cat(
            [
                v3d,
                torch.ones((v3d.shape[0], 1), dtype=v3d.dtype, device=v3d.device),
            ],
            dim=1,
        )
        # v2d_homo = (P @ v3d_homo.T).T
        v2d_homo = (P @ v3d_homo.T).mT.permute(1, 2, 0)
        v2d = v2d_homo[:, :2] / v2d_homo[:, 2:]

        point_inlier_counts = np.zeros(len(v3d))

        if save_animation:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30
            first_mask = cv2.imread(str(self.msk_list[0]))
            h, w = first_mask.shape[:2]
            out = cv2.VideoWriter("tmp/output.mp4", fourcc, fps, (w, h))

        # always use all frames for filtering SfM model!
        total_frames = len(self.msk_list)  # self.obj_w2c.shape[0]
        for frame_id in range(total_frames):
            # load binary object mask
            msk_p: Path = self.msk_list[frame_id]
            msk = cv2.imread(str(msk_p))
            msk_binary = (msk == 50).astype(np.uint8)[:, :, 0]
            h, w = msk.shape[:2]

            # extract 2D vertices for this frame
            pts = v2d[:, :, frame_id].cpu().numpy()
            pts = np.clip(pts, [0, 0], [w - 1, h - 1])

            # compute valid points (inside image)
            valid_mask = (
                (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            )
            valid_pts = pts[valid_mask]

            # valid points -> inliers + outliers
            inliers_mask = msk_binary[
                valid_pts[:, 1].astype(int), valid_pts[:, 0].astype(int)
            ]

            # count #inliers incl. invalid points for consistency
            all_inliers = msk_binary[pts[:, 1].astype(int), pts[:, 0].astype(int)]
            num_inliers = np.sum(all_inliers)
            num_points = len(pts)
            inlier_percentage = (num_inliers / num_points) * 100

            # Update inlier counts for each point
            point_inlier_counts[all_inliers > 0] += 1

            # consider frame w/ less than threshold % inliers bad!
            if inlier_percentage <= inlier_threshold * 100:
                bad_frames.append(frame_id)

            # Draw visualization
            if save_animation:
                inlier_pts = valid_pts[inliers_mask > 0]
                outlier_pts = valid_pts[inliers_mask == 0]
                image.draw_circles(msk, inlier_pts, (0, 255, 0), radius=1)
                image.draw_circles(msk, outlier_pts, (0, 0, 255), radius=1)
                text = f"Inliers: {num_inliers}/{num_points} ({inlier_percentage:.1f}%)"
                cv2.putText(
                    msk, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                )
                cv2.putText(
                    msk, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1
                )
                out.write(msk)

        if save_animation:
            out.release()

        point_inlier_percentages = point_inlier_counts / total_frames * 100
        good_v3d = np.where(point_inlier_percentages > (reproj_threshold * 100))[
            0
        ].tolist()

        # use filtered SfM model for objectGS initialization
        obj_pc_filtered = trimesh.PointCloud(
            vertices=obj_pc_cano.vertices[good_v3d], colors=obj_pc_cano.colors[good_v3d]
        )

        # compare the two...
        # obj_pc_cano.export("tmp/v3d_original.ply")
        # obj_pc_filtered.export("tmp/v3d_filtered.ply")

        if self.verbose:
            logger.warning(f"Found {len(bad_frames)} frames w/ poor init object pose.")

        return bad_frames, obj_pc_filtered

    def set_avg_hand_color(self):
        """Compute average hand color for HandGS initialization."""
        # NOTE: again...only right-hand support for now
        accum_hand_color = torch.zeros((3,))

        for i in range(self.num_frames):
            msk = cv2.imread(self.msk_list[i])
            img = cv2.imread(self.img_list[i])
            hand_msk = (msk == SEGM_IDS["right"]).astype(np.uint8)
            hand_msk = hand_msk[:, :, 0]
            masked_img = cv2.bitwise_and(img, img, mask=hand_msk)
            non_zero_pixels = masked_img[np.nonzero(hand_msk)]
            accum_hand_color += torch.from_numpy(np.mean(non_zero_pixels, axis=0))

        avg_hand_color = accum_hand_color / self.num_frames  # BGR

        self.avg_hand_color = avg_hand_color[[2, 1, 0]] / 255  # BGR -> RGB

    def set_cam_properties(self, cam_info):
        """Define the (monocular) camera properties, used to capture this dataset."""
        znear, zfar = 0.01, 100.0

        cam_info: edict = edict(cam_info)
        self.K: torch.Tensor = torch.Tensor(cam_info.K)  # 3x3
        self.Rt_cam: torch.Tensor = torch.Tensor(cam_info.Rt_cam)  # 4x4

        self.cam = camera.Camera(
            uid=0,
            image_size=(self.height, self.width),
            world_view_transform=self.Rt_cam.cuda(),
            cam_intrinsics=self.K.cuda(),
        )

        # this is wrong in some HO3D 'ground truth' instances
        self.height, self.width, _ = cv2.imread(self.img_list[0]).shape

        self.projection_matrix = graphics.getProjectionMatrix(
            znear=znear, zfar=zfar, fovX=self.fovx, fovY=self.fovy
        ).transpose(0, 1)

        # sequences pre-processed w/ latest code contain two List[Path]
        #  objects in `cam_info`; use them to get the img_list and msk_list!!
        if "valid_frames" in cam_info.keys() and "valid_masks" in cam_info.keys():
            self.img_list = cam_info.valid_frames
            self.msk_list = cam_info.valid_masks

        return

    def set_obj_properties(self, obj_data: dict) -> None:
        """load obj data from pre-processing"""
        obj_data = edict(obj_data)
        self.obj_scale: float = obj_data.obj_scale
        # self.obj_rot_axis_angle = obj_data.object_poses[:, :3]
        # self.obj_trans = obj_data.object_poses[:, 3:]
        self.obj_v3d_orig = obj_data.v3d_orig
        self.obj_w2c: torch.Tensor = obj_data.w2c_mats  # N_frames x 4 x 4
        return

    def set_hnd_properties_l(self, hand_data):
        """load left hand data from pre-processing"""
        self.betas_l = torch.Tensor(hand_data["mean_shape"])
        self.hand_poses_l = torch.Tensor(hand_data["hand_poses"])
        self.hand_trans_l = torch.Tensor(hand_data["hand_trans"])

        return

    def set_hnd_properties_r(self, hand_data):
        """load right hand data from pre-processing"""
        self.betas_r = torch.Tensor(hand_data["mean_shape"])
        self.hand_poses_r = torch.Tensor(hand_data["hand_poses"])
        self.hand_trans_r = torch.Tensor(hand_data["hand_trans"])
        self.set_avg_hand_color()
        return

    def construct_targets(self, segmentation_mask) -> edict:
        """
        Given merged SAM mask, extract hand & object masks + bbox.
        Returns a dict of masks & bboxes.
        The masks are torch.Tensors of shape CxHxW, where C=3.
        """
        targets: edict = edict({})
        for entity in self.entities:
            msk = (segmentation_mask * 255 == SEGM_IDS[entity]).float()
            targets[f"msk_{entity}"] = msk  # NOTE: msk.shape = 3xHxW
            # FIXME: breaks if no mask is found; i.e. msk contains only zeros
            # i.e. check mask for 0018.png in ABF14
            bbox = masks_to_boxes(msk)[0].to(int)
            targets[f"bbox_{entity}"] = bbox
        return targets

    def get_single_item(self, i):
        # TODO: optimize data-loader; most of the data is constant;
        #       only load the necessary data for each frame on the GPU...
        if self.split == "train":
            idx = self.train_split[i]
        elif self.split == "val":
            idx = self.val_split[i]
        elif self.split == "anim":
            idx = i

        datum = {}
        if self.split in ["anim", "train", "val"]:
            img: np.array = cv2.imread(self.img_list[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img: Image = Image.fromarray(img)
            img: torch.Tensor = to_tensor(img)
            msk: Image = Image.fromarray(cv2.imread(self.msk_list[idx]))
            msk: torch.Tensor = to_tensor(msk)

            datum.update({"gt_img": img, "gt_msk": msk})

            # extract hand/obj masks & bboxes from segmentation mask
            # TODO: refactor code, s.t. the following two lines are obsolete
            msk_and_bbox = self.construct_targets(msk)
            datum.update(msk_and_bbox)

        datum.update(
            {
                "fovx": self.fovx.item(),
                "fovy": self.fovy.item(),
                "image_path": self.img_list[idx],
                "image_height": self.height,
                "image_width": self.width,
                "world_view_transform": self.Rt_cam,
                "c2w": self.Rt_cam.inverse(),
                "cam_intrinsics": self.K,
                "near": self.cam.znear,
                "far": self.cam.zfar,
                # NOTE: add params for obj pose
                "obj_scale": self.obj_scale,
                "obj_w2c": self.obj_w2c[idx],
            }
        )
        if "right" in self.entities:
            datum.update(
                {
                    "hand_pose_right": self.hand_poses_r[idx],
                    "hand_trans_right": self.hand_trans_r[idx],
                    "betas_right": self.betas_r,
                }
            )
        if "left" in self.entities:
            datum.update(
                {
                    "hand_pose_left": self.hand_poses_l[idx],
                    "hand_trans_left": self.hand_trans_l[idx],
                    "betas_left": self.betas_l,
                }
            )

        return edict(datum)

    def get_object_pc(self):
        """Everytime this method is called, the .ply file is loaded."""
        obj_pc: trimesh.PointCloud = trimesh.load(
            self.obj_ply_p
        )  # original v3d of object (unscaled, incl. color)
        return obj_pc

    def load_data_to_cuda(self):
        self.cached_data = []
        for i in tqdm(range(self.__len__()), desc="Loading data to GPU"):
            datum = self.get_single_item(i)
            for k, v in datum.items():
                if isinstance(v, torch.Tensor):
                    datum[k] = v.to("cuda")
            self.cached_data.append(datum)
