# load ARCTIC Ground Truth Data here for now..
import PIL
import cv2
import torch
import trimesh
from tqdm import tqdm
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Union
import torchvision.transforms as T


# local
import sys

# hack: add project root to path, s.t. aligned_data npy can be loaded
#       otherwise, it cannot be unpacked because it doesnt know 'common'
# todo: only save default dict type in aligned_data.npy; not xdicts!

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.constants import arctic_raw_p, arctic_meta, project_root, device, SEGM_IDS
from src.utils import camera, subdivide_model
from src.common.body_models import build_subject_smplx
from smplx import SMPLX


class ArcticGroundTruthData(torch.utils.data.Dataset):
    """Minimal Data-Loader for Ground Truth ARCTIC Data."""

    def __init__(self):
        # hard-code properties
        self.subject: str = "s01"  # "s02", ..., "s10"
        self.obj_name: str = "box"  # capsulemachine, espressomachine, ketchup, ...
        self.action: str = "grab"  # "use"
        self.device = device

        smplx_p: Path = (
            arctic_raw_p / self.subject / f"{self.obj_name}_{self.action}_01.smplx.npy"
        )
        obj_p: Path = (
            arctic_raw_p / self.subject / f"{self.obj_name}_{self.action}_01.object.npy"
        )
        img_p: Path = (
            project_root / "arctic_s01_box_grab_01_1" / "images"
        )  # aka img_dir in HOLDataset
        msk_p: Path = (
            project_root / "arctic_s01_box_grab_01_1" / "masks"
        )  # aka msk_dir in HOLDataset
        assert smplx_p.exists(), smplx_p
        assert obj_p.exists(), obj_p
        assert img_p.exists(), img_p
        assert msk_p.exists(), msk_p

        self.init_cams()
        self.init_images(img_p=img_p, msk_p=msk_p)
        self.init_smplx(smplx_p=smplx_p)
        self.init_object(obj_p=obj_p)

        if not hasattr(self, "cached_data"):
            self.load_data_to_cuda(n=15)

        # HOLD dataset compatibility...
        self.frames_w_bad_init_poses: List[int] = []

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int):
        if self.cached_data is not None:  # is not None
            return self.cached_data[idx]
        else:
            return self.get(idx)

    def load_data_to_cuda(self, n: int = None) -> None:
        """Load data to CUDA. Only load dataset partially for memory/efficeny reasons (i.e when debugging.)"""

        # Get every 10th frame starting at 0
        frame_indices = range(0, self.__len__(), 10)
        if n:
            logger.warning(
                f"Loading only the first {n}/{self.__len__()} frames to CUDA!"
            )
            # Take only first n=15 frames
            selected_indices = list(frame_indices)[:n]
        else:
            selected_indices = list(frame_indices)

        # n_frames: int = self.__len__() if n is None else n
        # self.num_frames = n_frames  # overwrite num_frames here

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
        # check ARCTIC/preprocess_dataset.py for more details (i.e. MANO support)
        out_dict = {}

        # smplx
        # TODO: do I need all of these params? to make it more (memory) efficient, I can probably skip most of them (we only care about hands primarily!)
        out_dict["smplx_transl"] = self.smplx_transl[idx]
        out_dict["smplx_global_orient"] = self.smplx_global_orient[idx]
        out_dict["smplx_body_pose"] = self.smplx_body_pose[idx]
        out_dict["smplx_jaw_pose"] = self.smplx_jaw_pose[idx]
        out_dict["smplx_leye_pose"] = self.smplx_leye_pose[idx]
        out_dict["smplx_reye_pose"] = self.smplx_reye_pose[idx]
        out_dict["smplx_left_hand_pose"] = self.smplx_left_hand_pose[idx]
        out_dict["smplx_right_hand_pose"] = self.smplx_right_hand_pose[idx]

        # object
        # pylint: disable=unsubscriptable-object
        out_dict["obj_rot"] = self.obj_rot[idx]
        out_dict["obj_trans"] = self.obj_trans[idx]  # to meter

        out_dict["query_names"] = self.obj_name

        # images/masks
        img: PIL.Image = PIL.Image.open(self.img_list[idx]).convert("RGB")
        img: torch.Tensor = T.ToTensor()(img)
        out_dict["gt_img"] = img

        # pylint: disable=no-member
        msk: PIL.Image = PIL.Image.fromarray(cv2.imread(self.msk_list[idx]))
        msk: torch.Tensor = T.ToTensor()(msk)
        out_dict["gt_msk"] = msk

        # TODO: make nice...
        msk_and_bbox = self.construct_targets(msk)
        out_dict.update(msk_and_bbox)

        return out_dict

    def init_object(self, obj_p: Path) -> None:
        from src.common.object_tensors import ObjectTensors

        # init ARCTIC object (re-use in ObjectGS)
        self.object_tensors: ObjectTensors = ObjectTensors(objects_list=[self.obj_name])
        self.object_tensors.to(self.device)

        self.init_pcd: Union[trimesh.Trimesh, trimesh.PointCloud] = trimesh.Trimesh(
            vertices=self.object_tensors.obj_tensors["v"].squeeze(0).cpu(),
            faces=self.object_tensors.obj_tensors["f"].squeeze(0).cpu(),
        )

        # init object with random colors
        self.init_pcd.visual.vertex_colors = np.random.randint(
            0, 255, size=(len(self.init_pcd.vertices), 3)
        )

        # load pose params, num_frames x 7
        obj_params = torch.FloatTensor(np.load(obj_p, allow_pickle=True))

        self.obj_rot = obj_params[:, 1:4]  # in axis-angle format
        self.obj_trans = obj_params[:, 4:] / 1000  # m -> mm

        assert torch.isnan(obj_params).sum() == 0
        assert torch.isinf(obj_params).sum() == 0

        self._sanity_check("obj_")

        return

    def init_cams(self) -> None:
        """Load all static cameras from ARCTIC dataset. NOTE: Using only the first one for now."""

        # world2cam is in OpenCV convention. w.r.t world frame
        world2cam = torch.FloatTensor(np.array(arctic_meta[self.subject]["world2cam"]))
        intris_mat = torch.FloatTensor(
            np.array(arctic_meta[self.subject]["intris_mat"])
        )
        image_size: List[int] = arctic_meta[self.subject]["image_size"]

        self.cameras: List[camera.Camera] = []
        for i in range(len(world2cam)):
            _cam = camera.Camera(
                cam_id=i,
                image_size=image_size[i],
                cam_intrinsics=intris_mat[i],
                world_view_transform=world2cam[i],
            )
            self.cameras.append(_cam)

        self.cam = self.cameras[0]  # use first cam!
        return

    def init_smplx(self, smplx_p: Path) -> None:

        # init personalized & sub-divided SMPL-X layer for ARCTIC subject
        smplx_pers: SMPLX = build_subject_smplx(subject_id=self.subject)
        smplx_pers_sub: SMPLX = subdivide_model.subdivide_model(
            model=smplx_pers.to(self.device), n_iter=2
        )

        self.smplx = smplx_pers_sub

        # load pose params
        smplx_data = np.load(smplx_p, allow_pickle=True).item()
        self.smplx_transl = torch.FloatTensor(smplx_data["transl"])
        self.smplx_global_orient = torch.FloatTensor(smplx_data["global_orient"])
        self.smplx_body_pose = torch.FloatTensor(smplx_data["body_pose"])
        self.smplx_jaw_pose = torch.FloatTensor(smplx_data["jaw_pose"])
        self.smplx_leye_pose = torch.FloatTensor(smplx_data["leye_pose"])
        self.smplx_reye_pose = torch.FloatTensor(smplx_data["reye_pose"])
        self.smplx_left_hand_pose = torch.FloatTensor(smplx_data["left_hand_pose"])
        self.smplx_right_hand_pose = torch.FloatTensor(smplx_data["right_hand_pose"])
        # TODO:add these params as nn.Params to HumanGS model!
        self._sanity_check("smplx_")

        return

    def init_images(self, img_p: Path, msk_p: Path) -> None:
        self.img_list: List[Path] = sorted(img_p.glob("*"))
        self.msk_list: List[Path] = sorted(msk_p.glob("*"))
        assert len(self.img_list) == len(
            self.msk_list
        ), "Mismatch in img & msk lists length! Check pre-processing..."
        self.num_frames = len(self.img_list)
        return

    def _sanity_check(self, _attr: str) -> None:
        """Ensure that the length of all attributes (tensors) that start with `_attr` = num_frames."""
        skip = ["obj_name"]  # these attributes are not lists/tensors!
        attrs = [
            attr for attr in dir(self) if attr.startswith(_attr) and attr not in skip
        ]
        for attr in attrs:
            curr_len = len(getattr(self, attr))
            assert (
                curr_len == self.num_frames
            ), f"Length mismatch: {attr} has length {curr_len}, expected {self.num_frames}"
        return

    def construct_targets(self, segmentation_mask) -> dict:
        """
        Given merged SAM mask, extract hand & object masks.
        The masks are torch.Tensors of shape CxHxW, where C=3.
        TODO: the ARCTIC masks are not really binary/in the same segmentation convention from HO3D!
        """
        targets = {}
        for entity in ["right", "object"]:
            lower, upper = SEGM_IDS[entity]
            msk = (
                (segmentation_mask * 255 >= lower) & (segmentation_mask * 255 <= upper)
            ).float()
            # msk = (segmentation_mask * 255 == SEGM_IDS[entity]).float()
            targets[f"msk_{entity}"] = msk  # NOTE: msk.shape = 3xHxW

        return targets
