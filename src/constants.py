import torch
import json
from pathlib import Path

# segmentation mask uuids
# NOTE: renamed "right" to "human"; this will break some code.
SEGM_IDS = {"bg": 0, "object": 50, "human": 150, "right": 150}


device = torch.device("cuda")

# set paths
project_root: Path = Path.cwd()
# data_p = project_root / "data"
assets_p = project_root / "assets"
tmp_p = project_root / "tmp"

if not tmp_p.exists():
    tmp_p.mkdir()

# models
models_p = (
    project_root / "datapipeline" / "submodules" / "camera_motion" / "data" / "models"
)
mano_p = models_p / "mano"
smplx_p = models_p / "SMPLX" / "neutral" / "SMPLX_neutral.npz"

# download this file from https://smpl-x.is.tue.mpg.de
smplx2mano_p = assets_p / "MANO_SMPLX_vertex_ids.pkl"
smplx2flame_p = assets_p / "SMPL-X__FLAME_vertex_ids.npy"

# data
# --- original HO3D data (v3)
# ho3d_p = data_p / "data_HO3D_v3"
# --- path to 'originally' preprocessed data from HOLD; contains hold_<seq_id>_ho3d sub-directories
# --- which in turn contain a corres.txt for mapping HOLD to HO3D frames;
# --- thus we can derive the mapping between the converged frames in HOLD-GS and the original HO3D frames
# hold_p = data_p / "data_HOLD"

# # --- original arctic data
# arctic_data_p: Path = Path.cwd() / "arctic/data/arctic_data/data"
# arctic_raw_p = arctic_data_p / "raw_seqs"

# NOTE: this is only required for th dataset (which requires some modifications for the latest training pipeline)
# arctic_meta_p = arctic_data_p / "meta"
# try:
#     with open(arctic_meta_p / "misc.json", "r") as f:
#         arctic_meta = json.load(f)
# except Exception as e:
#     arctic_meta = None
#     print(e)


# ensure paths exist to avoid errors
assert assets_p.exists()
assert tmp_p.exists()

assert mano_p.exists()
assert smplx_p.exists()


################################################################################
# SMPLX 2 MANO
################################################################################

# taken from https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
# len = 144
#
SMPLX_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    # l/r hand root
    "left_wrist",  # 20
    "right_wrist",  # 21
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    # left hand: indices 25-39
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    # right hand: indices 40-54
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    # left finger tips: 66-70
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    # right finger tips: 71-75
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

right_hand_indices = [
    21,  # right_wrist
    40,  # right_index1
    41,  # right_index2
    42,  # right_index3
    43,  # right_middle1
    44,  # right_middle2
    45,  # right_middle3
    49,  # right_ring1
    50,  # right_ring2
    51,  # right_ring3
    46,  # right_pinky1
    47,  # right_pinky2
    48,  # right_pinky3
    52,  # right_thumb1
    53,  # right_thumb2
    54,  # right_thumb3
    71,  # right_thumb
    72,  # right_index
    73,  # right_middle
    74,  # right_ring
    75,  # right_pinky
]

left_hand_indices = [
    20,  # left_wrist
    25,  # left_index1
    26,  # left_index2
    27,  # left_index3
    28,  # left_middle1
    29,  # left_middle2
    30,  # left_middle3
    34,  # left_ring1
    35,  # left_ring2
    36,  # left_ring3
    31,  # left_pinky1
    32,  # left_pinky2
    33,  # left_pinky3
    37,  # left_thumb1
    38,  # left_thumb2
    39,  # left_thumb3
    66,  # left_thumb
    67,  # left_index
    68,  # left_middle
    69,  # left_ring
    70,  # left_pinky
]

# # extract hand joints from smplx output (after forward pass)
# right_hand_joints = smplx_output.joints[:, right_hand_indices, :]
# left_hand_joints = smplx_output.joints[:, left_hand_indices, :]
# # combine both hands (if so desired)
# all_hand_joints = torch.cat([right_hand_joints, left_hand_joints], dim=1)
