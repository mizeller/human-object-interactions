from pathlib import Path
import torch

device = torch.device("cuda")

SEGM_IDS = {"bg": 0, "object": 50, "right": 150, "human": 150, "left": 250}

# the MC camera motion repo, is the key submodule!
submodules_p: Path = Path.cwd() / "submodules"
camera_motion_p = submodules_p / "camera_motion"

# smplx
SMPLX_NEUTRAL_P = (
    camera_motion_p / "data" / "models" / "SMPLX" / "neutral" / "SMPLX_neutral.npz"
)
assert SMPLX_NEUTRAL_P.exists(), SMPLX_NEUTRAL_P

# gloss
gloss_cfg_p = camera_motion_p / "lib" / "pipeline" / "gloss_no_bg_floor.toml"
assert gloss_cfg_p.exists(), gloss_cfg_p
gloss_cfg_p = str(gloss_cfg_p.absolute())

# sam2
sam2_ckpt = (
    camera_motion_p
    / "data"
    / "pretrained-models"
    / "afv2_data"
    / "sam2_ckpts"
    / "sam2_hiera_large.pt"
)
_sam2_model_cfg = camera_motion_p / "lib" / "pipeline" / "sam2" / "sam2_hiera_l.yaml"
assert _sam2_model_cfg.exists(), _sam2_model_cfg
assert sam2_ckpt.exists(), sam2_ckpt
# hack s.t. hydra accepts the path
sam2_model_cfg = "/" + str(_sam2_model_cfg.absolute())


# trellis
trellis_p = submodules_p / "TRELLIS/models/TRELLIS-image-large"


# taken from https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
# len = 144

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


rh_idx = [
    21,  # right_wrist
    40,  # right_index1
    41,  # right_index2
    42,  # right_index3
    72,  # right_index
    43,  # right_middle1
    44,  # right_middle2
    45,  # right_middle3
    73,  # right_middle
    46,  # right_pinky1
    47,  # right_pinky2
    48,  # right_pinky3
    75,  # right_pinky
    49,  # right_ring1
    50,  # right_ring2
    51,  # right_ring3
    74,  # right_ring
    52,  # right_thumb1
    53,  # right_thumb2
    54,  # right_thumb3
    71,  # right_thumb
]

lh_idx = [
    20,  # left_wrist
    25,  # left_index1
    26,  # left_index2
    27,  # left_index3
    67,  # left_index
    28,  # left_middle1
    29,  # left_middle2
    30,  # left_middle3
    68,  # left_middle
    31,  # left_pinky1
    32,  # left_pinky2
    33,  # left_pinky3
    70,  # left_pinky
    34,  # left_ring1
    35,  # left_ring2
    36,  # left_ring3
    69,  # left_ring
    37,  # left_thumb1
    38,  # left_thumb2
    39,  # left_thumb3
    66,  # left_thumb
]

# # extract hand joints from smplx output (after forward pass)
# right_hand_joints = smplx_output.joints[:, rh_idx, :]
# left_hand_joints = smplx_output.joints[:, lh_idx, :]
# all_hand_joints = torch.cat([right_hand_joints, left_hand_joints], dim=1)
