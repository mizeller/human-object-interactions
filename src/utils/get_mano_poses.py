import torch
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict
from src.constants import assets_p


def main() -> None:
    seq: str = "hold_cube"  # has N frames

    build_p: Path = Path(f"./data/{seq}/build")

    # can safely assume these files/directories to exist in build dir
    data_p: Path = build_p / "data.npy"
    seq_data: edict = edict(np.load(data_p, allow_pickle=True).item())

    hand_data = seq_data.entities["right"]  # right hand for now

    hand_pose = torch.Tensor(hand_data["hand_poses"][:, 3:])  # N x 45
    global_orient = torch.Tensor(hand_data["hand_poses"][:, :3])  # N x 3
    betas = torch.Tensor(hand_data["mean_shape"])  # 1 x 10
    transl = torch.Tensor(hand_data["hand_trans"])  # N x 3

    # TODO: save these variables to a .npy file or

    out = {
        "betas": betas,
        "global_orient": global_orient,
        "hand_pose": hand_pose,
        "transl": transl,
    }

    torch.save(out, assets_p / "mano_seq.pt")

    pass


if __name__ == "__main__":
    main()
