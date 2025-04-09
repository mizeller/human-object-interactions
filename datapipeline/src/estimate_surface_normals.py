"""Inspired by: https://github.com/baegwangbin/DSINE/blob/main/projects/dsine/test_minimal.py"""

import cv2
import imageio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger

import torch
import torch.nn.functional as F
from torchvision import transforms
from utils.video import get_writer_cfg
from utils.constants import device

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../submodules/DSINE"))
import projects.dsine.config as config
from utils.projection import intrins_from_fov, intrins_from_txt
from utils.utils import load_checkpoint, get_padding
from models.dsine.v02 import DSINE_v02 as DSINE


def main(args) -> None:
    # set paths
    paths = {
        "video": Path.cwd() / "data" / args.seq_name / "video.mp4",
    }

    for v in paths.values():
        assert v.exists(), v

    out_p = paths["video"].parent / "normals.mp4"

    model = DSINE(args).to(device)
    model = load_checkpoint(args.ckpt_path, model)
    model.eval()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # TODO: replace cv2
    cap = cv2.VideoCapture(paths["video"])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Processing {paths['video']} with {total_frames} frames...")
    with torch.no_grad():
        with imageio.get_writer(out_p, **get_writer_cfg()) as writer:
            pbar = tqdm(total=total_frames, desc="Processing frames")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = frame.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

                # pad input
                _, _, orig_H, orig_W = img.shape
                lrtb = get_padding(orig_H, orig_W)
                img = F.pad(img, lrtb, mode="constant", value=0.0)
                img = normalize(img)

                # Use default intrinsics with 60-degree FOV -> TODO: use estimated intrinsics
                intrins = intrins_from_fov(
                    new_fov=60.0, H=orig_H, W=orig_W, device=device
                ).unsqueeze(0)
                intrins[:, 0, 2] += lrtb[0]
                intrins[:, 1, 2] += lrtb[2]

                pred_norm = model(img, intrins=intrins)[-1]
                pred_norm = pred_norm[
                    :, :, lrtb[2] : lrtb[2] + orig_H, lrtb[0] : lrtb[0] + orig_W
                ]

                pred_norm = pred_norm.squeeze().detach().cpu().permute(1, 2, 0).numpy()
                pred_norm = (((pred_norm + 1) * 0.5) * 255).astype(np.uint8)

                # update
                writer.append_data(pred_norm)
                pbar.update(1)

            pbar.close()

    cap.release()
    logger.success(f"Processing complete. Output saved to {out_p}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--seq_name", type=str, help="sequence name")
    args = parser.parse_args()

    # prep sys.argv for DSINE's config.py (hack)
    # # DSINE expects config file as sys.argv[1]
    sys.argv = [sys.argv[0], args.config]

    base_conf = config.get_args()
    assert Path(base_conf.ckpt_path).exists(), base_conf.ckpt_path
    base_conf.seq_name = args.seq_name

    main(base_conf)
