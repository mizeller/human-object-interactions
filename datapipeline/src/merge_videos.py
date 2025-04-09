import cv2
import torch
import imageio
import zipfile
import argparse
import numpy as np
from typing import Dict, List
from pathlib import Path
from loguru import logger
from utils import debug, video


def create_video_grid(ps: List[Path], verbose: bool = False) -> None:
    cap = cv2.VideoCapture(str(ps["gt"]))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    f, s, t, c = cv2.FONT_HERSHEY_SIMPLEX, 2.0, 2, (128, 128, 128)
    (text_width, text_height), _ = cv2.getTextSize("PLACEHOLDER", f, s, t)
    text_x = (W - text_width) // 2
    text_y = (H + text_height) // 2

    placeholder = np.ones((H, W, 3)) * 255
    cv2.putText(placeholder, "PLACEHOLDER", (text_x, text_y), f, s, c, t)
    placeholder = placeholder.astype(np.uint8)

    logger.info(f"Reading videos")
    frames = {}
    for k, v in ps.items():
        frames[k] = np.array(video.video2frames(v, return_images=True))
        if args.verbose:
            print(k, frames[k].shape)

    with imageio.get_writer(ps["out"], **video.get_writer_cfg(height=896)) as writer:
        for i in range(N):
            top, bottom = [], []
            for key in ["mfv", "sam2", "colmap"]:
                if i >= len(frames.get(key, [])):
                    top.append(placeholder)
                else:
                    top.append(frames[key][i])

            for key in ["gt", "post_alignment"]:
                if i >= len(frames.get(key, [])):
                    bottom.append(placeholder)
                else:
                    bottom.append(frames[key][i])

            top = np.concatenate(top, axis=1)
            bottom = np.concatenate(bottom, axis=1)

            if top.shape[1] != bottom.shape[1]:  # ensure shapes match
                bottom = cv2.resize(bottom, (top.shape[1], bottom.shape[0]))

            grid = np.concatenate([top, bottom], axis=0)

            # add frame counter
            (text_width, text_height), _ = cv2.getTextSize(f"Frame: {i+1}/{N}", f, s, t)
            cv2.putText(grid, f"Frame: {i+1}/{N}", (10, 10 + text_height), f, s, c, t)
            writer.append_data(grid)

    return


def zip_dataset(data_p: Path, zip_p: Path, exclusions: List[str] = ["scratch"]) -> None:
    """Zip input directory to zip_p w/o exclusions (files/directories)."""

    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in data_p.rglob("*"):
            if path.is_dir():
                if any(excl in path.parts for excl in exclusions):
                    continue
            else:
                if any(excl in path.name for excl in exclusions):
                    continue
                zipf.write(path, path.relative_to(data_p))

    logger.info(f"Exported zipfile to {zip_p}")
    return


def clean_dataset() -> None:
    """Remove obsolete files."""
    raise NotImplementedError


def main(args) -> None:
    """
    - [x] Merge the output visualizations of all separate steps & align them in a nice grid here
    - [x] Create an overview of the k:v pairs in data.pt
    - TODO: zip data
    - TODO: remove obsolete data
    """
    # paths
    seq_p = Path.cwd() / "data" / args.seq_name

    ps: Dict[str, Path] = {
        # "gt": seq_p / "video.mp4",
        "out": seq_p / "out.mp4",
        "data": seq_p / "data.pt",
        "zip": seq_p.parent / f"{args.seq_name}.zip",
        "gt": seq_p / "scratch" / "mfv" / "vis_video.mp4",
        "mfv": seq_p / "scratch" / "mfv" / "vis_overlay.mp4",
        "sam2": seq_p / "scratch" / "sam2" / "vis_sam2.mp4",
        "colmap": seq_p / "scratch" / "colmap" / "vis_colmap.mp4",
        "post_alignment": seq_p / "scratch" / "alignment" / "vis_post_align.mp4",
    }

    # NOTE: assume these files exist for now.. ~.~
    create_video_grid(ps=ps, verbose=args.verbose)

    if args.verbose:
        data = torch.load(ps["data"], map_location="cpu")
        debug.log(data)
    if args.zip:
        zip_dataset(data_p=seq_p, zip_p=ps["zip"])
    if args.clean:
        clean_dataset()

    logger.success(f"Pre-Processing done, check {ps['out']}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, required=True)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    # TODO: implement these two!
    parser.add_argument("--zip", action="store_true", help="Create zip file.")
    parser.add_argument("--clean", action="store_true", help="Remove obsolete files.")
    args = parser.parse_args()
    main(args)
