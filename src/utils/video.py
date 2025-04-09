import cv2
import torch
import ffmpeg
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple


def probe(video_path: Path) -> Tuple[Union[int, float]]:
    """Extract specs from video using FFMPEG probe."""
    probe = ffmpeg.probe(str(video_path))
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")

    n_frames = int(video_info["nb_frames"])
    height = int(video_info["height"])
    width = int(video_info["width"])
    fps = float(video_info["avg_frame_rate"].split("/")[0])
    return n_frames, height, width, fps


def get_writer_cfg(fps: int = 30, is_mask: bool = False, height: int = None):
    """Get standard video writer configuration.
    - if is_mask=True -> lossless, binary video
    - the provided height should be an even number (req. H.264 codec)!
    """
    config = {
        "fps": fps,
        "format": "FFMPEG",  # for .mp4 outputs
        "codec": "libx264",
        "macro_block_size": 1,
        "pixelformat": "yuv420p",
        "mode": "I",
        "output_params": [],
    }
    if is_mask:
        # it's important that the video only contains valid pixel values
        # (i.e. for segmentation masks).
        # only use lossless mode for H.264 if required though (big files)
        config["output_params"].extend(["-crf", "0"])
        config["pixelformat"] = "gray8"
    if height:
        # NOTE: do it this way to ensure that width remains an even number 
        config["output_params"].extend(["-vf", f"scale=trunc(oh*a/2)*2:{height}"])

    return config


def video2frames(
    video_p: Union[str, Path],
    out_p: Optional[Union[str, Path]] = None,
    max_height: Optional[int] = None,
    return_images: bool = False,
    to_tensor: bool = True,
) -> Optional[Union[torch.Tensor, List[np.ndarray]]]:
    """Convert input video to frames.

    Args:
        video_p: Path to input video
        out_p: Optional output path to save frames
        max_height: Optional maximum height for resizing
        return_images: Whether to return the frames
        to_tensor: Whether to return as PyTorch tensor (ignored if return_images=False)

    Returns:
        If return_images=True:
            if to_tensor=True: torch.Tensor of shape (T, C, H, W)
            if to_tensor=False: List[np.ndarray] of RGB frames
        If return_images=False:
            None (frames are saved to out_p)
    """
    print(f"Loading {video_p}")
    if isinstance(out_p, Path):
        out_p.mkdir(parents=True, exist_ok=True)
        out_p = str(out_p)
    if isinstance(video_p, Path):
        video_p = str(video_p)

    frames = []
    cap = cv2.VideoCapture(video_p)

    while cap.isOpened():
        ret, frame = cap.read()  # frame -> BGR
        if not ret:
            break

        # Get frame dimensions
        height, width = frame.shape[:2]
        size_change = False

        # Ensure dimensions are divisible by 2
        if width % 2 != 0:
            width += 1
            size_change = True
        if height % 2 != 0:
            height += 1
            size_change = True

        if max_height is not None and max_height < height:
            new_height = max_height - max_height % 2
            new_width = int(width * new_height / height)
            new_width = new_width - new_width % 2
            width, height = new_width, new_height
            size_change = True

        assert (
            width % 2 == 0 and height % 2 == 0
        ), f"Width and height must be multiples of 2. Got {width} and {height}"

        if size_change:
            print(
                "Warning - cv2.INTER_CUBIC sub-optimal in case video contains segmentation masks!"
            )
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

        if return_images:
            frame = frame[..., ::-1]  # BGR -> RGB
            frames.append(frame)

        if out_p:
            cv2.imwrite(
                f"{out_p}/{len(frames):04d}.png", frame[..., ::-1]
            )  # RGB -> BGR

    cap.release()

    if return_images:
        if to_tensor:
            # Convert to tensor of shape (T, C, H, W)
            frames = np.stack(frames)  # (T, H, W, C)
            frames = torch.from_numpy(frames)  # to torch
            frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        return frames
    return None
