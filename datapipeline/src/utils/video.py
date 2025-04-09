"""video utils."""

import cv2
from pathlib import Path
from typing import Union


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
    out_p: Union[str, Path] = None,
    max_height=None,
    return_images: bool = False,
):
    """Convert input video to images.
    If out_p is specified       ->  save images there.
    If return_images is True    ->  return images as np.array. RGB format.
    """
    # cv2 incompatible with path objects!
    if isinstance(out_p, Path):
        out_p.mkdir(parents=True, exist_ok=True)
        out_p = str(out_p)
    if isinstance(video_p, Path):
        video_p = str(video_p)

    count = 0
    frames = []
    cap = cv2.VideoCapture(video_p)
    while cap.isOpened():
        ret, frame = cap.read()  # frame -> BRG
        if ret == True:
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

            # TODO: more general, check for max_dim (height OR width)...
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
                # Resize the frame
                print(
                    "Warning - cv2.INTER_CUBIC sub-optimal in case video contains segmentation masks!"
                )
                frame_resized = cv2.resize(
                    frame, (width, height), interpolation=cv2.INTER_CUBIC
                )
            else:
                frame_resized = frame

            if return_images:
                frames.append(frame_resized[..., ::-1])  # convert to RGB

            if out_p:
                cv2.imwrite(f"{out_p}/{count:04d}.png", frame_resized)

            count += 1
        else:
            break
    cap.release()

    return frames
