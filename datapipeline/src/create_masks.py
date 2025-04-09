import torch
import pickle
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt

from utils import constants
from utils.video import video2frames, get_writer_cfg

from sam2.build_sam import build_sam2_video_predictor

# floating point precision
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def create_colored_mask(mask, obj_id=None):
    if obj_id is None:
        color = np.random.random(3)
    else:
        colors = np.array(
            [
                [0.12, 0.47, 0.71],
                [0.20, 0.63, 0.17],
                [0.89, 0.10, 0.11],
                # Add more as needed
            ]
        )
        color = colors[obj_id % len(colors)]

    return np.dstack([mask * c * 0.6 for c in color])


# main routine #####################################################################################


def main(args) -> None:
    logger.warning("Starting SAM2 routine🤖")

    uuid_o: int = 0
    uuid_h: int = 1

    # get/set paths
    paths: Dict[str:Path] = get_paths(args.seq_name)

    # load `video.mp4` into memory
    frames = video2frames(video_p=paths["video"], return_images=True)

    if paths["mask_cache"].exists() and args.use_cache:
        logger.info("Loading masks from cache.")
        video_segments = pickle.load(open(paths["mask_cache"], "rb"))
    else:
        # init SAM-2 predictor & inference state
        ckpt = constants.sam2_ckpt
        model_cfg = constants.sam2_model_cfg

        predictor = build_sam2_video_predictor(model_cfg, ckpt)
        inference_state = predictor.init_state(video_frames=frames)

        # load prompts from your annotation step
        try:
            prompts = np.load(paths["prompt_cache"], allow_pickle=True).item()
            logger.info(f"Loaded prompts for {len(prompts)} frames")
        except:
            logger.error("No prompts.npy found! Run create_mask_prompts.py first.")
            return

        # Add prompts for each frame that has annotations
        for frame_idx in sorted(prompts.keys()):
            frame_data = prompts[frame_idx]

            if uuid_o in frame_data:
                points, labels = frame_data[uuid_o]
                _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=uuid_o,
                    points=points,
                    labels=labels,
                )

            if uuid_h in frame_data:
                points, labels = frame_data[uuid_h]
                _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=uuid_h,
                    points=points,
                    labels=labels,
                )

        ###########################################################

        logger.info("Propagating masks throughout video...")
        video_segments = {}

        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        pickle.dump(video_segments, open(paths["mask_cache"], "wb"))

    plt.close("all")

    with imageio.get_writer(paths["vis"], **get_writer_cfg(height=896)) as writer:
        for frame_idx in tqdm(range(0, len(frames))):
            frame = frames[frame_idx]

            for obj_id, mask in video_segments[frame_idx].items():
                mask = mask.reshape(frame.shape[0], frame.shape[1])
                colored_mask = create_colored_mask(mask, obj_id)
                mask_3d = np.dstack([mask] * 3)
                frame = np.where(mask_3d, frame * 0.4 + colored_mask * 255, frame)

            writer.append_data(frame.astype(np.uint8))

        logger.info(f"Sanity Check: {paths['vis']}")

    with imageio.get_writer(paths["masks"], **get_writer_cfg(is_mask=True)) as writer:
        for frame, segmentation in tqdm(
            video_segments.items(), "Merging human-object masks."
        ):
            object_mask = segmentation[uuid_o].squeeze()
            hand_mask = segmentation[uuid_h].squeeze()

            combined_mask = np.zeros_like(object_mask, dtype=np.uint8)
            combined_mask[object_mask] = constants.SEGM_IDS.get("object", 50)
            combined_mask[hand_mask] = constants.SEGM_IDS.get("right", 150)

            writer.append_data(combined_mask.astype(np.uint8))

        logger.success(f"Success. {paths['masks']}")
    return


def get_paths(seq_name: str) -> Dict[str, Path]:
    """Initialize paths; make sure they exist!"""
    cwd = Path.cwd()
    data = cwd / "data" / seq_name

    p = {
        "cwd": cwd,
        "data": data,
        "video": data / "video.mp4",
        "masks": data / "masks.mp4",
        # "images": data / "images",
        # "masks": data / "masks",
        "sam2": data / "scratch" / "sam2",
    }

    assert p["video"].exists(), p["video"]
    # p["images"].mkdir(parents=True, exist_ok=True)
    # p["masks"].mkdir(parents=True, exist_ok=True)
    p["sam2"].mkdir(parents=True, exist_ok=True)

    # Add cache and visualization paths
    p["prompt_cache"] = p["sam2"] / "prompts.npy"
    p["mask_cache"] = p["sam2"] / "segmentations.npy"
    p["vis"] = p["sam2"] / "vis_sam2.mp4"

    return p


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, required=True)
    parser.add_argument("--use_cache", action="store_true")
    args = parser.parse_args()

    main(args=args)
