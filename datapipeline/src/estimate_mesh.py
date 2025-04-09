"""Extract mesh from multiple object-only images with TRELLIS. Heavily inspired by `example_multi_image.py`"""

import os

# Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["ATTN_BACKEND"] = "xformers"

# Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.
os.environ["SPCONV_ALGO"] = "native"

import sys
import random
import imageio
import numpy as np
from PIL import Image
from typing import List
from pathlib import Path
from torchvision import transforms
from torchvision.utils import make_grid


# local
from utils.constants import SEGM_IDS, trellis_p
from utils.video import video2frames

sys.path.append(os.path.join(os.path.dirname(__file__), "../submodules/TRELLIS"))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def main(args) -> None:
    # set paths
    data_p: Path = Path.cwd() / "data" / args.seq_name

    out_p = data_p / "scratch" / "trellis"
    out_p.mkdir(parents=True, exist_ok=True)
    glb_p = out_p / "v3d_o.glb"
    vis_p = out_p / "vis_trellis.mp4"
    video_p = data_p / "video.mp4"
    masks_p = data_p / "masks.mp4"

    # video.mp4 + masks.mp4 = object_only images -> SfM
    assert video_p.exists(), video_p
    assert masks_p.exists(), masks_p

    frames = video2frames(video_p=data_p / "video.mp4", return_images=True)
    masks = video2frames(video_p=data_p / "masks.mp4", return_images=True)

    assert (
        len(np.unique(masks[0])) == 3
    ), f"masks.mp4 should only contain three pixel values, found {len(np.unique(masks[0]))}."

    N = len(frames)
    assert N == len(masks), "video.mp4 & masks.mp4 have different number of frames!"

    # Sample random frames
    random.seed(42)
    sampled_indices = random.sample(range(N), k=10)

    images: List[Image.Image] = []
    for i in sampled_indices:
        frm = frames[i]
        msk = masks[i]

        # binary obj. mask
        obj_mask = np.all(msk == SEGM_IDS.get("object", 50), axis=-1)
        obj_only = frm.copy()
        obj_only[~obj_mask] = 255  # non-object pixels = white

        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(obj_only)
        images.append(pil_img)

    # Create and save grid visualization
    to_tensor = transforms.ToTensor()
    tensor_images = [to_tensor(img) for img in images]
    grid = make_grid(tensor_images, nrow=5, padding=2)
    grid_img = transforms.ToPILImage()(grid)
    grid_img.save(out_p / "obj_only_imgs.png")

    # Load a pipeline from a model folder or a Hugging Face model hub.
    # pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    print("WARNING: Hard-Coded Path to Trellis-Checkpoint!")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(trellis_p))
    pipeline.cuda()

    # Run the pipeline
    outputs = pipeline.run_multi_image(
        images,
        seed=1,
        # Optional parameters
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3,
        },
    )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    video_gs = render_utils.render_video(outputs["gaussian"][0])["color"]
    video_mesh = render_utils.render_video(outputs["mesh"][0])["normal"]
    video = [
        np.concatenate([frame_gs, frame_mesh], axis=1)
        for frame_gs, frame_mesh in zip(video_gs, video_mesh)
    ]
    imageio.mimsave(vis_p, video, fps=30)

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        # Optional parameters
        simplify=0.95,  # Ratio of triangles to remove in the simplification process
        texture_size=1024,  # Size of the texture used for the GLB
    )
    glb.export(glb_p)
    print("success.")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, help="sequence name")
    args = parser.parse_args()

    main(args)
