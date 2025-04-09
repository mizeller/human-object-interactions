import cv2
import torch
import joblib
import shutil
import trimesh
import pycolmap
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger

from utils.colmap import (
    clean_pc,
    verify_colmap,
    analyze_preprocessing,
    colmap2gloss,
)
from utils.constants import SEGM_IDS
from utils.video import video2frames

import sys

sys.path.append(str(Path.cwd() / "submodules" / "hloc"))

from hloc.utils import viz_3d

# imported from: # https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/
from hloc import (
    extract_features,
    match_features,
    # match_dense,
    pairs_from_retrieval,
    reconstruction,
)


def main(args):
    """Run SfM pipeline on `object_only` images, interpolate missing poses, and store the scratch data."""

    # set paths
    data_p: Path = Path.cwd() / "data" / args.seq_name

    out_p = data_p / "scratch" / "colmap"
    out_p.mkdir(parents=True, exist_ok=True)

    ply_p = out_p / "v3d_o.ply"
    sfm_p = out_p / "model"  # sfm_model; find .bin files here
    sfm_pairs_p = out_p / "pairs-netvlad.txt"
    feat_p = out_p / "features.h5"
    obj_data_p = out_p / "obj_data.npy"  # obj. data in COLMAP space
    img_p = out_p / "images"  # obj. only images
    mfv_data_p = data_p / "scratch" / "mfv" / "results.pkl"
    mfv_video_p = data_p / "scratch" / "mfv" / "vis_video.mp4"
    video_p = data_p / "video.mp4"
    masks_p = data_p / "masks.mp4"

    # video.mp4 + masks.mp4 = object_only images -> SfM
    assert video_p.exists(), video_p
    assert masks_p.exists(), masks_p

    logger.warning("Extracting the camera params from the mfv output!")
    camera_world = joblib.load(mfv_data_p)["camera_world"]
    camera_focal = camera_world["img_focal"]
    camera_center = camera_world["img_center"]

    # World Cam Extrinsics from MfV step (OpenGL)
    camera_extrinsics = np.eye(4)
    camera_extrinsics[:3, :3] = camera_world["Rcw"][0]
    camera_extrinsics[:3, 3] = camera_world["Tcw"][0]

    frames = video2frames(video_p=data_p / "video.mp4", return_images=True)
    masks = video2frames(video_p=data_p / "masks.mp4", return_images=True)

    assert (
        len(np.unique(masks[0])) == 3
    ), f"masks.mp4 should only contain three pixel values, found {len(np.unique(masks[0]))}."

    N = len(frames)
    assert N == len(masks), "video.mp4 & masks.mp4 have different number of frames!"

    if not args.skip_sfm:

        img_p.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(N), "Creating object-only images"):
            frm = frames[i]
            msk = masks[i]

            # binary obj. mask
            obj_mask = np.all(msk == SEGM_IDS.get("object", 50), axis=-1)
            obj_only = frm.copy()
            obj_only[~obj_mask] = 255  # non-object pixels = white

            cv2.imwrite(f"{img_p}/{i:04d}.png", obj_only[..., ::-1])  # BGR

        assert args.num_pairs <= N, f"{args.num_pairs} should be <= {N}"

        if args.gt_intrinsics is None:
            camera_params = None
            logger.warning("Running COLMAP with known camera intrinsics.")

            # these camera params belong to the video @ mfv_video_p
            # this might be a scaled version of the video @ video_p.
            # re-scale camera_world params accordingly!
            full_video = cv2.VideoCapture(video_p)
            mfv_video = cv2.VideoCapture(mfv_video_p)
            s = full_video.get(cv2.CAP_PROP_FRAME_WIDTH) / mfv_video.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )  # should be the same for height
            full_video.release()
            mfv_video.release()

            camera_center = [x * s for x in camera_center]
            camera_focal *= s

            camera_params = f"{camera_focal},{camera_center[0]},{camera_center[1]}"
        else:
            logger.warning("Running COLMAP with known camera intrinsics.")
            camera_params = args.gt_intrinsics
            # camera intrinsics for MC1 dataset (extracted from meta data)
            # logger.error("HARDCODED CAMERA INTRINSICS!")
            # camera_params = "617.343,312.42,241.42"  # f,cx,cy

        # run SfM pipeline
        model = None

        logger.info("Running SfM pipeline")
        # NOTE: make sure you have the required .bin files in the sfm_p
        retrieval_conf = extract_features.confs["netvlad"]

        # NOTE: option 1
        feature_conf = extract_features.confs["superpoint_aachen"]
        matcher_conf = match_features.confs["superglue"]

        # NOTE: option 2 - sometimes, if option 1 does not converge, this helps...
        # feature_conf = extract_features.confs["superpoint_max"]
        # feature_conf["model"]["max_keypoints"] = 8192
        # feature_conf["preprocessing"]["resize_max"] = 2048
        # matcher_conf = match_features.confs["superpoint+lightglue"]
        # matcher_conf["model"].update(
        #     {
        #         "n_layers": 9,
        #         "flash": True,
        #         "depth_confidence": 0.95,
        #     }
        # )

        references = [p.relative_to(img_p).as_posix() for p in (img_p).iterdir()]

        retrieval_path = extract_features.main(
            retrieval_conf, img_p, image_list=references, feature_path=feat_p
        )
        pairs_from_retrieval.main(
            retrieval_path, sfm_pairs_p, num_matched=args.num_pairs
        )

        feature_path = extract_features.main(feature_conf, img_p, out_p)
        match_path = match_features.main(
            matcher_conf, sfm_pairs_p, feature_conf["output"], out_p
        )

        # NOTE: adding support for known camera intrinsics
        #       i.e. in the case of HO3D data, we can assume to know them...

        image_options, mapper_options = None, None

        if camera_params is not None:
            image_options = dict(
                camera_model="SIMPLE_PINHOLE", camera_params=camera_params
            )
            # prevent camera intrinsics from being optimized
            mapper_options = dict(
                ba_refine_focal_length=False, ba_refine_extra_params=False
            )

        # at this point, have SfM model!
        model: pycolmap.Reconstruction = reconstruction.main(
            sfm_dir=sfm_p,
            image_dir=img_p,
            pairs=sfm_pairs_p,
            features=feature_path,
            matches=match_path,
            camera_mode=pycolmap.CameraMode.PER_FOLDER,
            mapper_options=mapper_options,
            image_options=image_options,
            verbose=False,
        )

        logger.success("Completed SfM pipeline")
        logger.info(model.summary())

        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
        )
        # fig.show()
        fig.write_html(f"{str(out_p)}/sparse_pcloud.html")

        shutil.rmtree(img_p)  # remove obj. only images
    else:
        logger.info("Loading .bin files into pycolmap.Reconstruction object.")
        model = pycolmap.Reconstruction(sfm_p)

    # <Image 'image_id=1, camera_id=1, name="0000.png", triangulated=314/354'>
    # img00 = model.images[1]
    # a = pycolmap.ListPoint2D(img00.points2D) # [<Point2D 'xy=[  693 815.5], point3D_id=575'>, ...]
    # kpts2d = [pt.xy for pt in a] # [array([693. , 815.5]), ...]
    # b = np.array(kpts2d)
    # _img = np.ascontiguousarray(frames[0])
    # from utils.colmap_utils import draw_circles
    # draw_circles(_img, b, (255,255,255), 5)
    # cv2.imwrite('tmp/test.png', _img[...,::-1])

    # clean up COLMAP pointcloud & export to ply_p
    v3d: trimesh.PointCloud = clean_pc(v3d=model.points3D, ply_p=ply_p)

    # sanity check: make sure poses are correct
    v2d, w2c_mats, K, converged_mask = verify_colmap(
        model=model, pc=v3d, n_frames=N, colmap_p=out_p, frames=frames
    )

    # TODO: add back SLERP (spherical linear interpolation) to interpolate failed poses
    #       check here: https://github.com/zc-alexfan/hold/blob/ed9188e57ffe490b6e739c5e2102358758cf7487/generator/src/colmap/colmap_utils.py#L71

    # TODO: add additional filter step, based on inlier count (optional)
    _, v3d_filtered, v2d_filtered = analyze_preprocessing(
        v2d=v2d,
        pc=v3d,
        frames=frames,
        masks=masks,
        reproj_threshold=0.45,
        save_animation=True,
        converged_frames=converged_mask,
        out_p=out_p,
    )

    # COLMAP predicts w2c mats w.r.t R=I, t=0 in OpenCV
    # MfV predicts SMPLX poses w.r.t world cam in OpenGL
    w2c_mats_gloss = colmap2gloss(w2c_mats, camera_extrinsics)

    # w2c_mats:             world2cam matrices for all frames (in GLOSS space); np.array; N x 4x4
    # K:                    cam intrinsics; np.array; 4x4; constant for all frames...;
    # v3d:                  original point cloud (trimmed & filtered) vertices;
    # v3d_rgb:              corresponding vertex colors
    # v2d:                  2D projections; np.array; N x N_verts x 2
    # converged_mask:       boolean array; (N,); True if corresponding entry in w2c_mats is valid, i.e. COLMAP found a pose;
    out: dict = {
        "w2c_mats": torch.FloatTensor(w2c_mats_gloss),
        "K": torch.FloatTensor(K[0]),  # camera intrinsics for full-resolution video!
        "v2d": torch.FloatTensor(v2d_filtered),
        "v3d": torch.FloatTensor(v3d_filtered.vertices),  # Keep original vertices
        "v3d_rgb": torch.FloatTensor(v3d_filtered.colors) / 255.0,
        "converged": torch.BoolTensor(converged_mask),
    }
    logger.info(f"Saving scratch data to {obj_data_p}.")
    for k, v in out.items():
        logger.info(f"--- {k}: {v.shape}")
    np.save(obj_data_p, out)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, help="sequence name")
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=40,
        help="number of the frames that the model is searching for connections",
    )
    parser.add_argument(
        "--gt_intrinsics",
        type=str,
        help="If known, provide camera intrinsics as f,cx,cy",
    )
    parser.add_argument(
        "--skip_sfm",
        action="store_true",
        help="skip SfM pipeline; for quick debugging; requires .bin files in bin_path",
    )
    args = parser.parse_args()
    main(args)
