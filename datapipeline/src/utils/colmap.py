import cv2
import trimesh
import imageio
import pycolmap
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Slerp, Rotation
from utils.video import get_writer_cfg


def colmap2gloss(w2c_mat, cam_extrinsics) -> np.ndarray:
    """Convert COLMAP camera matrices to GLOSS camera matrices.

    Args:
        w2c_mat: world to (colmap) camera matrix [N, 4, 4]
        cam_extrinsics: (gloss) camera extrinsics [4, 4]
    Returns:
        w2c_mat_gloss: world to (gloss) camera matrix [N, 4, 4]
    """
    if w2c_mat.ndim == 3:
        N = w2c_mat.shape[0]
    else:
        N = 1
        w2c_mat = w2c_mat[None]  # [1, 4, 4]

    R = cam_extrinsics[:3, :3]
    t = cam_extrinsics[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t

    T = np.eye(4)
    T[:3, :3] = R_inv
    T[:3, 3] = t_inv

    w2c_mat_gloss = np.zeros_like(w2c_mat)
    for i in range(N):
        w2c_mat_gloss[i] = T @ w2c_mat[i]

    return w2c_mat_gloss.astype(np.float32)


def project_3d_points(
    model: pycolmap.Reconstruction, img: pycolmap.Image, pc: trimesh.PointCloud
):
    """Project points3D onto the image using the camera intrinsics and pose from COLMAP."""
    # extract camera intrinsics...
    camera = model.cameras[img.camera_id]
    f = camera.params[0]
    cx, cy = camera.params[1], camera.params[2]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    # ...and w2c pose
    R = img.rotation_matrix()
    t = img.tvec

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t

    # projection matrix
    P = K @ np.hstack((R, t.reshape(3, 1)))

    # project 3D points to 2D image coordinates
    points_3d_homogeneous = np.hstack((pc.vertices, np.ones((pc.vertices.shape[0], 1))))
    points_2d_homogeneous = (P @ points_3d_homogeneous.T).T
    v2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    return v2d, w2c, K


def draw_circles(img, points, color, radius=2):
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)


def process_image(
    image_id: int,
    model: pycolmap.Reconstruction,
    pc: trimesh.PointCloud,
    imgs: np.ndarray,
):
    n_v3d: int = len(pc.vertices)

    if image_id not in model.images:
        _img = imgs[image_id - 1]
        _img = np.ascontiguousarray(_img)  # hack for cv2 compatibility
        f = cv2.FONT_HERSHEY_SIMPLEX
        t = "no pose found"
        s = cv2.getTextSize(t, f, 1, 2)[0]
        x = (_img.shape[1] - s[0]) // 2
        y = 30 + s[1]
        cv2.putText(_img, t, (x, y), f, 1, (255, 0, 0), 2)

        return (
            _img,
            np.full((n_v3d, 2), np.nan),
            np.full((4, 4), np.nan),
            np.full((3, 3), np.nan),
        )

    img: pycolmap.Image = model.images[image_id]
    img_name: str = img.name
    img_id: int = int(img_name.split(".")[0])
    _img = imgs[img_id]
    _img = np.ascontiguousarray(_img)  # hack for cv2 compatibility

    v2d, w2c, K = project_3d_points(model, img, pc)
    h, w = _img.shape[:2]

    # clip points to image boundaries
    pts = np.clip(v2d, [0, 0], [w - 1, h - 1])
    mid = len(pts) // 2
    valid_mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    draw_circles(_img, pts[:mid][valid_mask[:mid]], (0, 0, 255), radius=10)  # blue
    draw_circles(_img, pts[mid:][valid_mask[mid:]], (255, 0, 0), radius=10)  # red

    return _img, v2d, w2c, K


def verify_colmap(
    model: pycolmap.Reconstruction,
    pc: trimesh.PointCloud,
    n_frames: int,
    colmap_p: Path,
    frames: np.ndarray,
):
    """
    Project the point cloud onto the images to verify the COLMAP model.
    v2d_array -> (n_frames, n_v3d, 2)
    w2c_array -> (n_frames, 4, 4)
    K_array -> (n_frames, 3, 3)
    converged_mask -> (n_frames,); boolean array
    """
    with ThreadPoolExecutor() as executor:
        results = []
        with imageio.get_writer(
            colmap_p / "vis_colmap.mp4", **get_writer_cfg(height=896)
        ) as writer:
            # process frames in parallel but write sequentially to maintain order
            for image_id in range(1, n_frames + 1):
                frame_data = process_image(
                    image_id=image_id, model=model, pc=pc, imgs=frames
                )
                writer.append_data(frame_data[0])
                results.append(frame_data[1:])

    logger.info(f"Sanity Check: {colmap_p}/vis_colmap.mp4")

    # merge results; all have len == n_frames
    # frames where COLMAP could not find a pose, have np.nan arrays
    v2d_array, w2c_array, K_array = map(np.stack, zip(*results))
    converged_mask = np.zeros(n_frames, dtype=bool)
    converged_mask[[i - 1 for i in model.images]] = True
    logger.info(
        f"COLMAP found poses for {converged_mask.sum()}/{n_frames} frames ({converged_mask.sum()/n_frames:.1%})"
    )

    return v2d_array, w2c_array, K_array, converged_mask


def interpolate_poses(w2c_all, key_frames):
    """Interpolate w2c matrices containing object poses in OpenCV format for all frames using spherical interpolation (SLERP).

    Args:
        w2c_all (np.array): Array containing the w2c projection matrices for all frames. Nx4x4
        key_frames (np.array): ...; Nx1
    """
    print(f"Interpolating missing poses")
    num_frames = w2c_all.shape[0]
    expected_frames = np.arange(num_frames)  # arrary from 0 - num_frames-1

    # NOTE: not sure why this is necessary; leaving it just in case
    start_time, end_time = key_frames[0], key_frames[-1]
    start_o2w, end_o2w = w2c_all[:1], w2c_all[-1:]
    start_time_query, end_time_query = expected_frames[0], expected_frames[-1]

    if start_time_query < start_time:
        w2c_all = np.concatenate((start_o2w, w2c_all), axis=0)
        key_frames = np.concatenate(([start_time_query], key_frames), axis=0)

    if end_time < end_time_query:
        w2c_all = np.concatenate((w2c_all, end_o2w), axis=0)
        key_frames = np.concatenate((key_frames, [end_time_query]), axis=0)

    # interpolate rotation
    rots = w2c_all[:, :3, :3]
    key_rots = Rotation.from_matrix(rots)
    slerp = Slerp(key_frames, key_rots)
    interp_rots = slerp(expected_frames).as_matrix()

    # interpolate translation (x, y, z separately)
    key_trans = w2c_all[:, :3, 3]
    interp_trans_x = np.interp(expected_frames, key_frames, key_trans[:, 0])
    interp_trans_y = np.interp(expected_frames, key_frames, key_trans[:, 1])
    interp_trans_z = np.interp(expected_frames, key_frames, key_trans[:, 2])
    interp_trans = np.vstack([interp_trans_x, interp_trans_y, interp_trans_z]).T

    # Create the interpolated w2c_mats matrix
    w2c_all_interpolated = np.zeros((num_frames, 4, 4))
    w2c_all_interpolated[:, :3, :3] = interp_rots
    w2c_all_interpolated[:, :3, 3] = interp_trans
    w2c_all_interpolated[:, 3, 3] = 1

    return w2c_all_interpolated


def analyze_preprocessing(
    v2d: np.ndarray,
    pc: trimesh.PointCloud,
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    inlier_threshold: float = 0.3,
    reproj_threshold: float = 0.5,
    save_animation: bool = False,
    converged_frames: List[int] = None,
    out_p: Path = Path.cwd(),
) -> Tuple[List[int], trimesh.PointCloud]:
    """Analyze the pre-processed data from COLMAP.
    Find frames with poor initial poses & track point reliability across frames.

    Args:
        v2d: 2D projections, shape (N_frames, N_verts, 2)
        K: Camera intrinsics matrix, shape (3, 3)
        w2c_mats: World to camera matrices, shape (N_frames, 4, 4)
        pc: Input point cloud
        msk_dir: Directory containing segmentation masks
        msk_paths: List of paths to segmentation masks
        inlier_threshold: Threshold for good frames (default: 0.3)
        reproj_threshold: Threshold for good vertices (default: 0.5)
        save_animation: Whether to save debug visualization (default: False)

    Returns:
        Tuple[List[int], trimesh.PointCloud]: Bad frame indices and filtered point cloud
    """
    bad_frames: List[int] = []
    point_inlier_counts = np.zeros(len(pc.vertices))

    if save_animation:
        writer = imageio.get_writer(
            out_p / "vis_pcloud.mp4",
            fps=30,
            mode="I",
            format="FFMPEG",
            macro_block_size=1,
        )

    masks = np.array(masks)

    for i in range(len(masks)):
        # skip frame if it did not converge
        if not converged_frames[i]:
            continue

        # msk_o for current frame
        msk = masks[i]
        msk_binary = (msk == 50).astype(np.uint8)[:, :, 0]
        h, w = msk.shape[:2]

        # v2d_o for current frame
        pts = v2d[i]
        pts = np.clip(pts, [0, 0], [w - 1, h - 1])

        # Valid points inside image
        valid_mask = (
            (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        )
        valid_pts = pts[valid_mask]

        # Check if valid points are inliers
        inliers_mask = msk_binary[
            valid_pts[:, 1].astype(int), valid_pts[:, 0].astype(int)
        ]

        # Count inliers including invalid points for consistency
        all_inliers = msk_binary[pts[:, 1].astype(int), pts[:, 0].astype(int)]
        num_inliers = np.sum(all_inliers)
        num_points = len(pts)
        inlier_percentage = (num_inliers / num_points) * 100

        # Update inlier counts
        point_inlier_counts[all_inliers > 0] += 1

        # Mark frames with low inlier percentage as bad
        if inlier_percentage <= inlier_threshold * 100:
            bad_frames.append(i)

        if save_animation:
            inlier_pts = valid_pts[inliers_mask > 0]
            outlier_pts = valid_pts[inliers_mask == 0]
            draw_circles(msk, inlier_pts, (0, 255, 0), radius=1)
            draw_circles(msk, outlier_pts, (0, 0, 255), radius=1)
            text = f"Inliers: {num_inliers}/{num_points} ({inlier_percentage:.1f}%)"
            cv2.putText(
                msk, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            cv2.putText(msk, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            writer.append_data(msk)  # msk has to be HWC

    if save_animation:
        writer.close()

    # Filter points based on inlier percentage across frames
    point_inlier_percentages = point_inlier_counts / len(v2d) * 100
    good_v3d = point_inlier_percentages > (reproj_threshold * 100)

    # Create filtered point cloud
    v2d_filtered = v2d[:, good_v3d]
    v3d_filtered = trimesh.PointCloud(
        vertices=pc.vertices[good_v3d],
        colors=pc.colors[good_v3d] if pc.colors is not None else None,
    )
    v3d_filtered.export(out_p / "v3d_o_filtered.ply")
    logger.info(f"Saved cleaned point cloud to {out_p}/v3d_o.ply")
    logger.info(f"Check out visualization at {out_p}/vis_pcloud.mp4")
    logger.warning(
        f"{len(v3d_filtered.vertices)}/{len(v2d[0])} are considered 'valid'."
    )

    if save_animation:
        writer = imageio.get_writer(
            out_p / "vis_colmap.mp4", **get_writer_cfg(height=896)
        )

        v2d_filtered = v2d_filtered.astype(np.int32)
        for frame_idx, frame in enumerate(frames):
            vis_frame = frame.copy()
            points = v2d_filtered[frame_idx]
            for pt in points:
                cv2.circle(vis_frame, tuple(pt), 2, (0, 255, 0), -1)
            writer.append_data(vis_frame)

        writer.close()

    return bad_frames, v3d_filtered, v2d_filtered


def clean_pc(
    v3d, percentile: int = 80, scale_factor: float = 1.5, ply_p: Path = None
) -> trimesh.PointCloud:
    """
    Remove outliers (points that are far from the median) from point cloud.
    v3d is of type <class 'pycolmap.MapPoint3DIdPoint3D'>
    Export final object point cloud to ply_p.
    """
    # construct trimesh point cloud
    pc = trimesh.PointCloud(
        vertices=np.array([p.xyz for p in v3d.values()]),
        colors=np.array([p.color for p in v3d.values()]),
    )
    center = np.median(pc.vertices, axis=0)
    dist = np.linalg.norm(pc.vertices - center[None, :], axis=1)
    thresh = np.percentile(dist, percentile)
    thresh = scale_factor * thresh
    mask = dist < thresh
    pc_trim = trimesh.PointCloud(vertices=pc.vertices[mask], colors=pc.colors[mask])
    logger.info(f"Removed outliers from SfM model ({ply_p})")
    pc_trim.export(ply_p)
    return pc_trim
