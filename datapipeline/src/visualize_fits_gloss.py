import cv2
import torch
import imageio
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from smplx import SMPLX
from typing import Union
from pathlib import Path
from loguru import logger
from utils import constants, video


try:
    from gloss import (
        Viewer,
        ViewerHeadless,
        VisMesh,
        MeshColorType,
        Colors,
        Verts,
        Faces,
        VisPoints,
        PointColorType,
        Edges,
        VisLines,  # , LineColorType,
    )
except:
    import sys

    logger.warning(f"Could not import gloss. ({sys.executable})")


def add_static_o3d_object(scene, o3d_object, object_name, point_size: float = 5.0):
    # copy from camera_motion/lib/pipeline/visualization.py
    mesh = scene.get_or_spawn_renderable(object_name)
    if type(o3d_object) == o3d.geometry.TriangleMesh:
        mesh.insert(Verts(np.asarray(o3d_object.vertices, dtype=np.float32)))
        mesh.insert(Colors(np.asarray(o3d_object.vertex_colors, dtype=np.float32)))
        mesh.insert(Faces(np.asarray(o3d_object.triangles, dtype=np.uint32)))
        mesh.insert(VisMesh(color_type=MeshColorType.PerVert))
    elif type(o3d_object) == o3d.geometry.PointCloud:
        mesh.insert(Verts(np.asarray(o3d_object.points, dtype=np.float32)))
        mesh.insert(Colors(np.asarray(o3d_object.colors, dtype=np.float32)))
        mesh.insert(
            VisPoints(
                show_points=True,
                point_size=point_size,
                color_type=PointColorType.PerVert,
            )
        )
    else:
        raise ValueError(f"Unsupported object type: {type(o3d_object)}")


def _save_pcd(v3d: np.ndarray, name: str) -> None:
    """Write .ply file to out_p."""
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(v3d)
    o3d.io.write_point_cloud(name, _pcd)
    return


def pcd_2d_to_pcd_3d(pcd, depth, intrinsic, cam2world=None):
    assert isinstance(pcd, np.ndarray), f"cannot process data type: {type(pcd)}"
    assert isinstance(
        intrinsic, np.ndarray
    ), f"cannot process data type: {type(intrinsic)}"
    assert len(pcd.shape) == 2 and pcd.shape[1] >= 2
    assert len(depth.shape) == 2 and depth.shape[1] == 1
    assert intrinsic.shape == (3, 3)
    if cam2world is not None:
        assert isinstance(
            cam2world, np.ndarray
        ), f"cannot process data type: {type(cam2world)}"
        assert cam2world.shape == (4, 4)

    x, y, z = pcd[:, 0], pcd[:, 1], depth[:, 0]
    append_ones = np.ones_like(x)
    xyz = np.stack([x, y, append_ones], axis=1)  # shape: [num_points, 3]
    inv_intrinsic_mat = np.linalg.inv(intrinsic)
    xyz = np.matmul(inv_intrinsic_mat, xyz.T).T * z[..., None]
    valid_mask_1 = np.where(xyz[:, 2] > 0)
    xyz = xyz[valid_mask_1]

    if cam2world is not None:
        append_ones = np.ones_like(xyz[:, 0:1])
        xyzw = np.concatenate([xyz, append_ones], axis=1)
        xyzw = np.matmul(cam2world, xyzw.T).T
        valid_mask_2 = np.where(xyzw[:, 3] != 0)
        xyzw = xyzw[valid_mask_2]
        xyzw /= xyzw[:, 3:4]
        xyz = xyzw[:, 0:3]

    if pcd.shape[1] > 2:
        features = pcd[:, 2:]
        try:
            features = features[valid_mask_1][valid_mask_2]
        except UnboundLocalError:
            features = features[valid_mask_1]
        assert xyz.shape[0] == features.shape[0]
        xyz = np.concatenate([xyz, features], axis=1)
    return xyz


def shoot_ray(x, y, cam_int, cam_ext, center):
    z = np.array([[1]])
    xy = np.array([[x, y]])
    pcd_3d = pcd_2d_to_pcd_3d(xy, z, cam_int, cam2world=cam_ext)[0].astype(np.float32)
    dir = pcd_3d - center
    dir = dir / np.linalg.norm(dir)
    return center, dir


def get_camera_poly(cam_int, cam_ext, center=None, size=1):
    # 3rd Row -> up direction
    # 2nd Col -> up vector
    up_direction = cam_ext[:3, :3].T[:, 1]  

    if center is None:
        center = -cam_ext[:3, :3].T @ cam_ext[:3, 3]
    cam_ext = np.linalg.inv(cam_ext)

    w, h = cam_int[:2, 2] * 2
    _, dir_0 = shoot_ray(0, 0, cam_int, cam_ext, center)
    _, dir_1 = shoot_ray(w, 0, cam_int, cam_ext, center)
    _, dir_2 = shoot_ray(w, h, cam_int, cam_ext, center)
    _, dir_3 = shoot_ray(0, h, cam_int, cam_ext, center)

    pt_0 = dir_0 * size + center
    pt_1 = dir_1 * size + center
    pt_2 = dir_2 * size + center
    pt_3 = dir_3 * size + center

    up_indicator = center + up_direction * (size * 0.5)

    points = np.array([center, pt_0, pt_1, pt_2, pt_3, up_indicator])
    edges = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 4],
            [0, 5],  # Edge to up direction indicator
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [3, 4, 1]])

    return points, edges, faces


@torch.no_grad()
def visualize(
    camera_extrinsics,
    camera_intrinsics,
    v3d_h,
    j3d_h,
    v3d_o,
    out_p: Union[str, Path],
    frames: np.array,
    fps: float = 30,
    debug: bool = False,
    side_view: bool = False,
) -> None:
    """
    1. Visualize the TRAM & COLMAP output in Gloss.
    2. Save relevant files for alignment stage.

    NOTE: This is heavily inspired by `visualize_mfv_gloss`
          Check there for more details.

    Args:
        out_p (str): out path for visualization/files/...
        mfv_data (dict): data from MfV step, results.pkl after demo.py
        obj_data (dict): data from COLMAP step, obj_data.npy
        images (_type_): _description_
        debug (bool, optional): only save first frame & break. Defaults to False.
        fps (int, optional): Defaults to 30.
        data_dir (str, optional): Path to body models. Defaults to "data/".
    """
    # ensure proper types (important for GLOSS)
    if isinstance(out_p, Path):
        out_p = str(out_p)
    if isinstance(v3d_h, torch.Tensor):
        v3d_h = v3d_h.detach().numpy().astype(np.float32)
    if isinstance(j3d_h, torch.Tensor):
        j3d_h = j3d_h.detach().numpy().astype(np.float32)
    if isinstance(v3d_o, torch.Tensor):
        v3d_o = v3d_o.detach().numpy().astype(np.float32)

    writer = imageio.get_writer(out_p, **video.get_writer_cfg(fps=fps, height=896))
    frame = frames[0][..., ::-1]
    viewer = ViewerHeadless(frame.shape[1], frame.shape[0], constants.gloss_cfg_p)
    side_viewer: Union[ViewerHeadless, None] = (
        ViewerHeadless(frame.shape[1], frame.shape[0], constants.gloss_cfg_p)
        if side_view
        else None
    )

    camera = viewer.get_camera()
    camera.set_extrinsics(camera_extrinsics.astype(np.float32))
    cam_focal = camera_intrinsics[0, 0]
    cx, cy = camera_intrinsics[:2, 2]
    camera.set_intrinsics(cam_focal, cam_focal, cx, cy)

    if side_view:
        side_cam = side_viewer.get_camera()
        side_cam.set_intrinsics(cam_focal, cam_focal, cx, cy)

        # mean position of human joints across all *valid* frames
        valid_frames = [i for i, j in enumerate(j3d_h) if not np.isnan(j).all()]
        valid_j3d = j3d_h[valid_frames]
        lookat = valid_j3d.mean(axis=(0, 1))

        # center = np.array([-1.0, lookat[1], 0.0])
        center = lookat + np.array([3, 0.0, 0.0])  # cam to the left

        up = [0, 1, 0]
        side_cam.set_up(up)
        side_cam.set_lookat(lookat)
        side_cam.set_position(center)

    # init smplx for face indices
    smplx = SMPLX(
        constants.SMPLX_NEUTRAL_P,
        batch_size=1,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
    )

    # visualize scene!
    for i, frame in tqdm(enumerate(frames), desc="Gloss"):
        viewer.start_frame()
        scene = viewer.get_scene()

        if side_view:
            side_viewer.start_frame()
            side_scene = side_viewer.get_scene()

            cam_mesh = side_scene.get_or_spawn_renderable(f"camera")
            v, e, f = get_camera_poly(
                cam_int=camera_intrinsics.astype(np.float32),
                cam_ext=camera_extrinsics.astype(np.float32),
                size=0.5,
            )
            cam_mesh.insert(Verts(v.astype(np.float32)))
            cam_mesh.insert(Edges(e.astype(np.uint32)))
            cam_mesh.insert(VisLines(show_lines=True, line_width=5.0))
            cam_mesh.insert(VisPoints(show_points=True, point_size=5.0))

        # extract vertices for this frame
        _v3d_h = v3d_h[i]
        _j3d_h = j3d_h[i]
        _v3d_o = v3d_o[i]

        valid_frame: bool = not np.isnan(_v3d_h).all()

        if valid_frame:
            # add smplx verts pointcloud (red)
            v3d_h_o3d = o3d.geometry.PointCloud()
            v3d_h_o3d.points = o3d.utility.Vector3dVector(_v3d_h)
            v3d_h_o3d.colors = o3d.utility.Vector3dVector(
                np.ones_like(_v3d_h) * np.array([1.0, 0.0, 0.0])
            )
            add_static_o3d_object(scene, v3d_h_o3d, "v3d_h", point_size=10)

            # add smplx joints pointcloud (blue)
            j3d_h_o3d = o3d.geometry.PointCloud()
            j3d_h_o3d.points = o3d.utility.Vector3dVector(_j3d_h)
            j3d_h_o3d.colors = o3d.utility.Vector3dVector(
                np.ones_like(_j3d_h) * np.array([0.0, 0.0, 1.0])
            )
            add_static_o3d_object(scene, j3d_h_o3d, "j3d_h", point_size=10)

            # add smplx as mesh
            mesh = scene.get_or_spawn_renderable(f"smpl_mesh")
            mesh.insert(Verts(_v3d_h))
            mesh.insert(Faces(smplx.faces.astype(np.uint32)))
            vert_col = 0.5 * np.ones_like(_v3d_h, dtype=np.float32)
            mesh.insert(Colors(vert_col))
            mesh.insert(VisMesh(color_type=MeshColorType.PerVert))

            # add object pointcloud (green)
            v3d_o_o3d = o3d.geometry.PointCloud()
            v3d_o_o3d.points = o3d.utility.Vector3dVector(_v3d_o)
            v3d_o_o3d.colors = o3d.utility.Vector3dVector(
                np.ones_like(_v3d_o) * np.array([0.0, 1.0, 0.0])
            )
            add_static_o3d_object(scene, v3d_o_o3d, "v3d_o", point_size=10)

            if side_view:
                add_static_o3d_object(side_scene, v3d_h_o3d, "v3d_h", point_size=10)
                add_static_o3d_object(side_scene, j3d_h_o3d, "j3d_h", point_size=10)
                add_static_o3d_object(side_scene, v3d_o_o3d, "v3d_o", point_size=10)
                mesh = side_scene.get_or_spawn_renderable(f"smpl_mesh")
                mesh.insert(Verts(_v3d_h))
                mesh.insert(Faces(smplx.faces.astype(np.uint32)))
                vert_col = 0.5 * np.ones_like(_v3d_h, dtype=np.float32)
                mesh.insert(Colors(vert_col))
                mesh.insert(VisMesh(color_type=MeshColorType.PerVert))

            if debug:
                _save_pcd(v3d=_v3d_o, name=f"tmp/{i:04d}_v3d_o.ply")
                _save_pcd(v3d=_j3d_h, name=f"tmp/{i:04d}_j3d_h.ply")
                _save_pcd(v3d=_v3d_h, name=f"tmp/{i:04d}_v3d_h.ply")

        viewer.update()

        if side_view:
            side_viewer.update()

        tex_numpy = (
            viewer.get_final_tex()
            .numpy(viewer.get_device(), viewer.get_queue())
            .astype(np.uint8)
        )

        alpha = tex_numpy[:, :, 3].astype(np.float32) / 255.0
        rgb = tex_numpy[:, :, :3].astype(np.float32)
        overlay = alpha[:, :, np.newaxis] * rgb.astype(np.float32)
        bg = (1.0 - alpha)[:, :, np.newaxis] * frame.astype(np.float32)
        rend = (bg + overlay).astype(np.uint8)

        if side_view:
            side_tex = (
                side_viewer.get_final_tex()
                .numpy(side_viewer.get_device(), side_viewer.get_queue())
                .astype(np.uint8)
            )
            side_rend = side_tex[:, :, :3]
            rend = np.concatenate([rend, side_rend], axis=1)

        if debug:
            out_frame = np.concatenate([frame, rend], axis=1)
            cv2.imwrite(f"tmp/tmp.png", out_frame[..., ::-1])
            logger.info(f"debug frame saved to: ./tmp/tmp.png")
            break

        # add render to video writer
        writer.append_data(rend)

        # reset scene
        for rn in [x for x in scene.get_renderable_names()]:
            scene.remove_renderable(rn)

        if side_view:
            for rn in [x for x in side_scene.get_renderable_names()]:
                side_scene.remove_renderable(rn)

    writer.close()
    logger.success(f"Find results at: {out_p}")
    return


def main(args) -> None:

    logger.info("Visualizing aligned data...")

    # set paths
    data_p: Path = Path(f"./data/{args.seq_name}")
    alignment_p = data_p / "scratch" / "alignment"
    assert alignment_p.exists(), alignment_p
    aligned_data_p = data_p / "data.pt"
    video_p = data_p / "video.mp4"
    out_p = alignment_p / "vis_fit.mp4"

    # load data/video/configs
    aligned_data = torch.load(aligned_data_p, map_location="cpu", weights_only=True)
    frames = video.video2frames(
        video_p=video_p,
        return_images=True,
    )
    frames = np.array(frames)

    camera_extrinsics = aligned_data["camera"]["extrinsics"].numpy().astype(np.float32)
    camera_intrinsics = aligned_data["camera"]["intrinsics"].numpy().astype(np.float32)

    if args.debug:
        from utils import debug

        logger.info("Aligned `data.pt` contains:")
        debug.log(aligned_data)

        vis_frame = frames[0].copy()
        w = vis_frame.shape[1]
        h = vis_frame.shape[0]
        for x, y in aligned_data["object"]["j2d"][0]:  # red
            x = int(torch.clip(x, 0, w - 1).item())
            y = int(torch.clip(y, 0, h - 1).item())
            cv2.circle(vis_frame, (x, y), radius=3, color=(255, 0, 0), thickness=-1)
        for x, y in aligned_data["human"]["j2d"][0]:  # green
            x = int(torch.clip(x, 0, w - 1).item())
            y = int(torch.clip(y, 0, h - 1).item())
            cv2.circle(vis_frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
        for x, y in aligned_data["human"]["v2d"][0]:  # blue
            x = int(torch.clip(x, 0, w - 1).item())
            y = int(torch.clip(y, 0, h - 1).item())
            cv2.circle(vis_frame, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

        cv2.imwrite("./tmp/debug_alignemt.png", vis_frame[..., ::-1])
        return

    visualize(
        camera_extrinsics=camera_extrinsics,
        camera_intrinsics=camera_intrinsics,
        v3d_h=aligned_data["human"]["v3d"],
        j3d_h=aligned_data["human"]["j3d"],
        v3d_o=aligned_data["object"]["j3d"],
        frames=frames,
        out_p=out_p,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
