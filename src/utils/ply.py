import torch
import numpy as np
import open3d as o3d
from typing import Union
from smplx import SMPLX, MANO, SMPL


def v3d_torch2ply(v3d_torch: torch.Tensor, path: str = "test.ply"):
    v3d_np = v3d_torch.cpu().detach().numpy()
    return v3d_np2ply(v3d_np, path)


def smpl2ply(model: Union[SMPLX, MANO, SMPL], path: str = "test.ply") -> None:
    return v3d_torch2ply(model.v_template, path)


def v3d_np2ply(v3d_np: np.ndarray, path: str = "test.ply"):
    """Convert a 3xn np.ndarray to PLY."""
    assert v3d_np.shape[1] == 3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v3d_np)
    o3d.io.write_point_cloud(path, pcd)
    return
