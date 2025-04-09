import torch
import numpy as np
from utils.rotations import axis_angle_to_matrix, matrix_to_axis_angle


class ObjectParameters(torch.nn.Module):
    def __init__(self, data: dict, camera: dict):
        super().__init__()

        self.N = len(data["poses"])
        self.device = data["poses"].device

        # camera stuff
        self.camera_intrinsics = camera["intrinsics"]
        self.camera_extrinsics = camera["extrinsics"]

        # obj. related nn.Params (scale/pose/pts)
        obj_rot_matrix = data["poses"][:, :3, :3]
        obj_transl = data["poses"][:, :3, 3]
        obj_rot_aa = matrix_to_axis_angle(obj_rot_matrix)
        obj_scale = torch.FloatTensor(np.array([1.0]))

        # object pointcloud from COLMAP; openCV
        v3d_cano_homo = torch.cat(
            [
                data["v3d_cano"],
                torch.ones(len(data["v3d_cano"]), 1, device=self.device),
            ],
            dim=-1,
        )

        # optimize for these three parameters
        self.register_parameter("obj_scale", torch.nn.Parameter(obj_scale))
        self.register_parameter("obj_rot", torch.nn.Parameter(obj_rot_aa))
        self.register_parameter("obj_transl", torch.nn.Parameter(obj_transl))
        # leave unchanged
        self.register_buffer("v3d_orig_homo", v3d_cano_homo)

        self.j2d_gt = data["j2d.gt"]

    def project2d(self, v3d) -> torch.Tensor:
        """Perspective Projection: 3D vertices in world coords. -> 2D points in pixel coords."""

        # [N, V, 4]
        v3d_homo = torch.cat([v3d, torch.ones_like(v3d[..., :1])], dim=-1)

        # [N, 4, 4]
        N_valid = v3d.shape[0]
        cam_extrinsics = self.camera_extrinsics.unsqueeze(0).expand(N_valid, -1, -1)

        # to camera coords.
        v3d_cam = torch.einsum("nij,nkj->nki", cam_extrinsics, v3d_homo)
        v3d_cam = v3d_cam[..., :3]  # [N, V, 3]

        # perspective projection
        x = v3d_cam[..., 0] / v3d_cam[..., 2]  # [N, V]
        y = v3d_cam[..., 1] / v3d_cam[..., 2]  # [N, V]

        # Extract camera intrinsics
        f = self.camera_intrinsics[0, 0]
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]

        # cam intrinsics [N, V]
        px = f * x + cx
        py = f * y + cy

        # [N, V, 2]
        v2d = torch.stack([px, py], dim=-1)

        return v2d

    def forward(self, mask):
        N_valid = mask.sum()  # converge/valid frames

        rot_mat = axis_angle_to_matrix(self.obj_rot).to(self.device)
        w2c_mats = torch.eye(4, device=self.device).unsqueeze(0).repeat(N_valid, 1, 1)
        w2c_mats[:, :3, :3] = rot_mat[mask]
        w2c_mats[:, :3, 3] = self.obj_transl[mask]

        # canonical -> camera/posed space
        v3d_cano = self.v3d_orig_homo[:, :3] * self.obj_scale
        v3d_homo = torch.nn.functional.pad(v3d_cano, (0, 1), value=1.0)
        v3d_homo = v3d_homo.unsqueeze(0).expand(N_valid, -1, -1)
        v3d_cam_homo = torch.einsum("nij,nvj->nvi", w2c_mats, v3d_homo)
        v3d_pred = v3d_cam_homo[:, :, :3]

        # camera -> pixel space
        v2d_pred = self.project2d(v3d_pred)

        # fill non-converged frames with NaN
        N_total = len(mask)
        v3d_full = torch.full(
            (N_total, *v3d_pred.shape[1:]), float("nan"), device=self.device
        )
        v2d_full = torch.full(
            (N_total, *v2d_pred.shape[1:]), float("nan"), device=self.device
        )
        w2c_full = torch.full((N_total, 4, 4), float("nan"), device=self.device)

        v3d_full[mask] = v3d_pred
        v2d_full[mask] = v2d_pred
        w2c_full[mask] = w2c_mats

        obj_scale = torch.tensor([self.obj_scale.item()])

        out = {
            "j3d": v3d_full,
            "j2d": v2d_full,
            "obj_scale": obj_scale,
            "w2c_mats": w2c_full,
        }
        return out
