import torch
import torch.nn as nn
from smplx import SMPLX
from utils.constants import SMPLX_NEUTRAL_P


class SMPLXParameters(nn.Module):
    def __init__(self, data: dict, camera: dict, batch_size: int):
        super().__init__()

        self.N: int = len(data["global_orient"])

        # dont want gradients for these params; only modify object related parameters
        self.register_buffer("betas", nn.Parameter(data["betas"]))
        self.register_buffer("transl", nn.Parameter(data["transl"]))
        self.register_buffer("global_orient", nn.Parameter(data["global_orient"]))
        self.register_buffer("body_pose", nn.Parameter(data["body_pose"]))
        self.register_buffer("left_hand_pose", nn.Parameter(data["left_hand_pose"]))
        self.register_buffer("right_hand_pose", nn.Parameter(data["right_hand_pose"]))

        self.smplx_layer = SMPLX(
            SMPLX_NEUTRAL_P,
            # batch_size=self.N,
            batch_size=batch_size,
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
        ).cuda()

        self.camera_intrinsics = camera["intrinsics"]
        self.camera_extrinsics = camera["extrinsics"]

    def project2d(self, v3d) -> torch.Tensor:
        """Perspective Projection: 3D vertices in world coords. -> 2D points in pixel coords."""
        N = v3d.shape[0]

        # to homogeneous coordinates
        ones = torch.ones_like(v3d[..., :1])
        v3d_homo = torch.cat([v3d, ones], dim=-1)  # [N, J, 4]

        cam_extrinsics = self.camera_extrinsics.to(v3d.device)
        if cam_extrinsics.ndim == 2:
            cam_extrinsics = cam_extrinsics.unsqueeze(0).expand(N, -1, -1)

        # to camera coordinates
        v3d_cam = torch.einsum("nij,nkj->nki", cam_extrinsics, v3d_homo)
        v3d_cam = v3d_cam[..., :3]

        # perspective projection
        x = v3d_cam[..., 0] / v3d_cam[..., 2]
        y = v3d_cam[..., 1] / v3d_cam[..., 2]

        # apply camera intrinsics
        px = (
            self.camera_intrinsics[0, 0].float() * x
            + self.camera_intrinsics[0, 2].float()
        )
        py = (
            self.camera_intrinsics[1, 1].float() * y
            + self.camera_intrinsics[1, 2].float()
        )

        j2d = torch.stack([px, py], dim=-1)

        return j2d

    def forward(self, mask):
        device = self.betas.device
        mask = mask.to(device)
        s = self.smplx_layer(
            betas=self.betas[mask],
            global_orient=self.global_orient[mask],
            body_pose=self.body_pose[mask],
            left_hand_pose=self.left_hand_pose[mask],
            right_hand_pose=self.right_hand_pose[mask],
            transl=self.transl[mask],
            pose2rot=True,
        )

        N = len(mask)
        mask = mask.to(device)
        v3d_full = torch.full((N, *s.vertices.shape[1:]), float("nan"), device=device)
        j3d_full = torch.full((N, 55, 3), float("nan"), device=device)
        v2d_full = torch.full((N, s.vertices.shape[1], 2), float("nan"), device=device)
        j2d_full = torch.full((N, 55, 2), float("nan"), device=device)

        # Fill converged frames
        v3d_full[mask] = s.vertices
        j3d_full[mask] = s.joints[:, :55]

        v2d_proj = self.project2d(v3d=s.vertices)
        j2d_proj = self.project2d(v3d=s.joints[:, :55])
        v2d_full[mask] = v2d_proj
        j2d_full[mask] = j2d_proj

        out = {
            "v3d": v3d_full,
            "v2d": v2d_full,
            "j3d": j3d_full,
            "j2d": j2d_full,
            "betas": self.betas,
            "smplx_transl": self.transl,
            "smplx_global_orient": self.global_orient,
            "smplx_body_pose": self.body_pose,
            "smplx_left_hand_pose": self.left_hand_pose,
            "smplx_right_hand_pose": self.right_hand_pose,
        }
        return out
