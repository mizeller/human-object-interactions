import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Iterator, Tuple
from dataclasses import dataclass

from src import constants
from src.utils import graphics


@dataclass
class ViewPoint:
    cam_intrinsics: torch.Tensor  # ≡ Camera.K
    image_height: int  # ≡ Camera.height
    image_width: int  # ≡ Camera.width
    world_view_transform: torch.Tensor
    c2w: Optional[torch.Tensor] = None
    projection_matrix: Optional[torch.Tensor] = None
    full_proj_transform: Optional[torch.Tensor] = None


class CameraTrajectory:
    def __init__(
        self,
        positions: Union[torch.Tensor, None] = None,  # (N, 3)
        centers: Union[torch.Tensor, None] = None,  # (N, 3)
        up_vectors: Union[torch.Tensor, None] = None,  # (N, 3)
        cache_p: Union[str, Path, None] = None,
    ):
        self.device: torch.device = constants.device
        self.positions = positions if positions else None
        self.centers = centers if centers else None
        self.up_vectors = up_vectors if up_vectors else None

        # load from disk
        if cache_p is not None:
            self.load(traj_p=cache_p)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict:
        return dict(
            position=self.positions[idx],
            center=self.centers[idx],
            up_vector=self.up_vectors[idx],
        )

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    def scale(self, radius: float):
        """Scale a camera trajectory to a new radius."""
        center = self.centers[0]  # assuming const. center
        current_radius = torch.norm(self.positions[0] - center)
        scale_factor = radius / current_radius
        new_positions = center + (self.positions - center) * scale_factor
        self.positions = new_positions
        return

    def load(self, traj_p: Union[str, Path]) -> None:
        traj = torch.load(traj_p)
        self.centers = traj["centers"]
        self.positions = traj["positions"]
        self.up_vectors = traj["up_vectors"]
        return

    def save(self, file_p: Union[str, Path]) -> None:
        """Save camera trajectory to disk."""
        torch.save(
            {
                "positions": self.positions,
                "centers": self.centers,
                "up_vectors": self.up_vectors,
            },
            file_p,
        )
        return

    def generate_spiral_trajectory(
        self,
        n_frames: int = 50,
        radius: float = 1.0,
        n_spirals: int = 1,
        center: Optional[torch.Tensor] = None,
    ):
        """
        Precompute camera positions, look-at points, and up vectors for a generic spiral trajectory around the world origin (at radius = 1).
        """
        if center is None:
            center = torch.zeros(3, device=self.device)

        t = torch.linspace(-1, 1, n_frames, device=self.device)
        latitude = torch.arcsin(t)
        longitude = 2 * torch.pi * n_spirals * t

        # Calculate camera positions
        positions = torch.stack(
            [
                radius * torch.cos(latitude) * torch.cos(longitude),
                radius * torch.sin(latitude),
                radius * torch.cos(latitude) * torch.sin(longitude),
            ],
            dim=1,
        )
        positions = positions + center

        # Calculate up vectors
        up_vectors = []
        centers = center.expand(n_frames, 3)  # All cameras look at center

        for pos in positions:
            forward = center - pos
            forward = forward / torch.norm(forward)

            world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            if abs(forward[1]) > 0.999:  # Looking straight up/down
                world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)

            right = torch.cross(forward, world_up)
            right = right / torch.norm(right)
            up = torch.cross(forward, right)
            up_vectors.append(up)

        up_vectors = torch.stack(up_vectors)

        self.positions = positions
        self.centers = centers
        self.up_vectors = up_vectors
        return


class Camera:
    def __init__(
        self,
        image_size: Tuple[int, int],  # (H, W)
        world_view_transform: torch.Tensor,  # [R|t], 4x4
        cam_intrinsics: torch.Tensor,  # 3x3
        cam_id: int = 0,
    ):
        self.device: torch.device = constants.device

        self.camera_id: int = cam_id
        self.image_height: int = image_size[0]
        self.image_width: int = image_size[1]

        # world coordinates, openCV convention
        self.world_view_transform = world_view_transform.to(self.device)
        self.R = world_view_transform[:3, :3]
        self.t = world_view_transform[:3, 3]

        self.cam_intrinsics = cam_intrinsics.to(self.device)
        self.fx: float = self.cam_intrinsics[0, 0].item()
        self.fy: float = self.cam_intrinsics[1, 1].item()
        self.fovx: float = graphics.focal2fov(focal=self.fx, pixels=self.image_width)
        self.fovy: float = graphics.focal2fov(focal=self.fy, pixels=self.image_height)

        self.near: float = 0.01
        self.far: float = 100.0

        self.trajectory: CameraTrajectory = CameraTrajectory(
            cache_p=constants.assets_p / "cam_orbit.pt"
        )
        self.viewpoints: List[ViewPoint] = None

    def visualize_trajectory(
        self, viewpoints: List[ViewPoint], v3d: torch.Tensor, out_p: Path
    ):
        positions = torch.stack([v.c2w[:3, 3] for v in viewpoints])
        fig = plt.figure(figsize=(20, 20))
        views = [(20, 45), (20, 135), (20, 225), (20, 315)]
        for idx, (elev, azim) in enumerate(views, 1):
            ax = fig.add_subplot(2, 2, idx, projection="3d")

            ax.scatter(
                positions[:, 0].cpu(),
                positions[:, 1].cpu(),
                positions[:, 2].cpu(),
                c=range(len(positions)),
                cmap="viridis",
            )

            for v in viewpoints[::5]:
                pos = v.c2w[:3, 3].cpu()
                # Forward direction is Z axis
                forward = v.c2w[:3, 2].cpu()
                ax.quiver(
                    pos[0],
                    pos[1],
                    pos[2],
                    forward[0],
                    forward[1],
                    forward[2],
                    length=0.05,
                    color="red",
                )

            vertices = v3d.cpu().numpy()
            ax.scatter(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                color="blue",
                s=1,
                alpha=0.5,
                marker=".",
            )

            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.2)
            ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.2)
            ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.2)

        fig.suptitle("Object Orbit for Mesh Extraction", fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 0.92, 0.95])
        plt.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close()

    def get_viewpoints(self) -> List[ViewPoint]:
        viewpoint_stack = []
        for step in self.trajectory:
            viewpoint: ViewPoint = self._compute_viewpoint(
                camera_position=step["position"],
                center=step["center"],
                up=step["up_vector"],
            )
            viewpoint_stack.append(viewpoint)

        return viewpoint_stack

    def _compute_viewpoint(
        self,
        camera_position: torch.Tensor,
        center: torch.Tensor,
        up: torch.Tensor,
    ) -> ViewPoint:
        """Compute view points for a camera position"""
        # look-at matrix
        forward = center - camera_position
        forward = forward / torch.norm(forward)
        right = torch.linalg.cross(forward,up)
        right = right / torch.norm(right)

        # camera-to-world matrix
        c2w = torch.eye(4, device=constants.device)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = camera_position

        # world-to-camera matrix
        w2c = torch.eye(4, device=constants.device)
        rot = c2w[:3, :3]
        w2c[:3, :3] = rot.T
        w2c[:3, 3] = -torch.matmul(rot.T, camera_position)

        # perspective projection matrix
        proj = torch.zeros((4, 4), device=constants.device)
        fx = self.image_width / (2 * np.tan(self.fovx / 2))
        fy = self.image_height / (2 * np.tan(self.fovy / 2))

        proj[0, 0] = 2 * fx / self.image_width
        proj[1, 1] = 2 * fy / self.image_height
        proj[2, 2] = -(self.far + self.near) / (self.far - self.near)
        proj[2, 3] = -2 * self.far * self.near / (self.far - self.near)
        proj[3, 2] = -1.0

        viewpoint: ViewPoint = ViewPoint(
            cam_intrinsics=self.cam_intrinsics,
            c2w=c2w,
            world_view_transform=w2c,
            projection_matrix=proj,
            full_proj_transform=proj @ w2c,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        return viewpoint
