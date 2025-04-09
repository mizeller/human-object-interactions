# Source: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/mesh_utils.py
#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
import torch
import imageio
import torchvision
import numpy as np
from tqdm import tqdm
import open3d as o3d
from pathlib import Path
from typing import List

from src import constants, utils, model
from src.renderer.render import render


class GaussianExtractor(object):
    def __init__(
        self,
        seq_id: str,
        gaussians: model.object_gs.ObjectGS,
        cam: utils.camera.Camera,
        bg_color=None,
    ):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render)
        >>> gaussExtrator.reconstruction(viewpoints)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        self.device = constants.device
        self.seq: str = seq_id
        self.gaussians: model.object_gs.ObjectGS = gaussians

        # set the active_sh to 0 to export only diffuse texture
        self.gaussians.active_sh_degree = 0

        self.bg_color = bg_color
        self.camera = cam
        self.center = torch.zeros(3, device=self.device)

        # mesh extraction attributes
        self.radius: float = 1.0
        self.depth_trunc: float = self.radius * 2

        self.clean()

    def get_object_radius(
        self,
        margin_factor: float = 1.2,
        min_radius: float = 0.05,
        max_radius: float = 0.5,
    ) -> float:
        """Calculate appropriate camera radius with bounds."""
        distances = torch.norm(self.gaussians.out["xyz"], dim=1)
        max_distance = distances.max().item()
        radius = max_distance * margin_factor
        radius = max(max_radius, min(min_radius, radius))
        print(f"selected radius: {radius}")
        return radius

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.rgbmaps = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, out_p: Path):
        """
        reconstruct radiance field given cameras
        """
        self.clean()

        # center point cloud in world origin
        self.gaussians.forward(
            obj_rot=torch.eye(3, device=self.device),
            obj_trans=torch.zeros(3, device=self.device),
        )
        self.gaussians.out["xyz"] -= self.gaussians.out["xyz"].mean(dim=0)

        # get radius, s.t camera observe whole object
        self.radius = self.get_object_radius()

        # scale camera trajectory accordingly
        self.camera.trajectory.scale(radius=self.radius)

        # compute view-point stack
        viewpoint_stack: List[utils.camera.ViewPoint] = self.camera.get_viewpoints()

        # & visualize it
        # self.camera.visualize_trajectory(
        #     viewpoints=viewpoint_stack,
        #     v3d=self.gaussians.out["xyz"],
        #     out_p=out_p / "trajectory.png",
        # )

        self.viewpoint_stack = viewpoint_stack

        writer = imageio.get_writer(
            out_p / f"{self.seq}_orbit.mp4",
            **utils.video.get_writer_cfg(fps=10, height=896),
        )

        for viewpoint_cam in tqdm(
            self.viewpoint_stack, desc="Reconstruct Radiance Field"
        ):
            render_pkg = render(
                gaussians=self.gaussians.out,
                viewpoint_cam=viewpoint_cam,
                bg_color=self.bg_color,
            )
            rgb = render_pkg["render"]
            normal = render_pkg["normals_render"]
            depth = render_pkg["depth"]
            img_grid = torchvision.utils.make_grid(
                [rgb, normal],
                normalize=True,
                nrow=2,
                padding=0,
            )
            frame_hwc = (img_grid.permute(1, 2, 0) * 255).clamp(0, 255).byte()
            writer.append_data(frame_hwc.cpu().numpy())

            # writer.append_data(
            #     (rgb.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            # )
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())

        writer.close()

    @torch.no_grad()
    def extract_mesh_bounded(self, mesh_res: int = 256):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        # voxel_size=0.004, sdf_trunc=0.02

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales

        return o3d.mesh
        """

        voxel_size = self.depth_trunc / mesh_res
        sdf_trunc = 5.0 * voxel_size

        print("Running tsdf volume integration ...")
        print(f"voxel_size: {voxel_size}")
        print(f"sdf_trunc: {sdf_trunc}")
        print(f"depth_truc: {self.depth_trunc}")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i, _cam_o3d in tqdm(
            enumerate(self.viewpoint_stack),
            desc="TSDF integration progress",
        ):
            cam_o3d = o3d.camera.PinholeCameraParameters()
            intrins = _cam_o3d.cam_intrinsics
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=_cam_o3d.image_width,
                height=_cam_o3d.image_height,
                cx=intrins[0, 2].item(),
                cy=intrins[1, 2].item(),
                fx=intrins[0, 0].item(),
                fy=intrins[1, 1].item(),
            )
            cam_o3d.extrinsic = _cam_o3d.world_view_transform.cpu().numpy()
            cam_o3d.intrinsic = intrinsic
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    np.asarray(
                        np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255,
                        order="C",
                        dtype=np.uint8,
                    )
                ),
                o3d.geometry.Image(
                    np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")
                ),
                depth_trunc=self.depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0,
            )

            volume.integrate(
                rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic
            )

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets.
        return o3d.mesh
        """

        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2 - mag) * (y / mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
            compute per frame sdf
            """
            new_points = (
                torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
                @ viewpoint_cam.full_proj_transform
            )
            z = new_points[..., -1:]
            pix_coords = new_points[..., :2] / new_points[..., -1:]
            mask_proj = ((pix_coords > -1.0) & (pix_coords < 1.0) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(
                depthmap.cuda()[None],
                pix_coords[None, None],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).reshape(-1, 1)
            sampled_rgb = (
                torch.nn.functional.grid_sample(
                    rgbmap.cuda()[None],
                    pix_coords[None, None],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .reshape(3, -1)
                .T
            )
            sdf = sampled_depth - z
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(
            samples, inv_contraction, voxel_size, return_rgb=False
        ):
            """
            Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1 / (
                    2 - torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9)
                )
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:, 0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:, 0])
            for i, _ in tqdm(
                enumerate(self.viewpoint_stack), desc="TSDF integration progress"
            ):
                sdf, rgb, mask_proj = compute_sdf_perframe(
                    i,
                    samples,
                    depthmap=self.depthmaps[i],
                    rgbmap=self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb[mask_proj]) / wp[
                    :, None
                ]
                # update weight
                weights[mask_proj] = wp

            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = self.radius * 2 / N
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)

        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R + 0.01, 1.9)

        mesh = utils.mcube.marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )

        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(
            torch.tensor(np.asarray(mesh.vertices)).float().cuda(),
            inv_contraction=None,
            voxel_size=voxel_size,
            return_rgb=True,
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    def post_process_mesh(self, mesh, cluster_to_keep=1000):
        """
        Post-process a mesh to filter out floaters and disconnected parts
        """
        import copy

        print(
            "post processing the mesh to have {} clusterscluster_to_kep".format(
                cluster_to_keep
            )
        )
        mesh_0 = copy.deepcopy(mesh)
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (
                mesh_0.cluster_connected_triangles()
            )

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
        n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        mesh_0.remove_unreferenced_vertices()
        mesh_0.remove_degenerate_triangles()
        print("num vertices raw {}".format(len(mesh.vertices)))
        print("num vertices post {}".format(len(mesh_0.vertices)))
        return mesh_0
