"""
Abstract Class for Gaussian Models.
Heavily inspired (and licensed) by: 


Code based on 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py
License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
"""

import torch
from torch import nn
from pathlib import Path
from loguru import logger
from src.utils.general import (
    inverse_sigmoid,
    strip_symmetric,
    build_scaling_rotation,
    build_rotation,
)
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw


class GaussianModel(torch.nn.Module):

    def __init__(self, sh_degree: int):

        super(GaussianModel, self).__init__()

        self.v: bool = False  # verbose flag

        # vanilla 3DGS
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0.01
        self.spatial_lr_scale = 0
        self.setup_functions()

        # GaussianAvatars: binding stuff...
        self.face_center = None
        self.face_scaling = None
        self.face_orien_mat = None
        self.face_orien_quat = None
        self.binding = None  # gaussian index to face index
        self.binding_counter = None  # number of points bound to each face
        self.timestep = None  # the current time-step
        self.num_timesteps = 1  # required by viewers

        self.out: dict = None

    def __repr__(self):
        repr_str = f"\nxyz:\t\t\t{self._xyz.shape}\n"
        repr_str += f"features_dc:\t\t{self._features_dc.shape}\n"
        repr_str += f"features_rest:\t\t{self._features_rest.shape}\n"
        repr_str += f"scaling:\t\t{self._scaling.shape} \n"
        repr_str += f"rotation:\t\t{self._rotation.shape} \n"
        repr_str += f"opacity:\t\t{self._opacity.shape} \n"
        repr_str += f"max_radii2D:\t\t{self.max_radii2D.shape} \n"
        repr_str += f"xyz_gradient_accum:\t{self.xyz_gradient_accum.shape} \n"
        repr_str += f"denom:\t\t\t{self.denom.shape} \n"
        return repr_str

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = (
            GaussianModel.build_covariance_from_scaling_rotation
        )
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_xyz(self):
        if self.binding is None or self.face_center is None:
            return self._xyz
        else:
            xyz = torch.bmm(
                self.face_orien_mat[self.binding], self._xyz[..., None]
            ).squeeze(-1)
            return (
                xyz * self.face_scaling[self.binding] + self.face_center[self.binding]
            )

    @property
    def get_scaling(self):
        if self.binding is None or self.face_scaling is None:
            return self.scaling_activation(self._scaling)
        else:
            scaling = self.scaling_activation(self._scaling)
            return scaling * self.face_scaling[self.binding]

    @property
    def get_rotation(self):
        if self.binding is None or self.face_orien_quat is None:
            return self.rotation_activation(self._rotation)
        else:
            # always need to normalize the rotation quaternions before chaining them
            rot = self.rotation_activation(self._rotation)
            face_orien_quat = self.rotation_activation(
                self.face_orien_quat[self.binding]
            )
            return quat_xyzw_to_wxyz(
                quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot))
            )  # roma
            # return quaternion_multiply(face_orien_quat, rot)  # pytorch3d



    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def create_from_pcd(self, pcd, extent):
        pass

    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def defrost_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def defrost(self, param_name, v: bool = False):
        for _param_name, param in self.named_parameters():
            if _param_name == param_name:
                param.requires_grad = True
                if v:
                    logger.info(
                        f"Defrosting {param_name} --> requires_grad={param.requires_grad}"
                    )
                break

    def freeze(self, param_name, v: bool = False):
        for _param_name, param in self.named_parameters():
            if _param_name == param_name:
                param.requires_grad = False
                if v:
                    logger.info(
                        f"Freezing {param_name} --> requires_grad={param.requires_grad}"
                    )
                break

    def print_requires_grad(self):
        logger.warning(f"{self.name} requires_grad status:")
        for param_name, param in self.named_parameters():
            logger.info(f"--- {param_name}: {param.requires_grad}")

    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            if self.v:
                logger.info(
                    f"Going from SH degree {self.active_sh_degree} to {self.active_sh_degree + 1}"
                )
            self.active_sh_degree += 1

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.non_densify_params_keys:
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.non_densify_params_keys:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def load_ckpt(self, p):
        """That's basically a wrapper for `restore`"""
        if Path(p).exists():
            self.restore(torch.load(p), cfg=self.cfg.lr)
            if self.v:
                logger.info(f"{self.name} loaded from ckpt ({p}):\n {self}")
        else:
            if self.v:
                logger.warning(f"Could not find {p} - training from scratch!")

    def prune_points(self, mask):
        if self.binding is not None:
            # make sure each face is bound to at least one point after pruning
            binding_to_prune = self.binding[mask]
            counter_prune = torch.zeros_like(self.binding_counter)
            counter_prune.scatter_add_(
                0,
                binding_to_prune,
                torch.ones_like(binding_to_prune, dtype=torch.int32, device="cuda"),
            )
            mask_redundant = (self.binding_counter - counter_prune) > 0
            mask[mask.clone()] = mask_redundant[binding_to_prune]

        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # Update common attributes
        self._xyz = optimizable_tensors["xyz"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[
            valid_points_mask
        ]

        if self.binding is not None:
            self.binding_counter.scatter_add_(
                0,
                self.binding[mask],
                -torch.ones_like(self.binding[mask], dtype=torch.int32, device="cuda"),
            )
            self.binding = self.binding[valid_points_mask]

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size: int,
        max_n_gs: int = None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        _n_gs: int = self.get_xyz.shape[0]
        max_n_gs = max_n_gs if max_n_gs else self.get_xyz.shape[0] + 1
        if self.get_xyz.shape[0] <= max_n_gs:
            grads_abs = self.xyz_gradient_accum_abs / self.denom
            grads_abs[grads_abs.isnan()] = 0.0
            ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
            Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

            self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
            self.densify_and_split(grads, max_grad, grads_abs, Q, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

        self.prune_points(prune_mask)
        self.n_gs = self.get_xyz.shape[0]
        if self.v:
            logger.info(f"{self.name}: {_n_gs} --> {self.n_gs} (max={max_n_gs})")
        torch.cuda.empty_cache()

    def densify_and_split(
        self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, N=2
    ):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[: grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(
            padded_grad_abs >= grad_abs_threshold, True, False
        )
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)

        if self.binding is not None:
            selected_scaling = self.get_scaling[selected_pts_mask]
            face_scaling = self.face_scaling[self.binding[selected_pts_mask]]
            new_scaling = self.scaling_inverse_activation(
                (selected_scaling / face_scaling).repeat(N, 1) / (0.8 * N)
            )
        else:
            new_scaling = self.scaling_inverse_activation(
                self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
            )

        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        if self.binding is not None:
            new_binding = self.binding[selected_pts_mask].repeat(N)
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(
                0,
                new_binding,
                torch.ones_like(new_binding, dtype=torch.int32, device="cuda"),
            )

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(
        self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent
    ):
        # Extract points that satisfy the gradient condition
        grad_cond = torch.norm(grads, dim=-1) >= grad_threshold
        scale_cond = (
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent
        )
        selected_pts_mask = torch.where(grad_cond, True, False)
        selected_pts_mask_abs = torch.where(
            torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False
        )
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask, scale_cond)

        new_xyz = self._xyz[selected_pts_mask]
        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[selected_pts_mask]
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask]
        )

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        if self.binding is not None:
            new_binding = self.binding[selected_pts_mask]
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(
                0,
                new_binding,
                torch.ones_like(new_binding, dtype=torch.int32, device="cuda"),
            )

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        gradient_accum = viewspace_point_tensor.grad.squeeze(0)[update_filter, :2]
        self.xyz_gradient_accum[update_filter] += torch.norm(
            gradient_accum, dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.xyz_gradient_accum_abs_max = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


class GaussianCoordinateFrame:
    """
    Use this object, so have a set of Gaussians representing a coordinate frame
    centered in the world origin. Useful for quick debugging of animation/camera
    pose & rendering methods.


    Usage:
    >>> gaussians = GaussianCoordinateFrame().create()
    >>> render_pkg = render(gaussians, bg_color, viewpoint_cam)
    """

    def __init__(self, points_per_axis: int = 5000):
        self.points_per_axis = points_per_axis
        self.active_sh_degree = 3  # Spherical harmonics degree
        self.total_points = points_per_axis * 3  # Total points for all axes
        self.device = "cuda"

    def generate_axis_points(self, axis_direction, color):
        """Generate points along an axis with some Gaussian spread"""
        points = torch.zeros((self.points_per_axis, 3), device=self.device)
        points[:, axis_direction] = torch.linspace(0, 0.5, self.points_per_axis)

        noise = torch.randn_like(points) * 0.02
        noise[:, axis_direction] = 0  # No noise along the axis direction
        points = points + noise

        shs = torch.zeros((self.points_per_axis, 16, 3), device=self.device)
        shs[:, 0] = torch.tensor(color)

        return points, shs

    def create(self):
        # init gaussians
        x_points, x_shs = self.generate_axis_points(0, [1, 0, 0])  # X > R-ed
        y_points, y_shs = self.generate_axis_points(1, [0, 1, 0])  # Y > G-reen
        z_points, z_shs = self.generate_axis_points(2, [0, 0, 1])  # Z > B-lue

        # set properties
        xyz = torch.cat([x_points, y_points, z_points], dim=0)
        scales = torch.ones_like(xyz) * 0.002  # Uniform scale for all Gaussians
        rotq = torch.zeros((self.total_points, 4), device=self.device)
        rotq[:, 0] = 1.0  # w component = 1, others = 0 for identity rotation
        shs = torch.cat([x_shs, y_shs, z_shs], dim=0)
        opacity = torch.ones((self.total_points, 1), device=self.device)

        # same structure as HumanGS & ObjectGS `out` dicts!
        out = {
            "xyz": xyz,
            "scales": scales,
            "rotq": rotq,
            "shs": shs,
            "opacity": opacity,
            "active_sh_degree": self.active_sh_degree,
            "normals": None,
            "joints": None,
        }

        return out
