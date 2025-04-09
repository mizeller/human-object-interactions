import torch
import trimesh
from torch import nn
from pathlib import Path
from typing import Union
from loguru import logger

from src import constants
from src.datasets.hold_dataset import HOLDataset
from src.datasets.arctic_dataset import ArcticGroundTruthData
from src.datasets.ho_dataset import HumanObjectData
from src.model.gaussian_model import GaussianModel
from src.utils import rotations, general, spherical_harmonics
from src.common.object_tensors import ObjectTensors


class ObjectGS(GaussianModel):
    def __init__(
        self,
        cfg,
        data: Union[HOLDataset, ArcticGroundTruthData, HumanObjectData],
        sh_degree: int = 3,
    ):
        super().__init__(sh_degree=sh_degree)

        self.type: str = "object"
        self.name: str = "ObjectGS"
        self.render_mode: str = "o"
        self.device = constants.device
        self.binding = None  # compatibility with HandModel
        self.non_densify_params_keys = ["obj_rot", "obj_trans", "obj_scale"]
        self.cfg = cfg.object
        self.v = cfg.train.verbose

        if data:
            self.create_from_pcd(data.init_pcd)
            self.initialize_poses(data.cached_data)
            self.setup_optimizer(cfg=self.cfg.lr)
            if self.v:
                logger.info(f"Initialized ObjectGS object:\n{self}")

            if self.cfg.ckpt:
                self.load_ckpt(self.cfg.ckpt)

    def initialize_poses(self, data):
        """Initialize nn.Params for object pose optimization."""

        init_obj_pose = torch.stack([x["obj_pose"] for x in data])  # (N, 4, 4)
        init_obj_rot = init_obj_pose[:, :3, :3]  # (N, 3, 3)
        init_obj_trans = init_obj_pose[:, :3, 3]  # (N, 3)
        init_obj_rot = rotations.matrix_to_axis_angle(init_obj_rot)

        init_obj_scale = data[0]["obj_scale"]

        self.create_rotation(
            global_orient=init_obj_rot,
            requires_grad=self.cfg.opt.rotation,
        )

        self.create_translation(
            transl=init_obj_trans,
            requires_grad=self.cfg.opt.translation,
        )
        self.create_scale(
            scale=torch.Tensor([init_obj_scale]).to(self.device),
            requires_grad=self.cfg.opt.scale,
        )

    def capture(self):
        state_dict = {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer": self.optimizer.state_dict(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "obj_rot": self.obj_rot,
            "obj_trans": self.obj_trans,
            "obj_scale": self.obj_scale,
        }
        return state_dict

    def restore(self, state_dict, cfg):
        self.active_sh_degree = state_dict["active_sh_degree"]
        self._xyz = state_dict["xyz"]
        self._features_dc = state_dict["features_dc"]
        self._features_rest = state_dict["features_rest"]
        self._scaling = state_dict["scaling"]
        self._rotation = state_dict["rotation"]
        self._opacity = state_dict["opacity"]
        self.max_radii2D = state_dict["max_radii2D"]
        xyz_gradient_accum = state_dict["xyz_gradient_accum"]
        denom = state_dict["denom"]
        opt_dict = state_dict["optimizer"]
        self.n_gs = len(self._xyz)
        self.spatial_lr_scale = state_dict["spatial_lr_scale"]
        self.obj_rot = state_dict["obj_rot"]
        self.obj_trans = state_dict["obj_trans"]
        self.obj_scale = state_dict["obj_scale"]

        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.setup_optimizer(cfg)
        self.optimizer.load_state_dict(opt_dict)

    def create_rotation(self, global_orient, requires_grad=False):
        """global orient: in axes angle format"""
        # NOTE: 6D repr of rotation more suitable for neural networks (https://arxiv.org/abs/1812.07035)
        global_orient = rotations.axis_angle_to_rotation_6d(global_orient)  # Nx3 -> Nx6

        self.obj_rot = nn.Parameter(global_orient, requires_grad=requires_grad)
        if self.v:
            logger.info(
                f"Created global_orient with shape: {global_orient.shape}, requires_grad: {requires_grad}"
            )

    def create_translation(self, transl, requires_grad=False):
        self.obj_trans = nn.Parameter(transl, requires_grad=requires_grad)  # N x 3
        if self.v:
            logger.info(
                f"Created transl with shape: {transl.shape}, requires_grad: {requires_grad}"
            )

    def create_scale(self, scale, requires_grad=False):
        """
        Add a nn.Parameter to learn the optimal scale of the object.
        In HOLD-Dataset, the canonical object point cloud is derived by
        applying the learned (initial) scale to the sparze COLMAP pointcloud.
        Thus, we initialise the scale parameter here w/ 1.0.
        """
        self.obj_scale = nn.Parameter(scale, requires_grad=requires_grad)

    def create_from_pcd(self, pcd: trimesh.PointCloud, spatial_lr_scale: float = 0.0):
        """Initialize Gaussian nn.Params (mean/rot/scale/sh/...)"""
        self.spatial_lr_scale: float = spatial_lr_scale

        fused_point_cloud = torch.tensor(
            pcd.vertices, device=self.device, dtype=torch.float32
        )
        fused_color = spherical_harmonics.RGB2SH(
            torch.tensor((pcd.visual.vertex_colors[:, :3] / 255), device=self.device)
        )
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0  # Ng, 3, 16

        if self.v:
            logger.info(
                f"Number of scene points at initialisation: {fused_point_cloud.shape[0]}"
            )
        self.n_gs: int = fused_point_cloud.shape[0]
        v3d_cano = fused_point_cloud.clone().detach()
        self.v3d_cano_homo = torch.cat(
            [v3d_cano, torch.ones(v3d_cano.shape[0], 1, device=v3d_cano.device)], dim=1
        )
        xy_scale = torch.log(
            torch.sqrt(0.0001 * torch.ones(self.n_gs).cuda())[..., None]
        ).repeat(1, 2)
        z_scale = torch.log(
            torch.sqrt(0.0000001 * torch.ones(self.n_gs).cuda())[..., None]
        )
        scales = torch.cat([xy_scale, z_scale], axis=1)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")  # Ng x 4
        rots[:, 0] = 1

        # NOTE: setting opacity value to 1 for all gaussians & not computing gradients...
        # opacities = torch.ones(
        #     (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
        # )

        opacities = self.inverse_opacity_activation(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )  # Ng x 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def setup_optimizer(self, cfg):
        self.percent_dense = cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.xyz_gradient_accum_abs_max = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        params = [
            {
                "params": [self._xyz],
                "lr": cfg.position_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {"params": [self._features_dc], "lr": cfg.feature, "name": "f_dc"},
            {
                "params": [self._features_rest],
                "lr": cfg.feature / 20.0,
                "name": "f_rest",
            },
            {"params": [self._opacity], "lr": cfg.opacity, "name": "opacity"},
            {"params": [self._scaling], "lr": cfg.scaling, "name": "scaling"},
            {"params": [self._rotation], "lr": cfg.rotation, "name": "rotation"},
            # add learnable obj. scale and pose
            {"params": [self.obj_rot], "lr": cfg.obj_rot, "name": "obj_rot"},
            {"params": [self.obj_trans], "lr": cfg.obj_trans, "name": "obj_trans"},
            {"params": [self.obj_scale], "lr": cfg.obj_scale, "name": "obj_scale"},
        ]

        if self.v:
            for param in params:
                logger.info(f"Parameter: {param['name']}, lr: {param['lr']}")

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = general.get_expon_lr_func(
            lr_init=cfg.position_init * self.spatial_lr_scale,
            lr_final=cfg.position_final * self.spatial_lr_scale,
            lr_delay_mult=cfg.position_delay_mult,
            max_steps=cfg.position_max_steps,
        )

    def forward(
        self,
        dataset_idx=-1,
        obj_rot=None,
        obj_trans=None,
    ):
        """
        Deform the canonical point cloud to the deformed space.
        Same procedure as in object_model.py/forward().
        """

        if hasattr(self, "obj_rot") and obj_rot is None:
            obj_rot = rotations.rotation_6d_to_matrix(self.obj_rot[dataset_idx])

        if hasattr(self, "obj_trans") and obj_trans is None:
            obj_trans = self.obj_trans[dataset_idx]

        # construct world to camera transformation matrix
        w2c = torch.eye(4, device=self._xyz.device)
        w2c[:3, :3] = obj_rot
        w2c[:3, 3] = obj_trans

        v3d_cano = self._xyz * self.obj_scale

        # canonical2posed transform
        homogen_coord = torch.ones_like(v3d_cano[..., :1])
        gs_xyz_homo = torch.cat([v3d_cano, homogen_coord], dim=-1)
        gs_xyz_deform_homo = torch.matmul(w2c, gs_xyz_homo.T).T
        v3d_deform = gs_xyz_deform_homo[:, :3] / gs_xyz_deform_homo[:, 3:4]

        gs_rotq = self.get_rotation  # [V, 4]
        gs_rotmat = rotations.quaternion_to_matrix(gs_rotq)  # [V, 3, 3]
        deformed_gs_rotmat = torch.matmul(obj_rot[None], gs_rotmat)  # [V, 3, 3]
        deformed_gs_rotq = rotations.matrix_to_quaternion(deformed_gs_rotmat)  # [V, 4]

        self.out = {
            "xyz": v3d_deform,  # ≡ obj_out.v
            "scales": self.get_scaling,
            "rotq": deformed_gs_rotq,
            "shs": self.get_features.clone(),
            "opacity": self.get_opacity,
            "active_sh_degree": self.active_sh_degree,
            # the following k:v pairs are just s.t. the
            # `out` object aligns w/ human_gs.
            "normals": None,
            "joints": None,
        }
