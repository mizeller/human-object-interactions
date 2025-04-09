# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Code based on 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

# external
import torch
import numpy as np
from torch import nn
from loguru import logger
from pytorch3d.structures import Meshes
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz

# internal
from src.datasets.arctic_dataset import ArcticGroundTruthData
from smplx import SMPLX
from src.utils import general, graphics, rotations, subdivide_model
from src.model.gaussian_model import GaussianModel
from src.common.body_models import build_subject_smplx
from src import constants


class HumanGS(GaussianModel):

    def __init__(self, cfg, data: ArcticGroundTruthData):
        super().__init__(sh_degree=cfg.human.sh_degree)

        self.type: str = "human"
        self.name: str = "HumanGS"
        self.render_mode: str = "h"
        self.device = constants.device
        self.cfg = cfg.human
        self.v = cfg.train.verbose

        _smplx: SMPLX = build_subject_smplx()
        _smplx = subdivide_model.subdivide_model(
            model=_smplx.to(self.device), n_iter=2, model_id="smplx"
        )

        self.smplx = _smplx.cuda()
        self.smplx_faces = torch.from_numpy(self.smplx.faces).to(self.device)

        # create initialization files w/ init_handgs.py
        # smplx_hands_init_gs_100000.pt -> initialize gaussians on hands only
        # smplx_all_init_gs_584095.pt -> initialize gaussians on full smplx body
        self.binding_counter = torch.load(
            constants.assets_p / "smplx_hands_init_gs_100000.pt"
        )

        # NOTE: Replace zeros with ones, s.t. each face has one gaussian!
        #       (to learn full avatar appearance)
        # binding_counter = torch.where(
        #     self.binding_counter == 0, torch.tensor(1), self.binding_counter
        # )
        # self.binding_counter = binding_counter

        self.binding = torch.repeat_interleave(
            torch.arange(len(self.smplx.faces)),
            self.binding_counter,
        ).to(self.device)
        self.n_gs: int = int(torch.sum(self.binding_counter))

        self.initialize_gs_params()
        if data:
            self.initialize_smplx(data.cached_data)
            self.setup_optimizer(cfg=self.cfg.lr)
            if self.v:
                logger.info(f"Initialized {self.name} object:\n{self}")
            if self.cfg.ckpt:
                self.load_ckpt(self.cfg.ckpt)

    def __repr__(self):
        repr_str = super().__repr__()
        repr_str += f"binding:\t\t{self.binding.shape} \n"
        repr_str += f"binding_counter:\t{self.binding_counter.shape} \n"
        repr_str += f"smplx_body_pose:\t{self.smplx_body_pose.shape} \n"
        repr_str += f"smplx_left_hand_pose:\t{self.smplx_left_hand_pose.shape} \n"
        repr_str += f"smplx_right_hand_pose:\t{self.smplx_right_hand_pose.shape} \n"
        repr_str += f"smplx_global_orient:\t{self.smplx_global_orient.shape} \n"
        repr_str += f"smplx_transl:\t\t{self.smplx_transl.shape} \n"
        repr_str += f"vert_offsets:\t\t{self.vert_offsets.shape} \n"
        return repr_str

    @property
    def get_normals(self):
        face_normals = self.current_mesh.faces_normals_packed()

        # normalize them before computing normals map
        norm = torch.norm(face_normals, dim=1, keepdim=True)
        face_normals = face_normals / norm

        # gaussians bound to a face should have the same normal as the face
        deformed_normals = face_normals[self.binding]

        # should be [-1, 1], 1
        # print(
        #     f"min {deformed_normals.min().item():.3f}, max {deformed_normals.max().item():.3f}, magnitude: {torch.norm(deformed_normals, dim=1).mean().item():.3f}"
        # )

        return deformed_normals

    def initialize_gs_params(self):
        """Initialize Gaussian nn.Params (mean/rot/scale/sh/...)"""
        _init_xyz = torch.zeros((self.n_gs, 3), device=self.device).float()
        _init_scales = torch.log(torch.ones((self.n_gs, 3), device=self.device))
        _init_opacities = general.inverse_sigmoid(
            0.1 * torch.ones((self.n_gs, 1), dtype=torch.float, device=self.device)
        )

        _init_features = torch.zeros((self.n_gs, 3, 16), device=self.device).float()
        _init_rgb = torch.tensor([1.0, 0.0, 0.0]).repeat(self.n_gs, 1)
        _init_features[:, :3, 0] = _init_rgb
        _init_features[:, 3:, 1:] = 0.0

        _init_rots = torch.zeros((self.n_gs, 4), device=self.device)
        _init_rots[:, 0] = 1

        self._features_dc = nn.Parameter(
            _init_features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            _init_features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._xyz = nn.Parameter(_init_xyz.requires_grad_(True))
        self._scaling = nn.Parameter(_init_scales.requires_grad_(True))
        self._rotation = nn.Parameter(_init_rots.requires_grad_(True))
        self._opacity = nn.Parameter(_init_opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.n_gs), device=self.device)

    def initialize_smplx(self, data):
        """Initialize nn.Params for SMPL-X optimization. Data-loader should already have moved all tensors to CUDA."""

        init_smplx_left_hand_pose = torch.stack(
            [x["smplx_left_hand_pose"] for x in data]
        )
        init_smplx_right_hand_pose = torch.stack(
            [x["smplx_right_hand_pose"] for x in data]
        )
        init_smplx_global_orient = torch.stack([x["smplx_global_orient"] for x in data])
        init_smplx_transl = torch.stack([x["smplx_transl"] for x in data])
        init_smplx_body_pose = torch.stack([x["smplx_body_pose"] for x in data])
        init_smplx_betas = data[0]["smplx_betas"]

        self.create_betas(init_smplx_betas, self.cfg.opt.smplx_betas)

        self.create_hand_pose(
            init_hand_pose=init_smplx_right_hand_pose,
            hand_type="r",
            requires_grad=self.cfg.opt.smplx_right_hand_pose,
        )
        self.create_hand_pose(
            init_hand_pose=init_smplx_left_hand_pose,
            hand_type="l",
            requires_grad=self.cfg.opt.smplx_left_hand_pose,
        )

        self.create_body_pose(
            init_body_pose=init_smplx_body_pose,
            requires_grad=self.cfg.opt.smplx_body_pose,
        )

        self.create_global_orient(
            init_global_orient=init_smplx_global_orient,
            requires_grad=self.cfg.opt.smplx_global_orient,
        )

        self.create_transl(
            init_transl=init_smplx_transl, requires_grad=self.cfg.opt.smplx_transl
        )

        # vertex offset parameter
        n_verts = self.smplx.v_template.shape[0]
        if self.cfg.normal_displacement:
            # only displace in normal direction
            init_vert_offsets = torch.Tensor(np.zeros([n_verts, 1])).to(self.device)
        else:
            init_vert_offsets = torch.Tensor(np.zeros([n_verts, 3])).to(self.device)

        self.create_vert_offsets(
            vert_offsets=init_vert_offsets,
            requires_grad=self.cfg.opt.vert_offsets,
        )

    def create_betas(self, betas, requires_grad=False):
        self.smplx_betas = nn.Parameter(betas, requires_grad=requires_grad)
        if self.v:
            logger.info(
                f"Created betas with shape: {betas.shape}, requires_grad: {requires_grad}"
            )

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.binding,
            self.binding_counter,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.smplx_left_hand_pose,
            self.smplx_right_hand_pose,
            self.smplx_body_pose,
            self.smplx_global_orient,
            self.smplx_transl,
            self.smplx_betas,
            self.vert_offsets,
        )

    def restore(self, state_dict, cfg):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.binding,
            self.binding_counter,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.smplx_left_hand_pose,
            self.smplx_right_hand_pose,
            self.smplx_body_pose,
            self.smplx_global_orient,
            self.smplx_transl,
            self.smplx_betas,
            self.vert_offsets,
        ) = state_dict

        self.n_gs = len(self._xyz)
        self.setup_optimizer(cfg)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # these keys correspond to hand-specific parameters which depend on the
        # number of frames in the data used for training/saving the ckpt
        # if they are NOT removed, loading pre-trained ckpts from different video
        # sequences will break the code
        # TODO: fix this problem at the root; i.e. don't save these parameters in the first place
        # logger.error("This works only when running on hold_knoppers (not in demo mode); require full sequence")
        keys_to_remove = []  # [6, 7, 8]
        for key in keys_to_remove:
            if key in opt_dict["state"]:
                del opt_dict["state"][key]

        updated_param_groups = []
        for group in opt_dict["param_groups"]:
            updated_params = [p for p in group["params"] if p not in keys_to_remove]
            if updated_params:  # Only add the group if it has remaining params
                group["params"] = updated_params
                updated_param_groups.append(group)

        opt_dict["param_groups"] = updated_param_groups

        self.optimizer.state.update(opt_dict)
        # self.optimizer.load_state_dict(opt_dict)

    def setup_optimizer(self, cfg):
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device
        )
        self.xyz_gradient_accum_abs = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device
        )
        self.xyz_gradient_accum_abs_max = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        self.spatial_lr_scale = cfg.human_spatial

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
        ]

        if (
            hasattr(self, "smplx_right_hand_pose")
            and self.smplx_right_hand_pose.requires_grad
        ):
            params.append(
                {
                    "params": self.smplx_right_hand_pose,
                    "lr": cfg.smplx_hand_pose,
                    "name": "smplx_right_hand_pose",
                }
            )

        if (
            hasattr(self, "smplx_left_hand_pose")
            and self.smplx_left_hand_pose.requires_grad
        ):
            params.append(
                {
                    "params": self.smplx_left_hand_pose,
                    "lr": cfg.smplx_hand_pose,
                    "name": "smplx_left_hand_pose",
                }
            )

        if hasattr(self, "smplx_body_pose") and self.smplx_body_pose.requires_grad:
            params.append(
                {
                    "params": self.smplx_body_pose,
                    "lr": cfg.smplx_body_pose,
                    "name": "smplx_body_pose",
                }
            )

        if (
            hasattr(self, "smplx_global_orient")
            and self.smplx_global_orient.requires_grad
        ):
            params.append(
                {
                    "params": self.smplx_global_orient,
                    "lr": cfg.smplx_global_orient,
                    "name": "smplx_global_orient",
                }
            )

        if hasattr(self, "smplx_transl") and self.smplx_transl.requires_grad:
            params.append(
                {
                    "params": self.smplx_transl,
                    "lr": cfg.smplx_transl,
                    "name": "smplx_transl",
                }
            )

        if hasattr(self, "vert_offsets") and self.vert_offsets.requires_grad:
            params.append(
                {
                    "params": self.vert_offsets,
                    "lr": cfg.vert_offsets,
                    "name": "vert_offsets",
                }
            )

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

    def create_hand_pose(
        self, init_hand_pose: torch.Tensor, hand_type: str, requires_grad: bool = False
    ) -> None:
        """Initialize nn.Parameter attributes self.smplx_<hand_type>_hand_pose

        Args:
            init_hand_pose (torch.Tensor): SMPL-X hand pose in axis-angle; frames x (num_hand_joints * 3)
            hand_type (str): l/r
            requires_grad (bool, optional): Dis/Enable gradients. Defaults to False.

        Raises:
            ValueError: If hand_type invalid.
        """
        smplx_hand_pose = rotations.axis_angle_to_rotation_6d(
            init_hand_pose.reshape(-1, 3)
        ).reshape(-1, self.smplx.NUM_HAND_JOINTS * 6)

        if hand_type == "l":
            self.smplx_left_hand_pose = nn.Parameter(
                smplx_hand_pose, requires_grad=requires_grad
            )
        elif hand_type == "r":
            self.smplx_right_hand_pose = nn.Parameter(
                smplx_hand_pose, requires_grad=requires_grad
            )
        else:
            raise ValueError("hand_type must be either 'l' or 'r'")

        if self.v:
            logger.info(
                f"Created {hand_type} hand pose with shape: {smplx_hand_pose.shape}, requires_grad: {requires_grad}"
            )

    def create_body_pose(
        self, init_body_pose: torch.Tensor, requires_grad: bool = False
    ) -> None:
        """Initialize nn.Parameter for hand pose.

        Args:
            init_body_pose (torch.Tensor): SMPL-X body pose in axis-angle; frames x 63; frames x (num_hand_joints * 3)
            hand_type (str): left/right
            requires_grad (bool, optional): Dis/Enable gradients. Defaults to False.

        Raises:
            ValueError: If hand_type invalid.
        """
        smplx_body_pose = rotations.axis_angle_to_rotation_6d(
            init_body_pose.reshape(-1, 3)
        ).reshape(-1, self.smplx.NUM_BODY_JOINTS * 6)

        self.smplx_body_pose = nn.Parameter(
            smplx_body_pose, requires_grad=requires_grad
        )

        if self.v:
            logger.info(
                f"Created body pose with shape: {smplx_body_pose.shape}, requires_grad: {requires_grad}"
            )

    def create_global_orient(
        self, init_global_orient: torch.Tensor, requires_grad: bool = False
    ) -> None:
        """Initialize nn.Parameter for global orientation of the body.

        Args:
            init_global_orient (torch.Tensor): SMPL-X global rotation in axis-angle; frames x 3
            requires_grad (bool, optional): Dis/Enable gradients. Defaults to False.
        """
        # NOTE: 6D repr of rotation more suitable for neural networks (https://arxiv.org/abs/1812.07035)

        smplx_global_orient = rotations.axis_angle_to_rotation_6d(
            init_global_orient
        )  # N x 3 -> N x 6
        self.smplx_global_orient = nn.Parameter(
            smplx_global_orient, requires_grad=requires_grad
        )
        if self.v:
            logger.info(
                f"Created global_orient with shape: {self.smplx_global_orient.shape}, requires_grad: {requires_grad}"
            )

    def create_transl(
        self, init_transl: torch.Tensor, requires_grad: bool = False
    ) -> None:
        self.smplx_transl = nn.Parameter(
            init_transl, requires_grad=requires_grad
        )  # N x 3
        if self.v:
            logger.info(
                f"Created transl with shape: {self.smplx_transl.shape}, requires_grad: {requires_grad}"
            )

    def create_vert_offsets(self, vert_offsets, requires_grad=False):
        self.vert_offsets = nn.Parameter(vert_offsets, requires_grad=requires_grad)
        if self.v:
            logger.info(
                f"Created vertex displacement parameter with shape: {vert_offsets.shape}, requires_grad: {requires_grad}"
            )

    def forward(
        self,
        dataset_idx=-1,
        smplx_global_orient=None,
        smplx_transl=None,
        smplx_body_pose=None,
        smplx_left_hand_pose=None,
        smplx_right_hand_pose=None,
        smplx_betas=None,
    ):
        # check arctic/processing.py/forward_gt_world() for more details!

        if hasattr(self, "smplx_global_orient") and smplx_global_orient is None:
            smplx_global_orient = rotations.rotation_6d_to_axis_angle(
                self.smplx_global_orient[dataset_idx].reshape(-1, 6)
            ).reshape(3)

        if hasattr(self, "smplx_body_pose") and smplx_body_pose is None:
            smplx_body_pose = rotations.rotation_6d_to_axis_angle(
                self.smplx_body_pose[dataset_idx].reshape(-1, 6)
            ).reshape(self.smplx.NUM_BODY_JOINTS * 3)

        if hasattr(self, "smplx_left_hand_pose") and smplx_left_hand_pose is None:
            smplx_left_hand_pose = rotations.rotation_6d_to_axis_angle(
                self.smplx_left_hand_pose[dataset_idx].reshape(-1, 6)
            ).reshape(self.smplx.NUM_HAND_JOINTS * 3)

        if hasattr(self, "smplx_right_hand_pose") and smplx_right_hand_pose is None:
            smplx_right_hand_pose = rotations.rotation_6d_to_axis_angle(
                self.smplx_right_hand_pose[dataset_idx].reshape(-1, 6)
            ).reshape(self.smplx.NUM_HAND_JOINTS * 3)

        if hasattr(self, "smplx_transl") and smplx_transl is None:
            smplx_transl = self.smplx_transl[dataset_idx]

        if hasattr(self, "smplx_betas") and smplx_betas is None:
            smplx_betas = self.smplx_betas

        params = {
            # skip FLAME params; use zero as default
            "jaw_pose": torch.zeros(3, device=self.device),
            "leye_pose": torch.zeros(3, device=self.device),
            "reye_pose": torch.zeros(3, device=self.device),
        }
        params["global_orient"] = smplx_global_orient
        params["body_pose"] = smplx_body_pose
        params["left_hand_pose"] = smplx_left_hand_pose
        params["right_hand_pose"] = smplx_right_hand_pose
        params["transl"] = smplx_transl
        params["betas"] = smplx_betas

        # add batch dimension
        params = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in params.items()}
        smplx_output = self.smplx(**params)

        # NOTE: adding vertex displacement to vertices in posed space
        hand_verts = smplx_output.vertices.squeeze(0)
        if hasattr(self, "vert_offsets"):
            hand_verts += self.vert_offsets

        # from pytorch3d.io import IO; IO().save_mesh(Meshes([hand_verts], [self.smplx_faces]), 'tmp/current_mesh.obj')
        self.current_mesh = Meshes([hand_verts], [self.smplx_faces])

        # position
        triangles = hand_verts[self.smplx_faces]
        self.face_center = triangles.mean(dim=1)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = graphics.compute_face_orientation(
            hand_verts,
            self.smplx_faces,
            return_scale=True,
        )
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(
            rotmat_to_unitquat(self.face_orien_mat)
        )  # roma

        # compute certain attributes for evaluation
        j3d = smplx_output.joints.squeeze(0)

        self.out = {
            "xyz": self.get_xyz,
            "scales": self.get_scaling,
            "rotq": self.get_rotation,
            "shs": self.get_features,
            "opacity": self.get_opacity,
            "active_sh_degree": self.active_sh_degree,
            "normals": self.get_normals,  # <-- normals of MANO mesh in deformed space
            "joints": j3d,  # required for eval
        }
