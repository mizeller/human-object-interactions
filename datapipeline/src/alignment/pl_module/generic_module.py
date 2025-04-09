import cv2
import torch
import numpy as np
import torch.nn as nn
from loguru import logger
import torch.optim as optim
import pytorch_lightning as pl

# local
import common.torch_utils as torch_utils
from alignment.params.human import SMPLXParameters
from alignment.params.object import ObjectParameters

mse_loss = nn.MSELoss(reduction="none")


class PLModule(pl.LightningModule):
    def __init__(self, data, mode: str, conf, loss_fn_h, loss_fn_o, loss_fn_ho):
        super().__init__()
        self.mode: str = mode  # h, o, ho
        self.conf = conf
        models = nn.ModuleDict()

        # loss fns
        self.loss_fn_h = loss_fn_h
        self.loss_fn_o = loss_fn_o
        self.loss_fn_ho = loss_fn_ho

        camera = {
            "extrinsics": data["camera_extrinsics"],
            "intrinsics": data["camera_intrinsics"],
        }
        self.camera: dict = camera
        self.frames: torch.Tensor = data["frames"]

        # entities; params
        entities = {
            "human": {
                "betas": data["smplx_betas"],
                "transl": data["smplx_transl"],
                "body_pose": data["smplx_body_pose"],
                "left_hand_pose": data["smplx_left_hand_pose"],
                "right_hand_pose": data["smplx_right_hand_pose"],
                "global_orient": data["smplx_global_orient"],
                # use v3d and/or j3d for coarse contact w/ object!
                "v3d": data["smplx_v3d"],
                "j3d": data["smplx_j3d"],
                "j2d.gt": data["smplx_j2d"],  # .gt important - see training_step()
            },
            "object": {
                "poses": data["object_poses"],
                "converged": data["object_converged"],
                "v3d_cano": data["object_v3d_cano"],
                "v3d": data["object_v3d"],
                "j2d.gt": data[
                    "object_v2d"
                ],  # output from camera_motion.analyze_results.project2d
            },
        }

        self.entities = entities

        self.targets = {
            # 2D human gt
            "human.j2d.gt": self.entities["human"]["j2d.gt"],
            # 3D human gt (joints)
            "human.j3d": self.entities["human"]["j3d"],
            "human.v3d": self.entities["human"]["v3d"],
            # 2D object gt
            "object.j2d.gt": self.entities["object"]["j2d.gt"],
        }

        with torch.no_grad():
            frame = self.frames[0].cpu().numpy().copy()
            for pt_gt in (
                self.targets["object.j2d.gt"][0].cpu().numpy().astype(np.int32)
            ):
                cv2.circle(frame, tuple(pt_gt), 3, (0, 255, 0), -1)
            for pt_gt in self.targets["human.j2d.gt"][0].cpu().numpy().astype(np.int32):
                cv2.circle(frame, tuple(pt_gt), 3, (0, 0, 255), -1)
            self.first_frame_w_gt = frame

        for key in self.entities.keys():
            if key == "object":  # works
                models[key] = ObjectParameters(entities[key], camera=camera)
            elif key == "human":  # works
                models[key] = SMPLXParameters(
                    data=entities[key],
                    camera=camera,
                    batch_size=self.entities["object"]["converged"].sum().item(),
                )
            else:
                logger.error(f"Invalid entity: {key}")

        self.models = models

        # # camera stuff
        # self.camera_extrinsics = data["camera_extrinsics"]
        # self.camera_intrinsics = data["camera_intrinsics"]

    def training_step(self, batch, batch_idx):
        self.condition_training()  # un-/freeze grads depending on mode/iteration
        preds = {}

        converged = self.entities["object"]["converged"]  # [N]
        for key in self.entities.keys():
            # forward pass
            entity_preds = self.models[key](mask=converged)

            # prepend 'human'/'object' to all keys in out dict from forward pass
            preds.update({key + "." + k: v for k, v in entity_preds.items()})

        # SANITY CHECK: plot the ground truth v. predicted 2D object vertices
        # if self.global_step % 100 == 0:
        #     with torch.no_grad():
        #         frame = self.first_frame_w_gt.copy()
        #         j2d_o_pred = preds["object.j2d"][0].cpu().numpy().astype(np.int32)
        #         for pt_gt in j2d_o_pred:
        #             cv2.circle(frame, tuple(pt_gt), 2, (255, 0, 0), -1)
        #         cv2.imwrite(
        #             f"./tmp/projection/{self.global_step:05}.png", frame[..., ::-1]
        #         )

        # Apply loss only on converged frames
        loss = 0.0
        if self.mode == "h":
            loss += self.loss_fn_h(preds, self.targets, self.conf, mask=converged)
        elif self.mode == "o":
            loss += self.loss_fn_o(preds, self.targets, self.conf, mask=converged)
        elif self.mode == "ho":
            loss += self.loss_fn_h(preds, self.targets, self.conf, mask=converged)
            loss += self.loss_fn_o(preds, self.targets, self.conf, mask=converged)
            loss += self.loss_fn_ho(preds, self.targets, self.conf, mask=converged)
        else:
            raise NotImplementedError

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.conf["lr"])
        self.optimizer = optimizer
        return optimizer

    def condition_training(self):
        step = self.global_step
        if step == 0:
            for val in self.models.values():
                torch_utils.toggle_parameters(val, requires_grad=False)

        # freeze the other model
        if self.mode == "h":
            # human model schedule
            if step == 0:
                print("Human: stage 0")
                for key, val in self.models.items():
                    print(key)
                    if key == "human":
                        # TODO: which nn.Params of the SMPLXParameters
                        # object should receive gradients?
                        val.left_hand_pose.requires_grad = True
                        val.right_hand_pose.requires_grad = True

            if step == 5000:
                print("Human: stage 2")
                for key, val in self.models.items():
                    if key == "human":
                        val.betas.requires_grad = True
            if step % self.conf["decay_every"] == 0:
                torch_utils.decay_lr(self.optimizer, self.conf["decay_factor"])

        elif self.mode == "o":
            if step == 0:
                print("Object - Stage 0: Optimize for Translation")
                self.models["object"].obj_transl.requires_grad = True

            if step == 1:
                # essentially, scale object pointcloud to 1 point to optimize the translation
                print("Object - Stage 1: Scale Down Object (x 0.001)")
                self.models["object"].obj_scale[:] = self.conf["obj_scale"]

            if step == 2000:
                print(
                    "Object - Stage 2: Optimize for Object Scale & Translation jointly"
                )
                self.models["object"].obj_scale.requires_grad = True

            if step % self.conf["decay_every"] == 0:
                torch_utils.decay_lr(self.optimizer, self.conf["decay_factor"])

        elif self.mode == "ho":
            for key, val in self.models.items():
                if key == "human":
                    val.left_hand_pose.requires_grad = True
                    val.right_hand_pose.requires_grad = True
                else:
                    val.obj_scale.requires_grad = True
                    val.obj_transl.requires_grad = True

            if step % self.conf["decay_every"] == 0:
                print("Decay")
                torch_utils.decay_lr(self.optimizer, self.conf["decay_factor"])
        else:
            raise NotImplementedError
