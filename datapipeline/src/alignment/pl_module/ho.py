import cv2
import torch
import imageio
import numpy as np
import torch.nn as nn
from pathlib import Path
from loguru import logger
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import sys

sys.path = [".."] + sys.path
from alignment.loss_terms import gmof
from alignment.params.human import SMPLXParameters
from alignment.params.object import ObjectParameters
from utils import constants, video

mse_loss = nn.MSELoss(reduction="none")
l1_loss = nn.L1Loss(reduction="none")


class HOModule(pl.LightningModule):
    def __init__(self, data: dict, conf: dict, out_p: Path):
        super().__init__()

        self.out_p: Path = out_p
        self.cfg = conf
        self.models = nn.ModuleDict()

        # logging
        self.vis_frames = []
        self.loss_history = {
            "total_loss": [],
            "contact_loss": [],
            "proj_loss": [],
            "translation_loss": [],
            "loss_z": [],
        }
        self.steps = []

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
        self.converged = self.entities["object"]["converged"]  # [N]

        # only use left/right hand joints for alignment optimization;
        # not all joints!

        j3d_h = self.entities["human"]["j3d"]
        j2d_h = self.entities["human"]["j2d.gt"]

        rh_3d = j3d_h[:, constants.rh_idx, :]
        lh_3d = j3d_h[:, constants.lh_idx, :]
        rh_2d = j2d_h[:, constants.rh_idx, :]
        lh_2d = j2d_h[:, constants.lh_idx, :]

        j3d_hands = torch.cat([rh_3d, lh_3d], dim=1).detach()

        j2d_hands = torch.cat([rh_2d, lh_2d], dim=1).detach()

        self.targets = {
            # 2D human gt
            "human.j2d.gt": j2d_hands,  # self.entities["human"]["j2d.gt"],
            # 3D human gt (joints)
            "human.j3d": j3d_hands,  #  self.entities["human"]["j3d"],
            "human.v3d": self.entities["human"]["v3d"],
            # 2D object gt
            "object.j2d.gt": self.entities["object"]["j2d.gt"],
        }

        with torch.no_grad():
            self.idx = 10
            frame = self.frames[self.idx].cpu().numpy().copy()
            for pt_gt in (
                self.targets["object.j2d.gt"][self.idx].cpu().numpy().astype(np.int32)
            ):
                cv2.circle(frame, tuple(pt_gt), 8, (0, 255, 0), -1)
            for pt_gt in (
                self.targets["human.j2d.gt"][self.idx].cpu().numpy().astype(np.int32)
            ):
                cv2.circle(frame, tuple(pt_gt), 8, (0, 0, 255), -1)

            obj_points = self.targets["object.j2d.gt"][self.idx].cpu().numpy()
            mean_point = np.mean(obj_points, axis=0).astype(np.int32)
            cv2.drawMarker(
                frame,
                tuple(mean_point),
                (255, 0, 255),
                cv2.MARKER_STAR,
                markerSize=12,
                thickness=2,
            )

            self.first_frame_w_gt = frame

        for key in self.entities.keys():
            if key == "object":  # works
                self.models[key] = ObjectParameters(entities[key], camera=camera)
            elif key == "human":  # works
                self.models[key] = SMPLXParameters(
                    data=entities[key],
                    camera=camera,
                    batch_size=self.converged.sum().item(),
                )
            else:
                logger.error(f"Invalid entity: {key}")

        # # camera stuff
        # self.camera_extrinsics = data["camera_extrinsics"]
        # self.camera_intrinsics = data["camera_intrinsics"]

    def loss_fn_o(self, preds, mask):
        v3d_o_pred = preds["object.j3d"][mask]
        v3d_h_gt = self.targets["human.j3d"][mask]
        v2d_o_pred = preds["object.j2d"][mask]
        v2d_o_gt = self.targets["object.j2d.gt"][mask]

        # ensure, that the 3D mean of the hand vertices and the object point-cloud are "in contact"
        centroid_h = v3d_h_gt.mean(dim=1)
        centroid_o = v3d_o_pred.mean(dim=1)
        contact_loss = l1_loss(centroid_h, centroid_o).mean() * self.cfg.contact

        translation_loss = (
            l1_loss(v2d_o_pred.mean(dim=1), v2d_o_gt.mean(dim=1)).mean() * self.cfg.o2d
        )

        projection_loss = (
            gmof(v2d_o_pred - v2d_o_gt, sigma=self.cfg.o2d_sigma).sum(dim=-1).mean()
            * self.cfg.o2d
        )

        # ensure object point-cloud is in front of the camera
        # (which is looking in the negative z-direction)
        z_max = torch.clamp(v3d_o_pred[:, :, 2].mean(dim=1), min=0.0)
        z_loss = torch.zeros(1, device=self.device)
        if z_max.sum() > 0:
            z_loss = z_max.sum() / torch.nonzero(z_max).shape[0]
            z_loss = z_loss * self.cfg.z_min

        total_loss = contact_loss + projection_loss + z_loss + translation_loss

        with torch.no_grad():

            self.log(name="contact", value=contact_loss, prog_bar=True)
            self.log(name="projection", value=projection_loss, prog_bar=True)
            self.log(name="translation", value=translation_loss, prog_bar=True)
            self.log(name="z", value=z_loss, prog_bar=True)
            self.log(name="total", value=total_loss, prog_bar=True)

            self.steps.append(self.global_step)
            self.loss_history["total_loss"].append(total_loss.item())
            self.loss_history["proj_loss"].append(projection_loss.item())
            self.loss_history["contact_loss"].append(projection_loss.item())
            self.loss_history["loss_z"].append(z_loss.item())
            self.loss_history["translation_loss"].append(translation_loss.item())

        return total_loss

    def on_train_end(self):

        with torch.no_grad():
            transl_p = self.out_p / "translations.png"
            loss_p = self.out_p / "loss_curves.png"
            video_p = self.out_p / "vis_ho_alignment.mp4"

            plt.figure(figsize=(15, 10))
            num_losses = len(self.loss_history)
            rows = (num_losses + 1) // 2
            cols = 2
            for idx, (loss_name, values) in enumerate(self.loss_history.items(), 1):
                plt.subplot(rows, cols, idx)
                plt.plot(self.steps, values, "b-")
                plt.xlabel("Iterations")
                plt.ylabel(f"{loss_name}")
                plt.title(f"{loss_name} vs Iterations")
                plt.grid(True)

            plt.tight_layout()
            plt.savefig(loss_p)
            plt.close()

            with imageio.get_writer(
                video_p, **video.get_writer_cfg(height=896)
            ) as writer:
                for frame in self.vis_frames:
                    writer.append_data(frame.astype(np.uint8))

            logger.info(f"HO Alignment:\t{video_p}")
            logger.info(f"Loss Curves:\t{loss_p}")
            logger.info(f"Translations:\t{transl_p}")

        return

    def training_step(self, batch, batch_idx):
        self.condition_training()  # un-/freeze grads depending on mode/iteration
        preds = {}

        for key in self.entities.keys():
            # forward pass
            entity_preds = self.models[key](mask=self.converged)

            # prepend 'human'/'object' to all keys in out dict from forward pass
            preds.update({key + "." + k: v for k, v in entity_preds.items()})

        try:
            self.log(name="min", value=int(preds["object.j2d"][0].min()), prog_bar=True)
            self.log(name="max", value=int(preds["object.j2d"][0].max()), prog_bar=True)
        except:
            logger.error(f"Object vertices out of bounds!")
            logger.warning(f"Has NaN: {torch.isnan(preds['object.j2d'][0]).any()}")
            logger.warning(f"Has inf: {torch.isinf(preds['object.j2d'][0]).any()}")
            self.on_train_end()
            raise ValueError(
                "Training failed: Object vertices out of bounds with NaN/inf values"
            )
        loss = self.loss_fn_o(preds, mask=self.converged)

        # add the first 100 frames to the visualization
        # & every 100th frame after that
        if self.global_step <= 100 or self.global_step % 100 == 0:
            with torch.no_grad():
                frame = self.first_frame_w_gt.copy()
                j2d_o_pred = (
                    preds["object.j2d"][self.idx].cpu().numpy().astype(np.int32)
                )
                for pt_gt in j2d_o_pred:
                    cv2.circle(frame, tuple(pt_gt), 5, (255, 0, 0), -1)

                text = f"Iteration: {self.global_step}"
                cv2.putText(
                    frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3
                )

                self.vis_frames.append(frame)
                # cv2.imwrite('tmp/test.png', frame[...,::-1])

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)
        self.optimizer = optimizer
        return optimizer

    def condition_training(self):
        step = self.global_step
        if step == 0:
            self.stage: str = "translation"
            for model in self.models.values():
                # toggle parameters -> requires_grad = False
                for param in model.parameters():
                    param.requires_grad = False

            self.models["object"].obj_transl.requires_grad = True
            self.models["object"].obj_scale.requires_grad = True

        if step == 1:
            print("Object - Stage 1: Scale Down Object (x 0.001)")
            self.models["object"].obj_scale.data[:] = self.cfg["obj_scale"]

            print(
                "Object - Stage 1: Set initial translation to mean hand position in COLMAP space"
            )
            with torch.no_grad():
                hand_joints = self.targets["human.j3d"][self.converged]
                mean_hand_positions = hand_joints.mean(dim=1)  # [N_converged, 3]
                self.models["object"].obj_transl.data[
                    self.converged
                ] = mean_hand_positions

        if step % self.cfg["decay_every"] == 0:
            # decay lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.cfg["decay_factor"]
