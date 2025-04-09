"""
DEPRECATION WARNING: Un-tested file.

This script is intended to be used to extract mesh from a 3DGS model using the mesh-extraction approach from 2DGS.
"""

import sys
import argparse
import open3d as o3d
from typing import List, Dict
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

sys.path.append(str(Path.cwd()))
from src.utils.mesh import GaussianExtractor
from train import GaussianTrainer


def main(args) -> None:
    # modify log_ps if running script directly w/o cli args
    log_ps: List[Path] = args.log_p

    for log_p in log_ps:
        logger.info(f"Extracting mesh from final checkpoint in {log_p.name}")

        cfg = OmegaConf.load(log_p / "args.yaml")

        ckpt_p = log_p / "ckpt"
        ckpt_id = "final"

        ckpts: Dict[str, Path] = {
            "h": ckpt_p / f"human_{cfg['seq']}_{ckpt_id}.pth",
            "o": ckpt_p / f"object_{cfg['seq']}_{ckpt_id}.pth",
        }

        cfg.object.ckpt = ckpts.get("o", None)
        cfg.train.anim_frames = 1  # expedite data-loading
        trainer = GaussianTrainer(cfg=cfg, exp=None)
        extractor = GaussianExtractor(
            seq_id=cfg.seq,
            gaussians=trainer.object_gs,
            cam=trainer.train_dataset.cam,
            bg_color=trainer.bg_color,
        )
        extractor.reconstruction(out_p=log_p)
        mesh_pre = extractor.extract_mesh_bounded()
        mesh_post = extractor.post_process_mesh(mesh_pre, cluster_to_keep=2)
        o3d.io.write_triangle_mesh(str(log_p / f"{cfg.seq}_mesh.ply"), mesh_post)
        logger.info(f"Mesh saved @ {log_p / f'{cfg.seq}_mesh.ply'}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_p",
        type=Path,
        nargs="+",
        required=False,
        default=[Path("logs/xxxxxxxxx")],  # NOTE: list!
    )
    args = parser.parse_args()
    main(args)
