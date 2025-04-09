"""
DEPRECATION WARNING: Since we were using the HANDS Evaluation Server, we had to convert our checkpoints to .pt files.


This file converts the checkpoints to .pt files for the HANDS Evaluation Server.
TODO: implement proper evaluation protocol (that does not rely on the HANDS server...)
"""

import yaml
import json
import torch
import argparse
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

# local
from src.utils.io import pred
from src import common, utils


def create_pred_zip(log_p: Path, ckpt: str = None, verbose: bool = False):
    """
    https://github.com/zc-alexfan/hold/blob/master/docs/arctic.md#evaluation-on-arctic
    """
    logger.info(f"Evaluating {log_p}")
    ckpt = ckpt if ckpt is not None else "final"

    # set paths
    out_p: Path = log_p / f"eval_{ckpt}"
    out_p.mkdir(exist_ok=True, parents=True)

    json_p = out_p / "eval.metric.json"
    npy_p = out_p / "eval.metric_all.npy"

    if json_p.exists():
        logger.warning(f"{log_p} has already been evaluated - skip.")
        # don't re-evaluate!
        with open(str(json_p), "r") as file:
            return json.load(file)

    exp_cfg_p: Path = log_p / "args.yaml"
    cfg = OmegaConf.load(exp_cfg_p)

    seq_p = Path.cwd() / "data" / cfg.seq
    video_p = seq_p / "video.mp4"
    corres_p = seq_p / "corres.txt"

    data_p = seq_p / "data.pt"  # need to extract camera params here...
    camera_dict = torch.load(data_p, map_location="cpu")["camera"]
    _fnames = corres_p.read_text().splitlines()

    # get frames that were used for this training & evaluate only those!
    s = 1 if cfg.sample is None else cfg.sample
    NUM_IMAGES, _, _, _ = utils.video.probe(video_p)
    selected_indices = list(range(0, NUM_IMAGES, s))

    # FIX: remove the _crop from the filenames!
    full_seq_name = f"{cfg.seq}_grab_01_1"
    full_seq_name = full_seq_name.replace("_crop", "")
    # _fnames = [
    #     f"./data/{full_seq_name}/build/image/{i:04d}.png"
    #     for i in range(len(selected_indices))
    # ]
    # fnames = [fname.replace("_crop", "") for fname in _fnames]
    out = {
        # "fnames": fnames,  # List[str], (NUM_IMAGES,)
        "full_seq_name": full_seq_name,  # str, arctic_s03_box_grab_01_1
    }

    # ---- parse the ckpts/gt files
    pred_data = pred.load_data(
        log_p / "ckpt", cfg=cfg, ckpt_id=ckpt, camera_dict=camera_dict
    )

    out.update(pred_data)

    out = common.xdict.xdict(out)
    # at this point, `out` contains more than the data we need in the .zip file.
    challenge_p = log_p.parent / "arctic_preds" / f"{full_seq_name}.pt"
    challenge_p.parent.mkdir(exist_ok=True, parents=True)
    out = out.rm(
        keyword="",
        keep_list=[
            "full_seq_name",
            "j3d_ra.left",
            "j3d_ra.right",
            "v3d_left.object",
            "v3d_right.object",
            "v3d_ra.object",
            "faces",
        ],
        verbose=True,
    )

    torch.save(out, challenge_p)

    logger.success(
        f"Converted {log_p.name} checkpoints to .pt file for HANDS Evaluation Server"
    )

    print(f"type: {type(out)}")
    utils.debug.log(out)

    return


def evaluate_logs(log_p: Path, ckpt: str = None, v: bool = True):
    ckpt = ckpt if ckpt is not None else "final"
    out_p: Path = log_p / f"eval_{ckpt}"
    out_p.mkdir(exist_ok=True, parents=True)
    # Define the file paths
    json_p = out_p / "eval.metric.json"
    npy_p = out_p / "eval.metric_all.npy"

    if json_p.exists():
        logger.warning(f"{log_p} has already been evaluated - skip.")
        # don't re-evaluate!
        with open(str(json_p), "r") as file:
            return json.load(file)

    if not isinstance(log_p, Path):
        if v:
            logger.warning(f"Converting {log_p} to Path object.")
        log_p = Path(log_p)

    exp_cfg_p: Path = log_p / "args.yaml"

    with open(exp_cfg_p, "r") as f:
        cfg = common.xdict.xdict(yaml.safe_load(f))
    logger.info(f"Evaluating {log_p}")

    # ---- parse the ckpts/gt files
    pred_data: common.xdict.xdict = pred.load_data(
        log_p / "ckpt", cfg=common.xdict.xdict(cfg), ckpt_id=ckpt
    )

    raise NotImplementedError("Implement evaluation protocol.")


def main(args) -> None:
    if args.log_p is not None:
        assert args.log_p.exists(), logger.error(f"{args.log_p} does not exist!")
        create_pred_zip(log_p=Path(args.log_p), ckpt=None, verbose=args.verbose)
    else:
        raise NotImplementedError("Evaluation protocol.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        required=False,
        default="baseline",
        help="Eval all log sub-dirs in exp. dir",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        default=None,
        help="Specify checkpoint to evaluate - otherwise use final ckpt.",
    )

    #
    parser.add_argument(
        "--log_p",
        type=Path,
        required=False,
        default=None,
        help="Eval specific log directory.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    main(args)
