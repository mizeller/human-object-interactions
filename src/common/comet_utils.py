import comet_ml
from tqdm import tqdm
import sys
import torch
import os
import numpy as np
import time
from loguru import logger
import os.path as op
import json
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Union, Tuple

import src.common.sys_utils as sys_utils

try:
    from dotenv import load_dotenv

    load_dotenv()
except:
    print("Couldn't load .env file. Create .env file with COMET_API_KEY=<your_key>.")


# folder used for debugging
DUMMY_EXP = "xxxxxxxxx"


# TODO: specify personal comet details here; don't share...
@dataclass(frozen=True)
class COMET:
    print("Warning: Personalized COMET API key!")
    api_key: str = os.getenv("COMET_API_KEY")
    workspace: str = "iccv25"

    def __repr__(self) -> str:
        repr_str = "*Personal* COMET settings:\n"
        repr_str += f"API key:\t{self.api_key}\n"
        repr_str += f"Workspace:\t{self.workspace}"
        return repr_str


def add_paths(args):
    exp_key = args.exp_key
    exp_name = args.exp_name.split("-")[0]  # comet workspace name

    if exp_key == DUMMY_EXP:
        args.log_dir = f"./logs/{DUMMY_EXP}"
        if os.path.exists(args.log_dir):
            # clear dummy experiment before next run
            import shutil

            shutil.rmtree(args.log_dir)
    else:
        experiment_logs = f"./logs/{exp_name}"
        if not os.path.exists(experiment_logs):
            os.makedirs(experiment_logs)
        args.log_dir = f"{experiment_logs}/{exp_key}"

    args_p = f"{args.log_dir}/args.yaml"
    ckpt_p = f"{args.log_dir}/checkpoints/last.ckpt"
    if not op.exists(ckpt_p) or DUMMY_EXP in ckpt_p:
        ckpt_p = ""
    if args.resume_ckpt != "":
        ckpt_p = args.resume_ckpt
    if args.load_ckpt != "":
        ckpt_p = args.load_ckpt
    args.ckpt_p = ckpt_p

    args.train_dir = f"{args.log_dir}/train"
    os.makedirs(args.train_dir, exist_ok=True)

    if args.infer_ckpt != "":
        basedir = "/".join(args.infer_ckpt.split("/")[:2])
        basename = op.basename(args.infer_ckpt).replace(".ckpt", ".params.pt")
        args.interface_p = op.join(basedir, basename)
    args.args_p = args_p
    if args.cluster:
        args.run_p = op.join(args.log_dir, "condor", "run.sh")
        args.submit_p = op.join(args.log_dir, "condor", "submit.sub")
        args.repo_p = op.join(args.log_dir, "repo")

    return args


def save_args(args, save_keys):
    args_save = {}
    for key, val in args.items():
        if key in save_keys:
            args_save[key] = val
    with open(args.args_p, "w") as f:
        json.dump(args_save, f, indent=4)
    logger.info(f"Saved args at {args.args_p}")


def create_files(args):
    os.makedirs(args.log_dir, exist_ok=True)
    if args.cluster:
        os.makedirs(op.dirname(args.run_p), exist_ok=True)
        sys_utils.copy_repo(op.join("logs", args.exp_key, "repo"))


def log_exp_meta(
    args: OmegaConf, experiment: Union[comet_ml.Experiment, comet_ml.ExistingExperiment]
):
    tags = [args.hostname, args.git_branch]
    logger.info(f"Experiment tags: {tags}")
    experiment.set_name(args.exp_key)
    experiment.add_tags(tags)
    args.python = sys.executable
    experiment.log_parameters(args)


def init_experiment(
    args,
) -> Tuple[Union[comet_ml.Experiment, comet_ml.ExistingExperiment, None], OmegaConf]:
    if args.debug:
        args.exp_key = DUMMY_EXP
    if args.exp_key == "":
        args.exp_key = generate_exp_key(seq=args.seq, exp_name=args.exp_name)

    args = add_paths(args)
    if op.exists(args.args_p) and args.exp_key not in [DUMMY_EXP]:
        with open(args.args_p, "r") as f:
            args_disk = json.load(f)
            args.git_commit = args_disk["git_commit"]
            args.git_branch = args_disk["git_branch"]
            if "comet_key" in args_disk.keys():
                args.comet_key = args_disk["comet_key"]
    else:
        args.git_commit = sys_utils.get_commit_hash()
        args.git_branch = sys_utils.get_branch()

    create_files(args)

    args.hostname = sys_utils.get_host_name()
    project_name = args.project
    disabled = args.mute
    comet_url = args["comet_key"] if "comet_key" in args.keys() else None
    if args.load_ckpt != "":
        comet_url = None

    if not args.cluster and not args.debug:
        comet_meta = COMET()
        if comet_url is None:
            experiment = comet_ml.Experiment(
                api_key=comet_meta.api_key,
                workspace=comet_meta.workspace,
                project_name=project_name,
                disabled=disabled,
                display_summary_level=0,
            )
            args.comet_key = experiment.get_key()
        else:
            experiment = comet_ml.ExistingExperiment(
                previous_experiment=comet_url,
                api_key=comet_meta.api_key,
                project_name=project_name,
                workspace=comet_meta.workspace,
                disabled=disabled,
                display_summary_level=0,
            )

        # BUG: https://github.com/pytorch/pytorch/issues/128819
        # we're using this version: https://github.com/pytorch/pytorch/issues/128819#issuecomment-2214553981
        # this is just a quick fix.
        # By environment gating with "CUDA_VISIBLE_DEVICES" we only make GPU X visible to torch v 2.0.0
        # GPU X then has gpu_index = 0 in this case....
        # I verified that this still only uses the GPU specified by "CUDA_VISIBLE_DEVICES", so we're good.
        # should probably update to later torch/study the issue more in-depth...

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_index = 0  # FIXME: hardcoded the gpu index here!!!!
        logger.error(f"Using HARD-CODED GPU {gpu_index}")
        device = f"cuda:{gpu_index}"

        # if "CUDA_VISIBLE_DEVICES" in os.environ:
        #     gpu_index = int(os.environ["CUDA_VISIBLE_DEVICES"])
        #     logger.info(f"Using GPU {gpu_index}")
        #     device = f"cuda:{gpu_index}"

        logger.add(
            os.path.join(args.log_dir, "train.log"),
            level="INFO",
            colorize=True,
        )
        logger.info(torch.cuda.get_device_properties(device))
        args.gpu = torch.cuda.get_device_properties(device).name

        log_exp_meta(args, experiment)
    else:
        experiment = None

    # NOTE: args is now of type OmegaConf, which only supports
    # primitive types --> don't add Experiments object to args.
    # args.experiment = experiment

    return args, experiment


def log_dict(
    experiment: Union[comet_ml.Experiment, comet_ml.ExistingExperiment],
    metric_dict,
    step: int = None,
    epoch: int = None,
    postfix=None,
):
    if experiment is None:
        return
    for key, value in metric_dict.items():
        if postfix is not None:
            key = key + postfix
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, torch.Tensor) and len(value.view(-1)) == 1:
            value = value.item()

        if isinstance(value, (int, float, np.float32)):
            experiment.log_metric(key, value, step=step, epoch=epoch)


def generate_exp_key(seq: str = "", exp_name: str = "") -> str:
    key = f"{time.strftime('%Y%m%d_%H%M%S')}_{seq}"
    # hyper-params
    hp = "-".join(exp_name.split("-")[1:])
    if len(hp) > 0:
        key = f"{key}_{hp}"
    return key


def fetch_key_from_experiment(experiment):
    if experiment is not None:
        key = str(experiment.get_key())
        # key = key[:9]  # NOTE: incompatible w/ new date_time key-format
        experiment.set_name(key)
    else:
        import random

        hash = random.getrandbits(128)
        key = "%032x" % (hash)
        key = key[:9]
    return key


def push_images(experiment, all_im_list, global_step=None, no_tqdm=False, verbose=True):
    if verbose:
        print("Pushing PIL images")
        tic = time.time()
    iterator = all_im_list if no_tqdm else tqdm(all_im_list)
    for im in iterator:
        im_np = np.array(im["im"])
        if "fig_name" in im.keys():
            experiment.log_image(im_np, im["fig_name"], step=global_step)
        else:
            experiment.log_image(im_np, "unnamed", step=global_step)
    if verbose:
        toc = time.time()
        print("Done pushing PIL images (%.1fs)" % (toc - tic))
