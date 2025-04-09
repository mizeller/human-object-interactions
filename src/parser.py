from comet_ml import ExistingExperiment, Experiment
from omegaconf import OmegaConf, dictconfig
from typing import Tuple, Union, List
import argparse
import sys

from src import utils, common


def parse_args() -> Tuple[dictconfig.DictConfig, List[dictconfig.DictConfig]]:
    """Extract all arguments from the command line and config file. Extract all permutations of arguments from the .yaml file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq",
        type=str,
        help="Specify training sequence.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="./configs/debug_fast.yaml",
        help="Specify config file.",
    )
    _set_legacy_args(parser)
    args, _ = parser.parse_known_args()
    args.cmd = " ".join(sys.argv)

    cfg_file = OmegaConf.load(args.cfg)
    # make sure, the project name is in both configs;
    # align keys in both configs (TODO)
    args.project = cfg_file.exp_name

    if args.seq is not None:
        cfg_file.seq = args.seq
    training_cfgs, _ = utils.config.get_cfg_items(cfg_file)
    default_cfg: dictconfig.DictConfig = OmegaConf.create(vars(args))
    return default_cfg, training_cfgs


def init_experiment(
    default_cfg: OmegaConf, training_cfg: OmegaConf
) -> Tuple[OmegaConf, Union[ExistingExperiment, Experiment]]:
    """Initialize Experiment on Comet_ML."""

    _cfgs: OmegaConf = OmegaConf.merge(default_cfg, training_cfg)

    # init comet_ml experiment + add some more k:v pairs
    cfgs, experiment = common.comet_utils.init_experiment(_cfgs)

    # dump args.yaml
    with open(cfgs.args_p, "w") as f:
        f.write(OmegaConf.to_yaml(cfgs))

    return cfgs, experiment


def _set_legacy_args(parser):
    "add these argument to be compatible w/ Alex' code in `common`"
    parser.add_argument("--mute", help="No logging", action="store_true")
    parser.add_argument("--exp_key", type=str, default="")
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument(
        "--infer_ckpt",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument(
        "--load_ckpt",
        type=str,
        default="",
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
