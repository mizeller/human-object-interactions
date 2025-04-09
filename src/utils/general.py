# Code based on 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/general_utils.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


import os
import torch
import random
import subprocess
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

# local
from src import constants
from src.utils import graphics


def debug_tensor(tensor, name):
    print(f"{name}: {tensor.shape} {tensor.dtype} {tensor.device}")
    print(
        f"{name}: min: {tensor.min().item():.5f} \
        max: {tensor.max().item():.5f} \
            mean: {tensor.mean().item():.5f} \
                std: {tensor.std().item():.5f}"
    )


class RandomIndexIterator:
    def __init__(self, max_index):
        self.max_index = max_index
        self.indices = list(range(max_index))
        random.shuffle(self.indices)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.max_index:
            self.current_index = 0
            random.shuffle(self.indices)
        index = self.indices[self.current_index]
        self.current_index += 1
        return index


def find_cfg_diff(default_cfg, cfg, delimiter="_"):
    default_cfg_list = OmegaConf.to_yaml(default_cfg).split("\n")
    cfg_str_list = OmegaConf.to_yaml(cfg).split("\n")
    diff_str = ""
    nlines = len(default_cfg_list)
    for lnum in range(nlines):
        if default_cfg_list[lnum] != cfg_str_list[lnum]:
            diff_str += cfg_str_list[lnum].replace(": ", "-").replace(" ", "")
            diff_str += delimiter
    diff_str = diff_str[:-1]
    return diff_str


from pathlib import Path
import shutil


def create_video(
    imgs_p: Path,
    video_p: str,
    reset_folder: bool = False,
    fps: int = 10,
    v: bool = True,
):
    """Write a bunch of .png images in `imgs_p` to `video_p`. If reset_folder is True, a new, empty folder is re-initialized (video_p should not point to imgs_p)! 
    TODO: extend w/ more file extensions (png, jpg, ...)
    TODO: deprecate this method in favour of imageio.writer
    """

    if isinstance(imgs_p, str):
        imgs_p = Path(imgs_p)
    if isinstance(video_p, str):
        video_p = Path(video_p)

    assert isinstance(
        imgs_p, Path
    ), f"imgs_p should be a Path object, is {type(imgs_p)}"
    assert isinstance(
        video_p, Path
    ), f"video_p should be a Path object, is {type(video_p)}"

    video_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        cmd = f"/usr/bin/ffmpeg -hide_banner -loglevel error -framerate {fps} -pattern_type glob -i '{imgs_p}/*.png' \
            -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" \
                -c:v libx264 -pix_fmt yuv420p {video_p} -y"
        subprocess.call(cmd, shell=True)
        if v:
            logger.info(f"Video is saved under {video_p}")
    except Exception as e:
        logger.error(f"Error while creating video: {e}")

    if reset_folder:
        shutil.rmtree(imgs_p)
        imgs_p.mkdir(parents=True, exist_ok=True)


def save_images(img, img_fname, txt_label=None):
    if not os.path.isdir(os.path.dirname(img_fname)):
        os.makedirs(os.path.dirname(img_fname), exist_ok=True)
    im = Image.fromarray(img)
    if txt_label is not None:
        draw = ImageDraw.Draw(im)
        txt_font = ImageFont.load_default()
        draw.text((10, 10), txt_label, fill=(0, 0, 0), font=txt_font)
    im.save(img_fname)


def eps_denom(denom, eps=1e-17):
    """Prepare denominator for division"""
    denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
    denom = denom_sign * torch.clamp(denom.abs(), eps)
    return denom


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_static_camera(img_size=512, fov=0.4, device="cuda"):
    fovx = fov
    fovy = fov
    zfar = 100.0
    znear = 0.01
    world_view_transform = torch.eye(4)

    cam_int = torch.eye(4)

    # convert fov to focal length (in pixels
    fx = graphics.fov2focal(fovx, img_size)
    fy = graphics.fov2focal(fovy, img_size)

    cam_int[0, 0] = fx
    cam_int[1, 1] = fy
    cam_int[0, 2] = img_size / 2
    cam_int[1, 2] = img_size / 2

    projection_matrix = graphics.getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
    ).transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    datum = {
        "fovx": fovx,
        "fovy": fovy,
        "image_height": img_size,
        "image_width": img_size,
        "world_view_transform": world_view_transform,
        "full_proj_transform": full_proj_transform,
        "camera_center": camera_center,
        "cam_int": cam_int,
        "cam_ext": world_view_transform,
        "near": znear,
        "far": zfar,
    }
    for k, v in datum.items():
        if isinstance(v, torch.Tensor):
            datum[k] = v.to(device)

    return datum





def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.set_device(constants.device)


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def torch_rotation_matrix_from_vectors(vec1: torch.Tensor, vec2: torch.Tensor):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector of shape N,3
    :param vec2: A 3d "destination" vector of shape N,3
    :return mat: A transform matrix (Nx3x3) which when applied to vec1, aligns it with vec2.
    """
    a = vec1 / torch.norm(vec1, dim=-1, keepdim=True)
    b = vec2 / torch.norm(vec2, dim=-1, keepdim=True)

    v = torch.cross(a, b, dim=-1)
    c = torch.matmul(a.unsqueeze(1), b.unsqueeze(-1)).squeeze(-1)
    s = torch.norm(v, dim=-1, keepdim=True)
    kmat = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
    kmat[:, 0, 1] = -v[:, 2]
    kmat[:, 0, 2] = v[:, 1]
    kmat[:, 1, 0] = v[:, 2]
    kmat[:, 1, 2] = -v[:, 0]
    kmat[:, 2, 0] = -v[:, 1]
    kmat[:, 2, 1] = v[:, 0]
    rot_mat = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0)
    rot_mat = (
        rot_mat + kmat + torch.matmul(kmat, kmat) * ((1 - c) / (s**2)).unsqueeze(-1)
    )
    return rot_mat


def clean(path: Path, v: bool = True) -> None:
    """Recursively delete all empty directories in the given path."""
    # Recursively iterate over all directories in the given path
    for dir_path in path.rglob("*"):
        if dir_path.is_dir():
            # If the directory is empty, delete it
            try:
                dir_path.rmdir()  # This will only remove the directory if it's empty
                if v:
                    logger.info(f"Deleted empty directory: {dir_path}")
            except OSError:
                # Directory is not empty, or there was an error, so skip it
                pass


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.clamp(input, -1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        tanh = torch.tanh(input)
        grad_input[input <= -1] = (1.0 - tanh[input <= -1] ** 2.0) * grad_output[
            input <= -1
        ]
        grad_input[input >= 1] = (1.0 - tanh[input >= 1] ** 2.0) * grad_output[
            input >= 1
        ]
        max_norm = 1.0  # set the maximum gradient norm value
        torch.nn.utils.clip_grad_norm_(grad_input, max_norm)
        return grad_input
