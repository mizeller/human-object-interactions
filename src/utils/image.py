# Code based on 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/image_utils.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

import os
import torch
import pathlib
import torchvision.transforms as T
from torch.autograd import Variable
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, List, BinaryIO
from loguru import logger
import cv2


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    img1 = img1.unsqueeze(0).float()  # Add a batch dimension
    img2 = img2.unsqueeze(0).float()  # Add a batch dimension
    return _ssim(img1, img2, window, window_size, channel, size_average, mask)


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


@torch.no_grad()
def normalize_depth(depth, min=None, max=None):
    if depth.shape[0] == 1:
        depth = depth[0]

    if min is None:
        min = depth.min()

    if max is None:
        max = depth.max()

    depth = (depth - min) / (max - min)
    depth = 1.0 - depth
    return depth


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    text_labels: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    txt_font = ImageFont.load_default()
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(im)
    draw.text((10, 10), text_labels, fill=(0, 0, 0), font=txt_font)
    im.save(fp, format=format)


def save_rgba_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = "PNG",
    text_labels: Optional[List[str]] = None,
    **kwargs,
) -> None:
    os.makedirs(os.path.dirname(fp), exist_ok=True)

    grid = make_grid(tensor, **kwargs)
    txt_font = ImageFont.load_default()
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    if text_labels is not None:
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), text_labels, fill=(0, 0, 0), font=txt_font)
    im.save(fp, format=format)


def add_caption(img_tensor, text):
    transform = T.ToPILImage()
    img = transform(img_tensor)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Scale font size to be ~10% of the image height
    font_size = int(height * 0.05)

    # Scale padding based on image dimensions
    padding = int(height * 0.02)  # 2% of image height for padding

    # Position the text at the top left of the image with scaled padding
    x = padding
    y = padding

    # Use existing font path
    font_p = pathlib.Path.cwd() / "assets/Arial.ttf"
    assert font_p.exists(), logger.error(
        # download i.e. from here: https://github.com/kavin808/arial.ttf
        f"Cannot add caption to image. Download font first!"
    )

    font = ImageFont.truetype(str(font_p), font_size)
    draw.text((x, y), text, font=font, fill="red")

    # Convert back to tensor
    img_tensor_with_text = T.ToTensor()(img)
    return img_tensor_with_text


def draw_circles(img, points, color=(255, 0, 0), radius: int = 2) -> None:
    "Draw circles onto image. Specify color RGB tuple and radius."
    if len(points) > 0:
        for x, y in points:
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
    return
