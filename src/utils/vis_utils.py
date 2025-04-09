import os
import os.path as op

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def aggregate_reshape(outputs, img_size):
    image = torch.cat(outputs, dim=0)
    image = image.reshape(*img_size, -1)
    return image


def scale_to_image(image):
    if image is None:
        return None
    image = (image * 255).cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image)
    return image


def segm_pred_to_cmap(segm_pred):
    # [r, g, b]
    class2color = torch.FloatTensor(
        np.array(
            [
                [0, 0, 0],  # background, black
                [255, 0, 0],  # object, red
                [100, 100, 100],  # right, grey,
                [0, 0, 255],  # left, blue
            ]
        )
    )
    segm_map = class2color[segm_pred] / 255.0
    return segm_map


def make_normal_transparent(imap_raw, normal_map):
    bg_idx = imap_raw == 0
    normal_map = np.array(normal_map)
    alpha = np.zeros_like(normal_map)[:, :, :1] + 255
    normal_map_alpha = np.concatenate((normal_map, alpha), axis=2)
    normal_map_alpha[bg_idx, 3] = 0
    normal_map_alpha = Image.fromarray(normal_map_alpha)
    return normal_map_alpha


def output2images(outputs, img_size):
    from common.xdict import xdict
    from common.ld_utils import ld2dl

    out = xdict()
    outputs = xdict(ld2dl(outputs))
    # process normals
    for key in outputs.search("normal").keys():
        pred_normal = aggregate_reshape(outputs[key], img_size)
        # transform to colormap in [0, 1]
        pred_normal = (pred_normal + 1) / 2
        # scale to image [0, 255]
        pred_normal = scale_to_image(pred_normal)
        out[key] = pred_normal

    # process mask
    for key in outputs.search("mask_prob").keys():
        pred_mask = aggregate_reshape(outputs[key], img_size).repeat(1, 1, 3)
        pred_mask = (pred_mask > 0.5).float()
        pred_mask = scale_to_image(pred_mask)
        out[key] = pred_mask

    # fg_rgb
    for key in outputs.search("fg_rgb.vis").keys():
        pred_rgb = aggregate_reshape(outputs[key], img_size)
        pred_rgb = scale_to_image(pred_rgb)
        out[key] = pred_rgb

    # composite rendering
    pred_imap = aggregate_reshape(outputs["instance_map"], img_size)
    pred_rgb = aggregate_reshape(outputs["rgb"], img_size)
    gt_rgb = aggregate_reshape(outputs["gt.rgb"], img_size)
    bg_rgb = aggregate_reshape(outputs["bg_rgb_only"], img_size)
    out.overwrite("normal", make_normal_transparent(pred_imap.squeeze(), out["normal"]))

    pred_imap = segm_pred_to_cmap(pred_imap.squeeze())

    pred_rgb = scale_to_image(pred_rgb)
    gt_rgb = scale_to_image(gt_rgb)
    bg_rgb = scale_to_image(bg_rgb)

    # concat PIL images horizontally
    rgb = Image.new("RGB", (gt_rgb.width + pred_rgb.width, gt_rgb.height))
    rgb.paste(gt_rgb, (0, 0))
    rgb.paste(pred_rgb, (gt_rgb.width, 0))

    out["rgb"] = rgb
    out["imap"] = scale_to_image(pred_imap)
    out["bg_rgb"] = bg_rgb
    return out


def create_transparent_image(normal, fg_mask_t):
    """
    Creates a transparent PNG image from the 'normal' RGB image (np.uint8) based on the 'fg_mask_t' (np.float32).
    When fg_mask_t[i, j] is 0, the pixel at the same position in 'normal' becomes transparent.

    :param normal: np.ndarray of shape (H, W, 3), representing an RGB image.
    :param fg_mask_t: np.ndarray of shape (H, W), where 0 values indicate transparent pixels.
    :return: PIL Image with transparency applied.
    """
    # Ensure the mask is in the correct range [0, 1]
    fg_mask_t = np.clip(fg_mask_t, 0, 1)

    # Convert the mask to an alpha channel (255 for visible, 0 for transparent)
    alpha_channel = (fg_mask_t * 255).astype(np.uint8)

    # Add the alpha channel to the original image
    rgba_image = np.dstack((normal, alpha_channel))

    # Convert to PIL Image for easier handling and saving
    return Image.fromarray(rgba_image, "RGBA")


def record_vis(idx, current_step, log_dir, experiment, vis_dict):
    idx = int(idx)
    filenames = [f"{key}/step_{current_step:09}_id_{idx:04}" for key in vis_dict.keys()]
    out_ps = [op.join(log_dir, "visuals", f"{fn}.png") for fn in filenames]
    for out_p, im in zip(out_ps, vis_dict.values()):
        os.makedirs(op.dirname(out_p), exist_ok=True)
        im.save(out_p)

        key = "/".join(out_p.split("/")[-2:])
        experiment.log_image(out_p, key, step=current_step)


def add_text_to_image(np_img, text, color=(0, 0, 0)) -> Image:
    pil_img = Image.fromarray(np_img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default(size=36)
    text_position = (10, np_img.shape[0] - 36 - 10)
    draw.text(text_position, text, fill=color, font=font)
    return pil_img


# copy from nerfstudio and 2DGS
import torch
from matplotlib import cm
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np


def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth,
    accumulation,
    near_plane=2.0,
    far_plane=6.0,
    cmap="turbo",
):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image


def save_points(path_save, pts, colors=None, normals=None, BRG2RGB=False):
    """save points to point cloud using open3d"""
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(path_save, cloud)


def colormap(img, cmap="jet") -> torch.Tensor:
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    # fig.colorbar(im, ax=ax) # remove colorbar
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img = torch.from_numpy(data / 255.0).float().permute(2, 0, 1)
    img = torch.from_numpy(data.copy()).float().permute(2, 0, 1)
    plt.close()
    return img / 255
