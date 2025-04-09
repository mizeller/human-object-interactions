import cv2
import numpy as np
import torch
import torch.nn.functional as F


def load_mask(mask_path, img_size) -> np.ndarray:
    if mask_path is None:
        mask_fake = np.zeros(img_size, dtype=np.uint8)
        return mask_fake
    mask = cv2.imread(mask_path)
    assert mask.max() != 255, "using original mask, not segm mask from aitviewer"
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def load_image(rgb_p) -> np.ndarray:
    img = cv2.imread(rgb_p)
    # preprocess: BGR -> RGB -> Normalize
    img = img[:, :, ::-1] / 255
    return img


def bilinear_interpolation(xs, ys, dist_map):
    x1 = np.floor(xs).astype(np.int32)
    y1 = np.floor(ys).astype(np.int32)
    x2 = x1 + 1
    y2 = y1 + 1

    dx = np.expand_dims(np.stack([x2 - xs, xs - x1], axis=1), axis=1)
    dy = np.expand_dims(np.stack([y2 - ys, ys - y1], axis=1), axis=2)
    Q = np.stack(
        [dist_map[x1, y1], dist_map[x1, y2], dist_map[x2, y1], dist_map[x2, y2]], axis=1
    ).reshape(-1, 2, 2)
    return np.squeeze(dx @ Q @ dy)  # ((x2 - x1) * (y2 - y1)) = 1


def get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max):
    samples_uniform_row = samples_uniform[:, 0]
    samples_uniform_col = samples_uniform[:, 1]
    index_outside = np.where(
        (samples_uniform_row < bbox_min[0])
        | (samples_uniform_row > bbox_max[0])
        | (samples_uniform_col < bbox_min[1])
        | (samples_uniform_col > bbox_max[1])
    )[0]
    return index_outside


def weighted_sampling(data, img_size, num_sample, hand_flag, bbox_ratio=0.9):
    from src.constants import SEGM_IDS

    """
    More sampling within the bounding box
    """

    # calculate bounding box
    mask = data["obj_mask"]
    num_sample_bbox = int(num_sample * bbox_ratio)
    num_sample_bbox_o = num_sample_bbox // 2
    num_sample_bbox_h = num_sample_bbox - num_sample_bbox_o

    # object samples
    samples_list = []
    where_o = np.asarray(np.where(mask == SEGM_IDS["object"]))
    where_h = np.asarray(np.where(mask == SEGM_IDS[hand_flag]))

    # no segm on either one, then use all for the one with segm
    if where_o.sum() < 10:
        num_sample_bbox_o = 0
        num_sample_bbox_h = num_sample_bbox

    if where_h.sum() < 10:
        num_sample_bbox_o = num_sample_bbox
        num_sample_bbox_h = 0

    if where_o.sum() < 10 and where_h.sum() < 10:
        num_sample_bbox_o = 0
        num_sample_bbox_h = 0

    if num_sample_bbox_o > 0:
        bbox_min = where_o.min(axis=1)
        bbox_max = where_o.max(axis=1)

        samples_bbox_o = np.random.rand(num_sample_bbox_o, 2)
        samples_bbox_o = samples_bbox_o * (bbox_max - bbox_min) + bbox_min
        samples_list.append(samples_bbox_o)

    # hand samples
    if num_sample_bbox_h > 0:
        bbox_min = where_h.min(axis=1)
        bbox_max = where_h.max(axis=1)

        samples_bbox_h = np.random.rand(num_sample_bbox_h, 2)
        samples_bbox_h = samples_bbox_h * (bbox_max - bbox_min) + bbox_min
        samples_list.append(samples_bbox_h)

    samples_bbox = np.concatenate(samples_list, axis=0)

    # uniform samples
    where = np.asarray(np.where(mask > 0))
    bbox_min = where.min(axis=1)
    bbox_max = where.max(axis=1)
    num_sample_uniform = num_sample - samples_bbox.shape[0]
    samples_uniform = np.random.rand(num_sample_uniform, 2)
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    # get indices for uniform samples outside of bbox
    index_outside = (
        get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max) + num_sample_bbox
    )

    indices = np.concatenate([samples_bbox, samples_uniform], axis=0)
    if "x2d" in data.keys():
        x2d = data["x2d"]
        # xy -> yx to be consistent with uv convention
        x2d = np.flip(x2d, axis=1)
        indices = np.concatenate((indices, x2d), axis=0)

    indices[:, 0] = np.clip(indices[:, 0], 0, img_size[0] - 2)
    indices[:, 1] = np.clip(indices[:, 1], 0, img_size[1] - 2)
    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [
                    bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                    for i in range(val.shape[2])
                ],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    return output, index_outside


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3)).cuda()
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi**2 + qj**2)
    return R


def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (
        (
            x
            - cx.unsqueeze(-1)
            + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
            - sk.unsqueeze(-1) * y / fy.unsqueeze(-1)
        )
        / fx.unsqueeze(-1)
        * z
    )
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


def get_camera_params(uv, pose, intrinsics, debug: bool = False):
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        p = torch.eye(4).repeat(pose.shape[0], 1, 1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else:  # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    if debug:
        from utils import debug
        import os

        if not os.path.exists("dbg/XX_world_frame.ply"):
            debug.debug_extrinsics(
                R=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                T=np.array([0, 0, 0]),
                filename="dbg/XX_world_frame.ply",
            )
        debug.debug_extrinsics(
            R=pose[0][:3, :3].cpu().numpy(),
            T=cam_loc[0].cpu().numpy(),
            filename="dbg/XX_cam_frame.ply",
        )
        debug.debug_camera_location(cam_loc[0].cpu().numpy())
        debug.debug_rays(ray_dirs[0].cpu().numpy(), cam_loc[0].cpu().numpy())

    return ray_dirs, cam_loc


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def split_input(model_input, total_pixels, n_pixels=10000):
    """
    Split the input to fit Cuda memory for large resolution.
    Can decrease the value of n_pixels in case of cuda out of memory error.
    """
    # n_pixels = 512
    split = []

    for i, indx in enumerate(
        torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)
    ):
        data = model_input.copy()
        data["uv"] = torch.index_select(model_input["uv"], 1, indx)
        split.append(data)
    return split


def merge_output(res, total_pixels, batch_size):
    """Merge the split output."""

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, 1) for r in res], 1
            ).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res], 1
            ).reshape(batch_size * total_pixels, -1)
    return model_outputs
