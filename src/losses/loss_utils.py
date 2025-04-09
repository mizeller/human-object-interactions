import torch
from math import exp
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import Variable
from pytorch3d.ops import laplacian
from src.common import torch_utils
from loguru import logger
from typing import Dict


###########################################################################
# LPIPS loss utils
###########################################################################
from lpips import LPIPS
from src.utils import sampler

# only init these two variables once;
lpips = LPIPS(net="vgg", pretrained=True).to("cuda")
for param in lpips.parameters():
    param.requires_grad = False

patch_sampler = sampler.PatchSampler(
    num_patch=4,
    patch_size=128,
    ratio_mask=0.9,
    dilate=0,
)


def get_lpips_loss(
    pred_img: torch.Tensor, gt_img: torch.Tensor, gt_msk: torch.Tensor
) -> torch.Tensor:
    assert gt_img.shape == gt_msk.shape == pred_img.shape, logger.error(
        "Shapes of image tensors must match!"
    )
    bg_color_lpips = torch.rand_like(pred_img)
    image_bg = pred_img * gt_msk + bg_color_lpips * (1.0 - gt_msk)
    gt_image_bg = gt_img * gt_msk + bg_color_lpips * (1.0 - gt_msk)
    _, pred_patches, gt_patches = patch_sampler.sample(
        gt_msk[0:1, :, :], image_bg, gt_image_bg
    )
    _loss = lpips(pred_patches.clip(max=1), gt_patches).mean()

    return _loss


###########################################################################
# SSIM loss utils
###########################################################################
def get_ssim_loss(
    pred_img: torch.Tensor, gt_img: torch.Tensor, gt_msk: torch.Tensor
) -> torch.Tensor:
    _loss = 1.0 - ssim(pred_img, gt_img)
    _loss = _loss * (gt_msk.sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
    return _loss


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


###########################################################################
###########################################################################


def total_variation_loss(img, mask=None):
    """
    Compute the scale-invariant total variation loss for an image.

    Parameters:
        img (torch.Tensor): Input image tensor.
        mask (torch.Tensor, optional): Optional mask tensor to apply the loss only on certain regions.

    Returns:
        torch.Tensor: Scale-invariant total variation loss.
    """
    assert len(img.size()) == 3, "Input image tensor must be 3D (H W C)"
    assert img.size(0) == 3, "Input image tensor must have 3 channels"
    # Calculate the total variation loss
    # Shift the image to get the differences in both x and y directions
    d_x = img[:, :, 1:] - img[:, :, :-1]
    d_y = img[:, 1:, :] - img[:, :-1, :]

    # Compute the L1 norm of the differences
    tv_loss = torch.sum(torch.abs(d_x)) + torch.sum(torch.abs(d_y))

    if mask is not None:
        tv_loss = tv_loss / mask.sum()
    else:
        # Normalize by the size of the image
        img_size = img.size(-1) * img.size(-2)
        tv_loss = tv_loss / img_size

    return tv_loss

def get_emd_loss(pred_sem_msk: torch.Tensor, gt_sem_msk: torch.Tensor):
    """Earth-Movers-Distance Loss (?)

    check out: 
    - https://openaccess.thecvf.com/content_cvpr_2018/papers/Alldieck_Video_Based_Reconstruction_CVPR_2018_paper.pdf eq. 4 in this paper.
    - https://github.com/thmoa/videoavatars/blob/febcbb8f514ed9c404aa11a92447f5dce8937bda/step1_pose.py#L190
    """
    logger.error("Not yet implemented!")
    return
    
    

def get_sem_loss(pred_sem_msk: torch.Tensor, gt_sem_msk: torch.Tensor):
    epsilon = 1e-4  # tolerance for floating-point comparison
    # make sure both masks are of shape H,W
    pred_sem_msk = pred_sem_msk[0,...] 
    gt_sem_msk = gt_sem_msk[0,...] 
    
    # Remap values with a tolerance
    semantic_gt = torch.round(gt_sem_msk.clone(), decimals=4)

    bnd_bg = semantic_gt < 0.0980 + epsilon
    bnd_o = torch.logical_and(
        0.0980 - epsilon <= semantic_gt, semantic_gt < 0.3922 + epsilon
    )
    bnd_r = torch.logical_and(
        0.3922 - epsilon <= semantic_gt, semantic_gt <= 1.000 + epsilon
    )

    # Apply remapping
    semantic_gt[bnd_bg] = 0.0000
    semantic_gt[bnd_o] = 0.1961
    semantic_gt[bnd_r] = 0.5882

    # Further remapping
    semantic_gt[torch.abs(semantic_gt - 0.0000) < epsilon] = 0
    semantic_gt[torch.abs(semantic_gt - 0.1961) < epsilon] = 1
    semantic_gt[torch.abs(semantic_gt - 0.5882) < epsilon] = 2

    semantic_gt_onehot = torch_utils.one_hot_embedding(
        labels=semantic_gt.to(gt_sem_msk.device), num_classes=3
    )  # shape: torch.Size([3, H, W, 3])

    sem_loss = l2_loss(network_output=pred_sem_msk, gt=semantic_gt_onehot)

    sem_loss = sem_loss.sum()
    return sem_loss


def get_l1_loss(
    pred_img: torch.Tensor, gt_img: torch.Tensor, gt_msk: torch.Tensor = None
) -> torch.Tensor:
    if gt_msk is not None:
        if gt_msk.shape[0] == 3:
            gt_msk = gt_msk[0, ...]
        return torch.abs((pred_img - gt_img)).sum() / gt_msk.sum()
    else:
        return torch.abs((pred_img - gt_img)).mean()


def l2_loss(pred_img, gt_img):
    return ((pred_img - gt_img) ** 2).mean()


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


def multivariate_normal_kl(mu_0, cov_0, mu_1, cov_1):
    # Create multivariate normal distributions
    mvn_0 = dist.MultivariateNormal(mu_0, covariance_matrix=cov_0)
    mvn_1 = dist.MultivariateNormal(mu_1, covariance_matrix=cov_1)

    # Calculate KL divergence
    kl_divergence = torch.distributions.kl.kl_divergence(mvn_0, mvn_1)

    return kl_divergence


def multivariate_normal_kl_v2(mu_0, cov_0, mu_1, cov_1):
    """
    Calculate the KL divergence between two batches of multivariate normal distributions.

    Parameters:
    - mu_0: Mean of the first distribution, shape (batch_size, n)
    - cov_0: Covariance matrix of the first distribution, shape (batch_size, n, n)
    - mu_1: Mean of the second distribution, shape (batch_size, n)
    - cov_1: Covariance matrix of the second distribution, shape (batch_size, n, n)

    Returns:
    - kl_divergence: KL divergence between the two batches of distributions, shape (batch_size,)
    """

    # Ensure covariance matrices are positive definite
    eye_like = torch.eye(3).to(cov_0)

    cov_0 = (cov_0 + cov_0.transpose(-2, -1)) / 2.0 + 1e-6 * eye_like.unsqueeze(0)
    cov_1 = (cov_1 + cov_1.transpose(-2, -1)) / 2.0 + 1e-6 * eye_like.unsqueeze(0)

    # Calculate KL divergence using the formula
    trace = lambda x: torch.einsum("...ii", x)
    term1 = 0.5 * (
        trace(cov_1.inverse() @ cov_0)
        + torch.sum(
            (mu_1 - mu_0).unsqueeze(-1).transpose(-2, -1)
            @ cov_1.inverse()
            @ (mu_1 - mu_0).unsqueeze(-1),
            dim=[-2, -1],
        )
    )
    term2 = -0.5 * mu_0.size(-1) + 0.5 * torch.log(cov_1.det() / cov_0.det())

    kl_divergence = term1 + term2

    return kl_divergence


def pcd_laplacian_smoothing(verts, edges, method: str = "uniform"):
    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        L = laplacian(verts, edges)

    loss = L.mm(verts)
    loss = loss.norm(dim=1)

    return loss.mean()


def gmof(x, sigma):
    """Geman-McClure error function"""
    x_squared = x**2
    sigma_squared = sigma**2
    loss_val = (sigma_squared * x_squared) / (sigma_squared + x_squared + 1e-8)
    return loss_val


###########################################################################
# 3D loss utils
###########################################################################
mse_loss = torch.nn.MSELoss(reduction="none")


def get_3D_smoothness_loss_old(v3d: Dict[str, torch.Tensor]):
    # src: https://github.com/zc-alexfan/hold/blob/31dd6128215ff11a966fc8893dff9a25f3cf16c2/generator/src/alignment/pl_module/ho.py#L57

    # FIXME: IMO this only makes sense for vertices belonging to the palm; not the whole hand!
    # TODO:  filter out vertices that belong to fingers & implement a separate loss term for them; i.e. contact w/ object!

    total_loss = 0.0

    for _, v in v3d.items():
        # 1. find mean 3D positions ∀ frames
        centroid = v.mean(dim=1)
        # 2. Δ centroid in consecutive frames
        diff = centroid[:-1] - centroid[1:]
        # 3. mean-squared-error between Δ & 0
        loss_smooth = mse_loss(
            diff, torch.zeros_like(diff).detach()
        ).mean()  # TODO: detach?
        # 4. accumulate loss
        total_loss += loss_smooth

    return total_loss


def _get_centroids(v3d: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Given the 3D vertices (H/O) in all frames, calculate the centroid position for all objects in all frames."""
    centroids = {}

    # get centroid position for all objects in all frames
    for _frame in v3d.keys():
        # _type: hand/object && _v3d: associated 3D vertices
        for _type, _v3d in v3d[_frame].items():
            if _v3d is None:
                # this frame was not seen during training...
                continue
            if _type not in centroids:
                centroids[_type] = []

            centroids[_type].append(_v3d.mean(dim=0))

    return centroids


def get_constant_distance_loss(
    v3d: Dict[int, Dict[str, torch.Tensor]], thresh: float = 0.01
):
    centroids = _get_centroids(v3d)

    # Stack centroids for all frames
    hand_centroids = torch.stack([c[0] for c in centroids["hand"]])
    object_centroids = torch.stack([c[0] for c in centroids["object"]])

    # Compute Euclidean distances for all frames
    distances = torch.norm(hand_centroids - object_centroids, dim=1)

    # Compute mean distance
    mean_distance = distances.mean()

    # Compute loss based on deviation from mean distance
    loss = torch.mean(
        torch.nn.functional.relu(torch.abs(distances - mean_distance) - thresh)
    )

    return loss


def get_pseudo_depth_loss(
    v3d: Dict[int, Dict[str, torch.Tensor]], thresh: float = 0.03
):
    # assuming the cam viewing direction to be the z-axis
    # compute the difference between the mean z-coordinates of the centroids
    centroids = _get_centroids(v3d)
    mean_hand_z = torch.stack(centroids["hand"])[:, 2].mean()
    mean_object_z = torch.stack(centroids["object"])[:, 2].mean()
    z_diff = mean_hand_z - mean_object_z
    loss = torch.nn.functional.relu(z_diff.abs() - thresh)
    return loss


def get_3D_smoothness_loss(v3d: Dict[int, Dict[str, torch.Tensor]]):
    # inspired by: https://github.com/zc-alexfan/hold/blob/31dd6128215ff11a966fc8893dff9a25f3cf16c2/generator/src/alignment/pl_module/ho.py#L57
    # FIXME: IMO this only makes sense for vertices belonging to the palm; not the whole hand!
    # TODO:  filter out vertices that belong to fingers & implement a separate loss term for them; i.e. contact w/ object!

    total_loss = 0.0
    centroids = _get_centroids(v3d)
    for _type, _centroids in centroids.items():
        _centroids = torch.stack(_centroids)  # Shape: (num_frames, 3)
        diff = _centroids[:-1] - _centroids[1:]
        loss_smooth = mse_loss(diff, torch.zeros_like(diff).detach()).mean()
        total_loss += loss_smooth

    return total_loss



###########################################################################
# DN-Splatter: 
###########################################################################