import torch
from easydict import EasyDict
from gsplat import rasterization_2dgs

# from src.utils.vis_utils import colormap


def render(
    gaussians: dict,
    viewpoint_cam: EasyDict,
    bg_color: torch.Tensor,
    precomp_colors: bool = False,
):
    # extract vars from out dict
    means3D = gaussians["xyz"]
    feats = gaussians["shs"]
    opacity = gaussians["opacity"]
    scales = gaussians["scales"]
    rotations = gaussians["rotq"]
    active_sh_degree = gaussians["active_sh_degree"]

    if precomp_colors:
        # use pre-defined colors; no need to rasterize w/ sh
        colors = feats
        sh_degree = None
    else:
        # rasterize w/ sh
        colors = feats.unsqueeze(0)
        sh_degree = active_sh_degree

    # add alpha channel (required in 2DGS rasterization)
    alpha = torch.tensor([1.0], device="cuda:0")
    bg_color = torch.cat([bg_color, alpha])

    (
        renders,
        alphas,
        normals,  # ∈ [0,1]
        normals_from_depth,  # ∈ [-1,1]
        distortions,
        render_median,
        info,
    ) = rasterization_2dgs(
        means=means3D,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(),
        colors=colors.squeeze(),
        viewmats=viewpoint_cam.world_view_transform.unsqueeze(0),
        Ks=viewpoint_cam.cam_intrinsics.unsqueeze(0),
        width=viewpoint_cam.image_width,
        height=viewpoint_cam.image_height,
        sh_degree=sh_degree,
        backgrounds=bg_color.unsqueeze(0),
        packed=False,
        render_mode="RGB+D",
    )

    if renders.shape[-1] == 4:
        colors, depths = renders[..., 0:3], renders[..., 3:4]
    else:
        colors, depths = renders, None

    # rgb
    colors = torch.clamp(colors, 0.0, 1.0)
    colors = colors.squeeze().permute(2, 0, 1)  # C, H, W

    # (median) depth
    median_depth = (render_median - render_median.min()) / (
        render_median.max() - render_median.min()
    )

    # https://github.com/nerfstudio-project/gsplat/blob/bd64a47414e182dc105c1a2fdb6691068518d060/examples/simple_trainer_2dgs.py#L616
    # https://github.com/nerfstudio-project/gsplat/blob/bd64a47414e182dc105c1a2fdb6691068518d060/examples/simple_trainer_2dgs.py#L807
    normals = normals.squeeze(0).permute((2, 0, 1))
    normals_from_depth *= alphas.squeeze(0).detach()
    normals_from_depth = normals_from_depth.permute(2, 0, 1)

    # same [-1,1] -> [0,1] mapping as in DSINE
    normals_render = (normals_from_depth + 1.0) * 0.5

    # distortions
    # distortions = distortions.squeeze(-1)
    # distortions_render = (distortions - distortions.min()) / (
    # distortions.max() - distortions.min()
    # )
    # distortions_render = colormap(distortions_render.cpu().detach().numpy()[0])

    radii = info["radii"].squeeze(0)
    try:
        info["means2d"].retain_grad()
    except:
        pass

    # from torchvision.utils import save_image
    # save_image(colors, "tmp/00_rgb.png")
    # save_image(depths.squeeze(-1), "tmp/00_depth.png")
    # save_image(normals_render, "tmp/00_normals.png")
    # save_image(distortions_render, "tmp/00_distortions.png")

    return {
        "render": colors,  # [3, H, W]
        "alphas": alphas.squeeze(-1),  # [1, H, W]
        "depth": depths.squeeze(-1),  # [1, H, W]
        "median_depth": median_depth.squeeze(-1),  # [1, H, W], (for depth loss; TODO)
        "normals": normals,  # [3, H, W], for normal loss
        "normals_from_depth": normals_from_depth,  # [3, H , W]
        "normals_render": normals_render,  # [3, H , W], normal map
        "distortions": distortions.squeeze(-1),  # [1, H, W], for distortion loss
        # "distortions_render": distortions_render,  # [3, H, W], colormap
        # relevant k:v pairs for densification
        "viewspace_points": info["means2d"],  # [1, N, 2]
        "visibility_filter": radii > 0,  # [N]
        "radii": radii,  # [N]
    }
