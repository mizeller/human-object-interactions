"""Useful class for storing & saving renders (depth/normal/rgb/...)"""

import torch
from typing import List
from pathlib import Path
from typing import Union
from loguru import logger
from torchvision.utils import save_image, make_grid

# local
from src.utils.image import add_caption
from src import constants


class Render:
    def __init__(self, mode: str, name: str, img: torch.Tensor):
        self.mode: str = mode
        self.name: str = name
        self.tensor: torch.Tensor = img

    def save(self, prefix: str = ""):
        """Save render to disk."""
        out_p: Path = Path.cwd() / f"{prefix}{self.mode}_{self.name}.png"
        logger.info(f"Saved render to: {out_p}")
        try:
            save_image(tensor=self.tensor, fp=out_p)
        except Exception as e:
            logger.error(f"Could not save tensor {out_p}")
            logger.warning(f"Error: {e}")
        return


class RenderManager:
    """
    Wrapper class for storing & saving all sorts of different renders (depth/normal/rgb/...).
    & associated methods.
    """

    def __init__(self):
        self.device = constants.device
        self._renders: List[Render] = []
        self.bg_color: float = 0.0  # black, default

    def __repr__(self):
        """Log all renders to console."""
        out = ["Renders in package:"]
        for render in self._renders:
            t = render.tensor
            s = f"\t{render.mode}_{render.name}"
            s += f"\t{t.shape}"
            s += f"\t{t.dtype}"
            s += f"\t{t.device}"
            s += f"\t{t.min().item():.4f}-{t.max().item():.4f}"
            out.append(s)
        return "\n".join(out)

    def __len__(self):
        return len(self._renders)

    def __iter__(self):
        """Iterate over stored renders."""
        return iter(self._renders)

    def reset(self) -> None:
        """Reset renders."""
        self._renders = []
        return

    def append(self, *renders) -> None:
        for render in renders:
            if render.tensor is None:
                render.tensor = self.get("gt", "placeholder")
            self._renders.append(render)
        return

    def dump(self) -> None:
        """Save *all* renders in to disk."""
        if not (Path.cwd() / "tmp").exists():
            (Path.cwd() / "tmp").mkdir(parents=True, exist_ok=True)
        for render in self._renders:
            render.save(prefix="tmp/dump_")
        return

    def save(
        self,
        run_mode: str = "train",
        render_mode: str = "h",
        out_p: Path = None,
        v: str = "",
    ) -> None:
        """
        Create different image grids for training/animation purposes.
        Save to `out_p` if specified, else return image tensor.
        """
        imgs = []
        # imgs_per_row: int = 2

        # _k = "pred_img_full" if render_mode == "h" else "pred_img"
        # gs_img = add_caption(self.get(render_mode, _k), f"Predicted Image{v}")
        # gt_img = add_caption(self.get(render_mode, "gt_img"), f"Original Image{v}")

        imgs.extend([self.get(render_mode, "pred_img") * 255])
        imgs_per_row = 1
        # imgs.extend([gt_img, gs_img])  # for animation, just incl. gt/pred rgb pixels
        # gt_msk = add_caption(self.get(render_mode, "gt_msk"), "Mask - GT")
        # pred_msk = add_caption(self.get(render_mode, "pred_msk"), "Mask - GS")
        # gt_normal = add_caption(self.get(render_mode, "gt_normal"), "Normals - GT")
        # pred_normal = add_caption(self.get(render_mode, "pred_normal"), "Normals - GS")

        # if run_mode == "train":
        #     # add masks as well...
        #     imgs.extend([gt_msk, pred_msk])
        # imgs.extend([gt_normal, pred_normal])

        img_grid = make_grid(imgs, normalize=True, nrow=imgs_per_row, padding=0)

        if out_p is not None:
            save_image(img_grid, out_p)

        return img_grid

    def get(self, mode: str, name: str) -> Union[torch.Tensor, None]:
        """Get specific render by name & render mode. (Assuming render.name + render.mode is unique!)"""
        _render = [
            render
            for render in self._renders
            if render.mode == mode and render.name == name
        ]

        if not _render:  # no render w/ name+mode found -> return placeholder.
            if name != "normals_gt":  # TODO: fix ugly if statement
                # NOTE: ∄ gt mesh for object -> ∄ normals_gt render
                # NOTE: this assumes at least one valid render to be present
                logger.warning(
                    f"No {mode}-render named {name} found! Returning placeholder."
                )

            return self.bg_color * torch.ones_like(self._renders[0].tensor)
        else:
            return _render[0].tensor

    def add_gt_renders(
        self, data: dict, bg_color: Union[torch.Tensor, None] = None
    ) -> None:
        # segmentation mask -> binary mask
        if bg_color is None:
            bg_color = 0.498 * torch.ones(3, dtype=torch.float32, device=self.device)
        combined_mask = (data["gt_msk"] != 0).float()

        msk_human = (data["gt_msk"] == constants.SEGM_IDS["human"]).float()
        msk_object = (data["gt_msk"] == constants.SEGM_IDS["object"]).float()

        self.append(
            # Render(mode="gt", name="gt_img", img=data["gt_img"]),
            Render(
                mode="gt",
                name="placeholder",
                img=bg_color[:, None, None] * torch.ones_like(data["gt_img"]),
            ),
            # h
            Render(mode="h", name="gt_msk", img=msk_human),
            Render(
                mode="h",
                name="gt_img",
                img=bg_color[:, None, None] * (1.0 - msk_human)
                + data["gt_img"] * msk_human,
            ),
            # Render(
            #     mode="h",
            #     name="gt_normal",
            #     img=bg_color[:, None, None] * (1.0 - msk_human)
            #     + data["gt_normal"] * msk_human,
            # ),
            # o
            Render(mode="o", name="gt_msk", img=msk_object),
            Render(
                mode="o",
                name="gt_img",
                img=bg_color[:, None, None] * (1.0 - msk_object)
                + data["gt_img"] * msk_object,
            ),
            # Render(
            #     mode="o",
            #     name="gt_normal",
            #     img=bg_color[:, None, None] * (1.0 - msk_object)
            #     + data["gt_normal"] * msk_object,
            # ),
            # ho
            Render(
                mode="ho",
                name="gt_img",
                img=bg_color[:, None, None] * (1.0 - combined_mask)
                + data["gt_img"] * combined_mask,
            ),
            # Render(
            #     mode="ho",
            #     name="gt_normal",
            #     img=bg_color[:, None, None] * (1.0 - combined_mask)
            #     + data["gt_normal"] * combined_mask,
            # ),
            Render(mode="ho", name="gt_msk", img=data["gt_msk"] / 255),
        )
        return
