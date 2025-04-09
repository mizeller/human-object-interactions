# external
import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union
from pathlib import Path
from loguru import logger

sys.path.append(str(Path.cwd()))
# internal
from src import constants, model, utils
from smplx import MANO

# the hand ckpt contains a tuple of len(19);
# for eval we only care about the following params:
hand_ckpt_mapping = {
    1: "v3d_h_local",  # 3D vertices of hand w.r.t local face frames of reference
    7: "binding",  # mapping between v3d & mano faces!
    14: "hand_pose",
    15: "betas",
    16: "global_orient",
    17: "transl",
    18: "vert_offsets",
}

CWD: Path = Path.cwd()
DATA_DIR: Path = CWD / "data"


def get_bbox_centers(
    vertices: Union[np.ndarray, torch.Tensor, List[np.ndarray]],
) -> np.ndarray:
    """
    Compute the centers of the tight bounding box for a moving point cloud.

    Parameters:
    - vertices: A numpy array of shape (frames, num_verts, 3) representing the vertices of the object over time.

    Returns:
    - A numpy array of shape (frames, 3) where each row represents the center of the bounding box for each frame.
    """

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()

    if isinstance(vertices, list):
        bbox_centers = []
        for verts in vertices:
            assert verts.shape[1] == 3
            bmin = np.min(verts, axis=0)
            bmax = np.max(verts, axis=0)
            bbox_center = (bmin + bmax) / 2
            bbox_centers.append(bbox_center)
        bbox_centers = np.stack(bbox_centers, axis=0)
    else:
        bbox_min = np.min(vertices, axis=1)
        bbox_max = np.max(vertices, axis=1)
        bbox_centers = (bbox_min + bbox_max) / 2
    return bbox_centers


def transform_vertices(vertices, transform_matrix):
    v_homo = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    v_transformed_homo = (transform_matrix @ v_homo.T).T
    return v_transformed_homo[:, :3]


@torch.no_grad()
def load_data(
    ckpt_p: Path,
    cfg: dict,
    ckpt_id: str = "final",
    camera_dict: dict = None,  # pass in the camera params, if they're not in the ckpt!
):
    """
    NOTE: check load_data() in hold/code/src/utils/io/ours.py for reference!
    NOTE: check hold/code/scripts_arctic/extract_preds.py for reference!
    """
    assert ckpt_p.exists(), logger.error(f"Checkpoint path {ckpt_p} does not exist")

    # get ckpts
    ckpts: Dict[str, Path] = {
        "h": ckpt_p / f"human_{cfg['seq']}_{ckpt_id}.pth",
        "o": ckpt_p / f"object_{cfg['seq']}_{ckpt_id}.pth",
    }

    for k, v in ckpts.items():
        assert v.exists(), FileNotFoundError(v)

    # get gaussian models
    hgs: model.human_gs.HumanGS = model.human_gs.HumanGS(cfg=cfg, data=None)
    hgs.restore(torch.load(ckpts["h"]), cfg["human"]["lr"]),
    mano_layer = MANO(model_path=constants.mano_p, use_pca=False, flat_hand_mean=True)
    mano_faces = torch.tensor(mano_layer.faces.astype(np.int32), dtype=torch.int16)

    smplx2mano = pickle.load(open(constants.smplx2mano_p, "rb"))
    # v3d_mano_l_orig = hgs.smplx.v_template[smplx2mano["left_hand"]]
    # v3d_mano_r_orig = hgs.smplx.v_template[smplx2mano["right_hand"]]
    # v3d_mano_l_sub = hgs.smplx.v_template[hgs.smplx.mano_vertex_ids["left_hand"]]
    # v3d_mano_r_sub = hgs.smplx.v_template[hgs.smplx.mano_vertex_ids["right_hand"]]
    # utils.ply.v3d_torch2ply(v3d_mano_l_sub, "tmp/aa_v3d_mano_l_sub.ply")
    # utils.ply.v3d_torch2ply(v3d_mano_r_sub, "tmp/aa_v3d_mano_r_sub.ply")
    # utils.ply.v3d_torch2ply(v3d_mano_l_orig, "tmp/aa_v3d_mano_l_orig.ply")
    # utils.ply.v3d_torch2ply(v3d_mano_r_orig, "tmp/aa_v3d_mano_r_orig.ply")
    # utils.ply.v3d_torch2ply(hgs.smplx.v_template, "tmp/aa_v3d_smplx.ply")

    ogs: model.object_gs.ObjectGS = model.object_gs.ObjectGS(cfg=cfg, data=None)
    ogs.restore(torch.load(ckpts["o"]), cfg["object"]["lr"]),

    # Set random indices once before the loop
    ogs.forward(0)
    if ogs.out["xyz"].shape[0] > 10000:
        random_indices = np.random.choice(ogs.out["xyz"].shape[0], 10000, replace=False)
        logger.warning(f"Will sample 10k object gaussians randomly.")

    empty_faces_tensor = torch.randint(0, 10, size=(10, 3), dtype=torch.int16)
    data = {
        # gaussians
        "v3d_gs_o": [],  # deformed obj. gs
        # extracted from smplx mesh!
        "j3d_lh": [],  # left hand joints
        "v3d_lh": [],  # left hand vertices
        "j3d_rh": [],  # right hand joints
        "v3d_rh": [],  # right hand vertices
        # # additional k:v paris in gt .pt files
        "faces": {"object": empty_faces_tensor},  # required
    }

    cam_ext = camera_dict["extrinsics"].numpy()  # cam from PromptHMR

    for idx in tqdm(range(0, ogs.obj_rot.shape[0]), desc="Deforming Hand & Object"):
        # --- human ---
        hgs.forward(idx)
        _v3d_smplx = hgs.current_mesh.verts_packed().detach().cpu().numpy()
        _j3d_smplx = hgs.out["joints"].detach().cpu().numpy()  # (127, 3)

        # --- object ---
        ogs.forward(idx)
        _v3d_gs_o = ogs.out["xyz"].detach().cpu().numpy()
        if len(_v3d_gs_o) > 10000:
            # down-sample
            _v3d_gs_o = _v3d_gs_o[random_indices]

        # --- map human & object from PromptHMR Cam to Cam @ origin
        v3d_smplx = transform_vertices(vertices=_v3d_smplx, transform_matrix=cam_ext)
        j3d_smplx = transform_vertices(vertices=_j3d_smplx, transform_matrix=cam_ext)
        v3d_gs_o = transform_vertices(vertices=_v3d_gs_o, transform_matrix=cam_ext)

        # --- extract hand joints & vertices
        v3d_lh = v3d_smplx[smplx2mano["left_hand"]]  # (778, 3)
        v3d_rh = v3d_smplx[smplx2mano["right_hand"]]  # (778, 3)
        j3d_lh = j3d_smplx[constants.left_hand_indices, :]  # (21, 3)
        j3d_rh = j3d_smplx[constants.right_hand_indices, :]  # (21, 3)

        # --- combine ---
        data["j3d_lh"].append(j3d_lh)
        data["v3d_lh"].append(v3d_lh)
        data["j3d_rh"].append(j3d_rh)
        data["v3d_rh"].append(v3d_rh)
        data["v3d_gs_o"].append(v3d_gs_o)

    for k, v in data.items():
        if k == "faces":
            continue
        data[k] = torch.from_numpy(np.array(v))

    # set 3D vertices
    out = {
        "v3d_c.left": data["v3d_lh"],
        "v3d_c.right": data["v3d_rh"],
        "v3d_c.object": data["v3d_gs_o"],
        # "j3d_c.left": data["j3d_lh"],
        # "j3d_c.right": data["j3d_rh"],
        "faces": data["faces"],  # has to be nested dict!
    }

    # add root aligned values
    root_lh = data["j3d_lh"][:, 0:1, :]
    root_rh = data["j3d_rh"][:, 0:1, :]
    root_o = torch.from_numpy(
        np.expand_dims(get_bbox_centers(data["v3d_gs_o"]), axis=1)
    )

    out.update(
        {
            # required: should be correct
            "j3d_ra.right": data["j3d_rh"] - root_rh,
            "j3d_ra.left": data["j3d_lh"] - root_lh,
            "v3d_right.object": data["v3d_gs_o"] - root_rh,
            "v3d_left.object": data["v3d_gs_o"] - root_lh,
            "v3d_ra.object": data["v3d_gs_o"] - root_o,
            # bonus - not required
            # "v3d_ra.right": data["v3d_rh"] - root_rh,
            # "v3d_ra.left": data["v3d_lh"] - root_lh,
            # "root.left": root_lh.squeeze(dim=1),
            # "root.right": root_rh.squeeze(dim=1),
            # "root.object": root_o.squeeze(dim=1),
        }
    )

    # convert OpenGL to OpenCV (for eval server) and reduce precision to float16
    for key in list(out.keys()):
        if key == "faces":
            continue
        tensor = out[key]
        if tensor.shape[-1] == 3:
            # float32 -> float16
            out[key] = tensor.to(torch.float16)

    # TODO: compute the 2D projections (if required)

    return out


if __name__ == "__main__":
    foo = load_data()

    pass
