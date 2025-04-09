# Inspired by: hugs/utils/subdivide_smpl.py

# external
import torch
import pickle
import trimesh
import numpy as np
from trimesh import grouping
from trimesh.geometry import faces_to_edges
from typing import Union

# local
from smplx import MANO, SMPLX
from src.constants import mano_p, smplx_p, smplx2mano_p


def _subdivide(vertices, faces, face_index=None, vertex_attributes=None):
    if face_index is None:
        face_mask = np.ones(len(faces), dtype=bool)
    else:
        face_mask = np.zeros(len(faces), dtype=bool)
        face_mask[face_index] = True

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_mask]
    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)
    # the new faces_subset with correct winding
    f = np.column_stack(
        [
            faces_subset[:, 0],
            mid_idx[:, 0],
            mid_idx[:, 2],
            mid_idx[:, 0],
            faces_subset[:, 1],
            mid_idx[:, 1],
            mid_idx[:, 2],
            mid_idx[:, 1],
            faces_subset[:, 2],
            mid_idx[:, 0],
            mid_idx[:, 1],
            mid_idx[:, 2],
        ]
    ).reshape((-1, 3))

    # add the 3 new faces_subset per old face all on the end
    # # by putting all the new faces after all the old faces
    # # it makes it easier to understand the indexes
    new_faces = np.vstack((faces[~face_mask], f))
    # stack the new midpoint vertices on the end
    new_vertices = np.vstack((vertices, mid))

    if vertex_attributes is not None:
        new_attributes = {}
        for key, values in vertex_attributes.items():
            if key == "v_id":
                attr_mid = values[edges[unique][:, 0]]
            elif key == "lbs_weights":
                attr_mid = values[edges[unique]].mean(axis=1)
            elif key in ["is_mano_left", "is_mano_right"]:
                # Reshape boolean arrays to match vstack requirements
                attr_mid = np.any(values[edges[unique]], axis=1)
                values = values.reshape(-1, 1)
                attr_mid = attr_mid.reshape(-1, 1)
            elif key == "shapedirs":
                attr_mid = values[edges[unique]].mean(axis=1)
                attr_mid = attr_mid.reshape(-1, values.shape[1])
            else:
                attr_mid = values[edges[unique]].mean(axis=1)
            new_attributes[key] = np.vstack((values, attr_mid))
    return new_vertices, new_faces, new_attributes


def get_connected_vertices(faces, original_vertex_ids):
    """Get all vertices that are part of faces containing original vertices."""
    connected_faces = np.any(np.isin(faces, original_vertex_ids), axis=1)
    connected_vertices = np.unique(faces[connected_faces].ravel())
    return connected_vertices


def _subdivide_model(model, mano_vertex_ids=None):
    device = model.v_template.device
    n_verts = model.v_template.shape[0]
    init_posedirs = model.posedirs.detach().cpu().numpy()
    init_lbs_weights = model.lbs_weights.detach().cpu().numpy()
    init_shapedirs = model.shapedirs.detach().cpu().numpy()
    init_v_id = model.v_id

    # SMPL-X specific
    has_expr = hasattr(model, "expr_dirs")
    if has_expr:
        init_expr_dirs = model.expr_dirs.detach().cpu().numpy()
        init_expr_dirs = init_expr_dirs.reshape(n_verts, -1)

    init_shapedirs = init_shapedirs.reshape(n_verts, -1)
    init_J_regressor = model.J_regressor.detach().cpu().numpy().transpose(1, 0)
    init_vertices = model.v_template.detach().cpu().numpy()
    init_faces = model.faces
    vertex_attributes = {
        "v_id": init_v_id,
        "lbs_weights": init_lbs_weights,
        "shapedirs": init_shapedirs,
        "J_regressor": init_J_regressor,
        "is_mano_left": np.zeros(len(init_vertices), dtype=bool),
        "is_mano_right": np.zeros(len(init_vertices), dtype=bool),
    }

    # track MANO vertices
    if mano_vertex_ids is not None:
        left_connected = get_connected_vertices(
            init_faces, mano_vertex_ids["left_hand"]
        )
        right_connected = get_connected_vertices(
            init_faces, mano_vertex_ids["right_hand"]
        )
        vertex_attributes["is_mano_left"][left_connected] = True
        vertex_attributes["is_mano_right"][right_connected] = True

    if has_expr:
        vertex_attributes["expr_dirs"] = init_expr_dirs

    # subdivision
    sub_vertices, sub_faces, attr = _subdivide(
        vertices=init_vertices, faces=init_faces, vertex_attributes=vertex_attributes
    )

    # smoothing
    sub_mesh = trimesh.Trimesh(vertices=sub_vertices, faces=sub_faces)
    sub_mesh = trimesh.smoothing.filter_mut_dif_laplacian(
        sub_mesh,
        lamb=0.5,
        iterations=5,
        volume_constraint=True,
        laplacian_operator=None,
    )
    sub_vertices = sub_mesh.vertices

    if model.name() == "MANO":
        new_model = MANO(
            model_path=mano_p,
            use_pca=model.use_pca,
            flat_hand_mean=model.flat_hand_mean,
        )
    elif model.name() == "SMPL-X":
        new_model = SMPLX(
            model_path=smplx_p,
            gender=model.gender,
            num_pca_comps=model.num_pca_comps,
            flat_hand_mean=model.flat_hand_mean,
            use_pca=model.use_pca,
        )
        # new_model = SMPLX(model_path=smplx_p)

    n_pose_basis = init_posedirs.shape[0]
    n_new_verts = sub_vertices.shape[0]

    # set attrs for new_model
    new_model.lbs_weights = torch.from_numpy(attr["lbs_weights"]).float().to(device)
    new_model.posedirs = torch.zeros(
        (n_pose_basis, n_new_verts * 3), dtype=torch.float32, device=device
    )
    new_model.shapedirs = (
        torch.from_numpy(attr["shapedirs"].reshape(n_new_verts, 3, -1))
        .float()
        .to(device)
    )
    new_model.v_template = torch.from_numpy(sub_vertices).float().to(device)
    new_model.faces_tensor = torch.from_numpy(sub_faces).long().to(device)
    new_model.J_regressor = torch.zeros_like(
        torch.from_numpy(attr["J_regressor"].transpose(1, 0)), device=device
    )
    new_model.J_regressor[:, :n_verts] = model.J_regressor
    new_model.faces = sub_faces
    new_model.v_id = attr["v_id"].astype(int)

    if has_expr:
        new_model.expr_dirs = (
            torch.from_numpy(attr["expr_dirs"].reshape(n_new_verts, 3, -1))
            .float()
            .to(device)
        )

    # recompute connected vertices
    new_model.mano_vertex_ids = {
        "left_hand": np.where(attr["is_mano_left"])[0],
        "right_hand": np.where(attr["is_mano_right"])[0],
    }

    return new_model


SMPLX_MODEL_P = {
    "male": smplx_p / "SMPLX_MALE.npz",
    "female": smplx_p / "SMPLX_FEMALE.npz",
    "neutral": smplx_p / "SMPLX_NEUTRAL.npz",
}
import smplx


lookup = {
    "mano": MANO(model_path=mano_p, use_pca=False),
    "smplx": SMPLX(
        model_path=smplx_p,
        gender="neutral",
        num_pca_comps=45,
        flat_hand_mean=True,
        use_pca=False,
    ),
}


def subdivide_model(
    model: Union[SMPLX, MANO, None] = None, model_id: str = "mano", n_iter: int = 1
) -> Union[MANO, SMPLX]:
    """Pass in SMPLX/MANO model to sub-divide or get a sub-divded model by specifying the desired model_id."""

    if model is None:
        model = lookup.get(model_id, None)
        assert model, "Could not look up model."

    model.v_id = np.arange(model.v_template.shape[0])[..., None]

    # track MANO vertices for SMPLX
    # MANO_SMPLX_vertex_ids.pkl can be downloaded from the SMPL-X website
    mano_vertex_ids = None
    if model_id == "smplx":
        mano_vertex_ids = pickle.load(open(smplx2mano_p, "rb"))

    for _ in range(n_iter):
        model = _subdivide_model(model, mano_vertex_ids)
        mano_vertex_ids = model.mano_vertex_ids

    return model
