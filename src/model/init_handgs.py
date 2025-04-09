"""
With this file, the the human/hand gaussians can be initialized.
I.e. this file: ./assets/smplx_hands_init_gs_100000.pt was created
using this script.
"""

import torch
import argparse
import numpy as np
from typing import Union

# local
from smplx.body_models import MANO, SMPLX
from src import constants
from src.utils.subdivide_model import subdivide_model


def main(model_id: str, num_points: int):
    model: Union[MANO, SMPLX] = subdivide_model(model_id=model_id, n_iter=2)
    binding_counter = get_binding_counter(num_points=num_points, model=model)

    # load mesh, extract correspondences & save point-clouds (to double-check)
    # import pickle
    # from src.utils.ply import v3d_torch2ply
    # smplx = SMPLX(model_path=constants.smplx_p)
    # smplx2mano = pickle.load(open(constants.smplx_p/ "MANO_SMPLX_vertex_ids.pkl", "rb"))
    # smplx2flame = np.load(constants.smplx2flame_p)
    # v3d_flame = smplx.v_template[smplx2flame]
    # v3d_mano_l = smplx.v_template[smplx2mano["left_hand"]]
    # v3d_mano_l_sub = model.v_template[model.mano_vertex_ids["left_hand"]]
    # v3d_mano_r = smplx.v_template[smplx2mano["right_hand"]]
    # v3d_mano_r_sub = model.v_template[model.mano_vertex_ids["right_hand"]]
    # v3d_torch2ply(v3d_mano_l, "tmp/v3d_mano_l.ply")
    # v3d_torch2ply(v3d_mano_l_sub, "tmp/v3d_mano_l_sub.ply")
    # v3d_torch2ply(v3d_mano_r, "tmp/v3d_mano_r.ply")
    # v3d_torch2ply(v3d_mano_r_sub, "tmp/v3d_mano_r_sub.ply")
    # v3d_torch2ply(smplx.v_template, "tmp/v3d_smplx.ply")
    # v3d_torch2ply(model.v_template, "tmp/v3d_smplx_sub.ply")
    # v3d_torch2ply(v3d_flame, "tmp/v3d_flame.ply")

    # TODO: perhaps pickle & save the L/R MANO Vertex Mapping as well!
    torch.save(
        binding_counter, constants.assets_p / f"{model_id}_init_gs_{num_points}.pt"
    )


def get_binding_counter(num_points: int, model: Union[SMPLX, MANO]):
    verts = model.v_template.numpy()  # (N_verts, 3)
    faces = model.faces  # (N_faces, 3)

    # Get masks for faces containing hand vertices
    # left_hand_faces = np.any(
    #     np.isin(faces, model.mano_vertex_ids["left_hand"].reshape(-1)), axis=1
    # )
    # right_hand_faces = np.any(
    #     np.isin(faces, model.mano_vertex_ids["right_hand"].reshape(-1)), axis=1
    # )
    # hand_faces_mask = left_hand_faces | right_hand_faces

    # incl. human!
    hand_faces_mask = np.ones_like(hand_faces_mask)

    # Calculate areas only for hand faces
    cross_products = np.cross(
        verts[faces[hand_faces_mask]][:, 0] - verts[faces[hand_faces_mask]][:, 1],
        verts[faces[hand_faces_mask]][:, 1] - verts[faces[hand_faces_mask]][:, 2],
    )
    areas = np.linalg.norm(cross_products, axis=1) / 2

    # Initialize binding counter with zeros
    binding_counter = np.zeros(len(faces), dtype=np.int32)

    # Distribute gaussians only on hand faces
    if len(areas) > 0:  # Check if there are any hand faces
        weights = areas / areas.min()
        hand_face_gaussians = np.ceil(weights * (num_points / weights.sum())).astype(
            np.int32
        )
        binding_counter[hand_faces_mask] = hand_face_gaussians

    binding_counter = torch.from_numpy(binding_counter)  # (N_faces,)
    print(f"Input: {num_points}, Actual: {binding_counter.sum()}")
    print(f"Hand faces: {hand_faces_mask.sum()}/{len(faces)}")
    print(
        f"Min: {binding_counter[binding_counter > 0].min()}, Max: {binding_counter.max()}"
    )

    return binding_counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=False,
        default="smplx",
        choices=["mano", "smplx"],
    )
    parser.add_argument(
        "--num_gaussians",
        type=int,
        required=False,
        default=150_000,
    )
    args = parser.parse_args()
    main(model_id=args.model_id, num_points=args.num_gaussians)
