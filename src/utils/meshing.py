import numpy as np
import torch
import trimesh
from skimage import measure

from libmise import mise


def generate_mesh(func, verts, level_set=0, res_init=32, res_up=3, point_batch=5000):
    scale = 1.1  # Scale of the padded bbox regarding the tight one.

    # compute tight bbox around canonical mesh
    gt_bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
    gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5

    # scale is the max dimensions of the bbox
    gt_scale = (gt_bbox[1] - gt_bbox[0]).max()
    mesh_extractor = mise.MISE(res_init, res_up, level_set)
    # [0, 128]**3 grid
    points = mesh_extractor.query()

    # query occupancy grid
    while points.shape[0] != 0:
        orig_points = points
        points = points.astype(np.float32)

        # mesh_extractor.resolution: 128

        # [-0.5, 0.5]**3 grid
        # sampled points are around tight bbox of canonical mesh
        # "scale" is for a looser bbox around tight bbox
        points = (points / mesh_extractor.resolution - 0.5) * scale
        points = points * gt_scale + gt_center

        # points here are in the same space of the canonical mesh of  server
        points = torch.tensor(points).float().cuda()
        values = []
        for _, pnts in enumerate((torch.split(points, point_batch, dim=0))):
            out = func(pnts)
            values.append(out["sdf"].data.cpu().numpy().reshape(-1))
        values = np.concatenate(values, axis=0).astype(np.float64)

        mesh_extractor.update(orig_points, values)

        points = mesh_extractor.query()

    # [0, 128]**3 grid
    value_grid = mesh_extractor.to_dense()

    # marching cube
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        volume=value_grid, gradient_direction="ascent", level=level_set
    )

    # verts in [0, 128] space
    # convert to space of actual object bbox
    verts = (verts / mesh_extractor.resolution - 0.5) * scale
    verts = verts * gt_scale + gt_center
    faces = faces[:, [0, 2, 1]]
    meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)

    # remove disconnect part
    connected_comp = meshexport.split(only_watertight=False)
    max_area = 0
    max_comp = None
    for comp in connected_comp:
        if comp.area > max_area:
            max_area = comp.area
            max_comp = comp
    meshexport = max_comp

    return meshexport
