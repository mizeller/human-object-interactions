import os
import trimesh
import os.path as op

from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.common.transforms import project2d, rigid_tf_torch_batch


def log(data: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Contents", show_header=True, header_style="bold magenta")
    table.add_column("Key", style="cyan")
    table.add_column("Shape", style="green")
    table.add_column("Dtype", style="yellow")
    table.add_column("Device", style="blue")

    def _process_dict(d, prefix=""):
        keys = sorted(d.keys())
        for k in keys:
            v = d[k]
            key_name = f"{prefix}{k}"
            if isinstance(v, dict):
                _process_dict(v, f"{key_name}.")
            elif isinstance(v, torch.Tensor):
                table.add_row(key_name, str(v.shape), str(v.dtype), str(v.device))
            else:
                table.add_row(key_name, "N/A", str(type(v)), "N/A")

    _process_dict(data)
    console.print(table)
    print("\n")


def all_arrays_are_equal(array_list: List[np.array]) -> bool:
    if len(array_list) < 2:
        return True
    first_array = array_list[0]
    return all(map(lambda x: np.array_equal(first_array, x), array_list))


def all_tensors_are_equal(tensor_list: List[torch.Tensor]) -> bool:
    if not tensor_list:
        return True
    first_tensor = tensor_list[0]
    for tensor in tensor_list[1:]:
        if not torch.equal(first_tensor, tensor):
            return False
    return True


def debug_params(self):
    return


def debug_deformer_mano(args, sample_dict, node):

    verts_c = node.server.verts_c
    faces = node.server.faces
    idx = sample_dict["idx"][0]
    output = sample_dict["output"]

    # save canonical meshes first
    mesh = trimesh.Trimesh(
        verts_c[0].cpu().numpy().reshape(-1, 3), faces, process=False
    )
    out_p = op.join(args.log_dir, "debug", f"mesh_{node.node_id}_cano", f"0_cano.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh.export(out_p)

    batch_size = sample_dict["batch_size"]

    # results from deformer
    verts = output["verts"].view(batch_size, -1, 3)[:1]
    x_c, outlier_mask = node.deformer.forward(
        verts.view(1, -1, 3),
        output["tfs"].view(batch_size, -1, 4, 4)[:1],
        return_weights=False,
        inverse=True,
        verts=verts,
    )
    mesh = trimesh.Trimesh(
        vertices=x_c.view(-1, 3).detach().cpu().numpy(),
        faces=faces,
        process=False,
    )
    out_p = op.join(args.log_dir, "debug", f"mesh_{node.node_id}_cano", f"{idx}.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh.export(out_p)

    # results of deformed space
    mesh = trimesh.Trimesh(
        vertices=verts.view(-1, 3).detach().cpu().numpy(),
        faces=faces,
        process=False,
    )
    out_p = op.join(
        args.log_dir, "debug", f"mesh_{node.node_id}_deform", f"{idx}_deform.obj"
    )
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh.export(out_p)


def debug_deformer_obj(args, sample_dict, node):
    obj_verts_c = node.server.verts_c
    idx = sample_dict["idx"][0]
    batch_size = sample_dict["batch_size"]

    # canonical mesh
    obj_output = sample_dict["obj_output"]
    mesh_obj = pts2mesh(obj_verts_c[0].cpu().numpy(), radius=0.01, num_samples=100)
    out_p = op.join(args.log_dir, "debug", "mesh_v3d_orig_homo", f"0_cano.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh_obj.export(out_p)

    # results from OBJ deformer
    obj_x = obj_output["verts"].view(batch_size, -1, 3)
    obj_x_c, _ = node.deformer.forward(obj_x, sample_dict["tfs"], inverse=True)
    mesh_obj = pts2mesh(
        obj_x_c[0].view(-1, 3).detach().cpu().numpy(), radius=0.01, num_samples=100
    )
    out_p = op.join(args.log_dir, "debug", "mesh_v3d_orig_homo", f"{idx}.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh_obj.export(out_p)

    # mesh in deform space
    obj_x = obj_output["verts"].view(batch_size, -1, 3)
    mesh_obj = pts2mesh(obj_x[0].detach().cpu().numpy(), radius=0.01, num_samples=100)
    out_p = op.join(args.log_dir, "debug", "mesh_obj_deform", f"{idx}_deform.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh_obj.export(out_p)

    # samples in deform space
    pts = sample_dict["points"].view(batch_size, -1, 3)
    mesh_obj = pts2mesh(
        pts[0].view(-1, 3).detach().cpu().numpy(), radius=0.01, num_samples=100
    )
    out_p = op.join(args.log_dir, "debug", "samples_obj_deform", f"{idx}.obj")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    mesh_obj.export(out_p)


def debug_deformer(sample_dicts, self):
    if not self.args.debug:
        return

    args = self.args

    for node in self.nodes.values():
        sample_dict = sample_dicts[node.node_id]
        if node.node_id in ["right", "left"]:
            debug_deformer_mano(args, sample_dict, node)
        elif node.node_id in ["object"]:
            debug_deformer_obj(args, sample_dict, node)
        else:
            assert False


def debug_world2pix(args, output, input, node_id):
    # Load data
    data = torch.load(op.join(args.log_dir, "dataset_info.pth"))
    K_cam = data["K_cam"]
    Rt_cam = data["Rt_cam"]
    img_paths = data["img_paths"]
    idx = int(input["idx"][0])

    # Load image
    im = Image.open(img_paths[idx])
    plt.imshow(im)

    # Perform transformations
    w2c = Rt_cam.clone().inverse()
    if "verts" in output.keys():
        v3d_deform = output["verts"].cpu()  # world coordinate
        v3d_cam = rigid_tf_torch_batch(
            points=v3d_deform[:1], R=w2c[:3, :3][None, :, :], T=w2c[:3, 3:4][None, :, :]
        )[0]
        trimesh.PointCloud(
            v3d_cam.detach().numpy(),
            np.tile(np.array([255, 0, 0, 255]), (len(v3d_cam), 1)),
        ).export("dbg/02_obj_v3d_cam_red.ply")
        v2d_obj = project2d(K_cam[:3, :3], v3d_cam).detach().numpy()
        plt.scatter(v2d_obj[:, 0], v2d_obj[:, 1], s=1, color="r", alpha=0.5)

    out_p = op.join(args.log_dir, "debug", "world2pix", node_id, f"world2pix_{idx}.png")
    os.makedirs(op.dirname(out_p), exist_ok=True)
    plt.savefig(out_p)
    plt.savefig("dbg/03_v3d_cam_reprojected.png")
    plt.close()
    return w2c, v3d_cam


def pts2mesh(pts, radius, num_samples):
    import random

    sampled_points_indices = random.sample(range(pts.shape[0]), num_samples)
    pts = pts[sampled_points_indices]

    # Initialize an empty scene
    scene = trimesh.Scene()

    # For each point, create a sphere and add it to the scene
    for pt in pts:
        # Create a sphere at the given point
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius, color=None)

        # Translate the sphere to the right position
        sphere.apply_translation(pt)

        # Add the sphere to the scene
        scene.add_geometry(sphere)

    # Combine all spheres into a single mesh
    combined = trimesh.util.concatenate(scene.dump())

    return combined


def debug_camera_location(cam_loc, filename: str = "dbg/XX_cam_loc.ply") -> None:
    trimesh.PointCloud(cam_loc[None, :], colors=np.array([0, 0, 0])).export(filename)
    return


def debug_rays(ray_dirs, cam_loc, filename: str = "dbg/XX_rays.ply") -> None:
    distances = np.linspace(0, 5, num=50)
    points = []
    for direction in ray_dirs:
        for distance in distances:
            point = cam_loc + distance * direction
            points.append(point)
    pts_on_rays = np.array(points)
    trimesh.PointCloud(vertices=pts_on_rays, colors=np.array([0, 0, 0, 100])).export(
        filename
    )
    return


def debug_extrinsics(R, T, filename: str = "dbg/XX_cam_frame.ply"):
    near, far, n = 0, 0.5, 10
    Rx = (np.linspace(near, far, n).reshape(n, 1) * R[0] + T).reshape(-1, 3)
    Ry = (np.linspace(near, far, n).reshape(n, 1) * R[1] + T).reshape(-1, 3)
    Rz = (np.linspace(near, far, n).reshape(n, 1) * R[2] + T).reshape(-1, 3)
    points = np.vstack((Rx, Ry, Rz))
    purple_color = np.array([255, 0, 0, 255])  # red for x-axis
    orange_color = np.array([0, 255, 0, 255])  # green for y-axis
    pink_color = np.array([0, 0, 255, 255])  # blue for z-axis
    colors = np.vstack(
        (
            np.tile(purple_color, (n, 1)),
            np.tile(orange_color, (n, 1)),
            np.tile(pink_color, (n, 1)),
        )
    )
    trimesh.PointCloud(vertices=points, colors=colors).export(filename)
    return