import torch
import smplx
import numpy as np

from src import constants, common


class MANODecimator:
    def __init__(self):
        data = np.load(
            constants.arctic_meta_p / "mano_decimator_195.npy",
            allow_pickle=True,
        ).item()
        mydata = {}
        for key, val in data.items():
            # only consider decimation matrix so far
            if "D" in key:
                mydata[key] = torch.FloatTensor(val)
        self.data = mydata

    def downsample(self, verts, is_right):
        dev = verts.device
        flag = "right" if is_right else "left"
        if self.data[f"D_{flag}"].device != dev:
            self.data[f"D_{flag}"] = self.data[f"D_{flag}"].to(dev)
        D = self.data[f"D_{flag}"]
        batch_size = verts.shape[0]
        D_batch = D[None, :, :].repeat(batch_size, 1, 1)
        verts_sub = torch.bmm(D_batch, verts)
        return verts_sub


SEAL_FACES_R = [
    [120, 108, 778],
    [108, 79, 778],
    [79, 78, 778],
    [78, 121, 778],
    [121, 214, 778],
    [214, 215, 778],
    [215, 279, 778],
    [279, 239, 778],
    [239, 234, 778],
    [234, 92, 778],
    [92, 38, 778],
    [38, 122, 778],
    [122, 118, 778],
    [118, 117, 778],
    [117, 119, 778],
    [119, 120, 778],
]

# vertex ids around the ring of the wrist
CIRCLE_V_ID = np.array(
    [108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120],
    dtype=np.int64,
)


def seal_mano_mesh(v3d, faces, is_rhand):
    # v3d: B, 778, 3
    # faces: 1538, 3
    # output: v3d(B, 779, 3); faces (1554, 3)
    seal_faces = torch.LongTensor(np.array(SEAL_FACES_R)).to(faces.device)
    if not is_rhand:
        # left hand
        seal_faces = seal_faces[:, np.array([1, 0, 2])]  # invert face normal
    centers = v3d[:, CIRCLE_V_ID].mean(dim=1)[:, None, :]
    sealed_vertices = torch.cat((v3d, centers), dim=1)
    faces = torch.cat((faces, seal_faces), dim=0)
    return sealed_vertices, faces


def seal_mano_mesh_np(v3d, faces, is_rhand):
    # v3d: B, 778, 3
    # faces: 1538, 3
    # output: v3d(B, 779, 3); faces (1554, 3)
    seal_faces = np.array(SEAL_FACES_R)
    if not is_rhand:
        # left hand
        seal_faces = seal_faces[:, np.array([1, 0, 2])]  # invert face normal
    centers = v3d[:, CIRCLE_V_ID].mean(axis=1)[:, None, :]
    sealed_vertices = np.concatenate((v3d, centers), axis=1)
    faces = np.concatenate((faces, seal_faces), axis=0)
    return sealed_vertices, faces


def seal_mano_v(v3d):
    # v3d: B, 778, 3
    # output: v3d(B, 779, 3)
    centers = v3d[:, CIRCLE_V_ID].mean(dim=1)[:, None, :]
    sealed_vertices = torch.cat((v3d, centers), dim=1)
    return sealed_vertices


def seal_mano_f(faces, is_rhand):
    # faces: 1538, 3
    # output: faces (1554, 3)
    seal_faces = torch.LongTensor(np.array(SEAL_FACES_R)).to(faces.device)
    if not is_rhand:
        # left hand
        seal_faces = seal_faces[:, np.array([1, 0, 2])]  # invert face normal
    faces = torch.cat((faces, seal_faces), dim=0)
    return faces


def build_layers(device=None):
    from common.object_tensors import ObjectTensors

    layers = {
        "right": build_mano_aa(True),
        "left": build_mano_aa(False),
        "object_tensors": ObjectTensors(),
    }

    if device is not None:
        layers["right"] = layers["right"].to(device)
        layers["left"] = layers["left"].to(device)
        layers["object_tensors"].to(device)
    return layers


def _build_smplx(
    batch_size: int = 1, gender: str = "neutral", v_template: np.ndarray = None
) -> smplx.SMPLX:
    return smplx.SMPLX(
        model_path=constants.smplx_p,
        gender=gender,
        batch_size=batch_size,
        num_betas=10,
        num_pca_comps=45,
        flat_hand_mean=True,
        use_pca=False,
        v_template=v_template,
    )


def build_subject_smplx(subject_id: str = "", batch_size: int = 1) -> smplx.SMPLX:
    vtemplate_p = constants.arctic_meta_p / "subject_vtemplates" / f"{subject_id}.obj"

    # try to load the subject properties
    if vtemplate_p.exists():
        mesh = common.mesh.Mesh(filename=vtemplate_p)
        vtemplate = mesh.v
        gender = constants.arctic_meta[subject_id]["gender"]
        return _build_smplx(batch_size, gender, vtemplate)

    return _build_smplx()  # neutral, default


def build_mano_aa(is_rhand, flat_hand=False):
    return smplx.MANO(
        constants.mano_p,
        use_pca=False,
        flat_hand_mean=flat_hand,
        is_rhand=is_rhand,
    )


def construct_layers(dev):
    mano_layers = {
        "right": build_mano_aa(True),
        "left": build_mano_aa(False),
        "smplx": _build_smplx(1, "neutral", None),
    }
    for layer in mano_layers.values():
        layer.to(dev)
    return mano_layers
