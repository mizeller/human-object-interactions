from scipy.spatial import KDTree
import torch

def distCUDA2(points):
    """
    Replace simple_knn dependency of vanilla Gaussian Splatting w/ this method
    Source: https://github.com/graphdeco-inria/gaussian-splatting/issues/292#issuecomment-2007934451  
    """
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)