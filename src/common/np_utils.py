import numpy as np


def permute_np(x, idx):
    original_perm = tuple(range(len(x.shape)))
    x = np.moveaxis(x, original_perm, idx)
    return x

def batches_are_equal(batch1, batch2, precision=4):
    """Check if two np.arrays are equal within a certain precision level."""
    if batch1.shape != batch2.shape:
        print(f"Shapes are different: {batch1.shape} vs {batch2.shape}; cannot compare!")
        return False

    difference = np.abs(batch1 - batch2)
    tolerance = 10 ** -precision

    return np.all(difference < tolerance)