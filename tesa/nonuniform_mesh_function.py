import numpy as np
from scipy.spatial import cKDTree


def nonuniform_mesh_function(p, pfix):
    """
    Nonuniform mesh size function based on distance to fixed points.

    Parameters
    ----------
    p : ndarray, shape (N, 2)
        Query points.
    pfix : ndarray, shape (M, 2)
        Fixed points (grain boundary nodes, etc.).

    Returns
    -------
    h : ndarray, shape (N,)
        Mesh size at each query point.
    """
    x = pfix[:, 0]
    y = pfix[:, 1]

    tree = cKDTree(np.column_stack([x, y]))
    dist, _ = tree.query(p)

    h = (dist / np.max(dist)) ** 2 + 1

    return h
