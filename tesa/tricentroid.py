"""
Centroid of triangles.
"""

import numpy as np


def tricentroid(p, t):
    """
    Centroid of each triangle.

    Parameters
    ----------
    p : (N, 2) node coordinates
    t : (M, 3) triangle connectivity (0-based)

    Returns
    -------
    tC : (M, 2) centroids
    """
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=int)
    tC = np.column_stack([
        p[t[:, 0], 0] + p[t[:, 1], 0] + p[t[:, 2], 0],
        p[t[:, 0], 1] + p[t[:, 1], 1] + p[t[:, 2], 1]
    ]) / 3.0
    return tC
