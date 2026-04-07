"""
Vectorized dsegment — computes distance from each point to each
line segment of a polygon.

Uses NumPy broadcasting for efficient batch computation.
"""

import numpy as np


def dsegment(p, pv):
    """
    Compute distance from each point p to each segment in pv.

    Parameters
    ----------
    p  : (N, 2) array of query points
    pv : (M, 2) array of polygon vertices.
         Segments are pv[0]->pv[1], pv[1]->pv[2], ..., pv[M-2]->pv[M-1].

    Returns
    -------
    ds : (N, M-1) array of distances from each point to each segment
    """
    p = np.asarray(p, dtype=float)
    pv = np.asarray(pv, dtype=float)

    # Segment start and direction vectors
    v0 = pv[:-1]                                # (M-1, 2)
    v  = pv[1:] - pv[:-1]                       # (M-1, 2) segment direction

    # Vector from each segment start to each query point
    w = p[:, None, :] - v0[None, :, :]           # (N, M-1, 2)

    # Dot products for projection
    c1 = np.sum(w * v[None, :, :], axis=2)       # (N, M-1) dot(w, v)
    c2 = np.sum(v * v, axis=1)                   # (M-1,)   dot(v, v)

    # Clamp projection parameter t to [0, 1]
    safe_c2 = np.where(c2 > 0, c2, 1.0)
    t = np.clip(c1 / safe_c2[None, :], 0.0, 1.0)  # (N, M-1)

    # Nearest point on each segment
    proj = v0[None, :, :] + t[:, :, None] * v[None, :, :]  # (N, M-1, 2)

    # Distance from query point to nearest point on segment
    diff = p[:, None, :] - proj                   # (N, M-1, 2)
    ds = np.sqrt(np.sum(diff ** 2, axis=2))       # (N, M-1)

    return ds
