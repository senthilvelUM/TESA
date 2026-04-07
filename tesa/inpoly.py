"""
Point-in-polygon test.
"""

import numpy as np
from matplotlib.path import Path


def _point_to_segment_dist_sq(p, v0, v1):
    """
    Compute squared distance from each point to each line segment.

    Parameters
    ----------
    p : ndarray, shape (N, 2)
        Query points.
    v0 : ndarray, shape (M, 2)
        Segment start points.
    v1 : ndarray, shape (M, 2)
        Segment end points.

    Returns
    -------
    d_sq : ndarray, shape (N, M)
        Squared distance from point i to segment j.
    """
    edge = v1 - v0                          # (M, 2)
    len_sq = np.sum(edge ** 2, axis=1)      # (M,)
    # Parametric projection, clamped to [0,1]
    # t[i,j] = dot(p[i]-v0[j], edge[j]) / len_sq[j]
    diff = p[:, None, :] - v0[None, :, :]   # (N, M, 2)
    t = np.sum(diff * edge[None, :, :], axis=2)  # (N, M)
    safe_len = np.where(len_sq > 0, len_sq, 1.0)
    t = np.clip(t / safe_len[None, :], 0.0, 1.0)
    proj = v0[None, :, :] + t[:, :, None] * edge[None, :, :]  # (N, M, 2)
    return np.sum((p[:, None, :] - proj) ** 2, axis=2)         # (N, M)


def inpoly(p, poly, tol=1e-10):
    """
    Test whether points are inside or on the boundary of a polygon.

    Matches MATLAB's inpolygon: returns True for interior AND boundary points.

    Parameters
    ----------
    p    : (N, 2) array of query points
    poly : (M, 2) array of polygon vertices
    tol  : float, distance tolerance for boundary classification

    Returns
    -------
    inside : (N,) boolean array
    """
    p = np.asarray(p, dtype=float)
    poly = np.asarray(poly, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)

    path = Path(poly)
    inside = path.contains_points(p)

    # Check points on boundary: compute minimum distance to any polygon edge
    if not np.all(inside):
        # Build edge segments (close the polygon if needed)
        if not np.allclose(poly[0], poly[-1]):
            poly_closed = np.vstack([poly, poly[0:1]])
        else:
            poly_closed = poly
        v0 = poly_closed[:-1]
        v1 = poly_closed[1:]

        # Only check points not already classified as inside
        not_inside = ~inside
        idx = np.where(not_inside)[0]
        if len(idx) > 0:
            # For large inputs, process in batches to limit memory
            batch = 2000
            for start in range(0, len(idx), batch):
                end = min(start + batch, len(idx))
                d_sq = _point_to_segment_dist_sq(p[idx[start:end]], v0, v1)
                min_d = np.sqrt(np.min(d_sq, axis=1))
                inside[idx[start:end]] |= (min_d < tol)

    return inside
