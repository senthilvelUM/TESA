"""
Area of triangles assuming CCW node ordering.
Author: Darren Engwirda (2007)
"""

import numpy as np


def triarea(p, t):
    """
    Signed area of triangles (positive for CCW ordering).

    Parameters
    ----------
    p : (N, 2) node coordinates
    t : (M, 3) triangle connectivity (0-based)

    Returns
    -------
    A : (M,) signed areas
    """
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=int)
    d12 = p[t[:, 1], :] - p[t[:, 0], :]
    d13 = p[t[:, 2], :] - p[t[:, 0], :]
    A = (d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0]) / 2.0
    return A
