"""
Signed distance function for a polygon.
Copyright (C) 2004-2012 Per-Olof Persson.
"""

import numpy as np
from .dsegment import dsegment
from .inpoly import inpoly


def dpoly(p, pv):
    """
    Signed distance from points to polygon boundary.
    Negative inside, positive outside.

    Parameters
    ----------
    p  : (N, 2) array of query points
    pv : (M, 2) array of polygon vertices (closed: first == last, or will be closed)

    Returns
    -------
    d : (N,) signed distances
    """
    p = np.asarray(p, dtype=float)
    pv = np.asarray(pv, dtype=float)

    # Compute distance to each segment, take minimum
    ds = dsegment(p, pv)      # (N, M-1)
    d = np.min(ds, axis=1)    # (N,)

    # Sign: negative inside, positive outside
    inside = inpoly(p, pv)
    d = ((-1.0) ** inside.astype(float)) * d

    return d
