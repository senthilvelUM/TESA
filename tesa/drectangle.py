"""
Signed distance function for a rectangle.
Copyright (C) 2004-2012 Per-Olof Persson.
"""

import numpy as np


def drectangle(p, x1, x2, y1, y2):
    """
    Compute signed distance from points to an axis-aligned rectangle.

    Negative inside, positive outside.

    Parameters
    ----------
    p : array_like, shape (N, 2)
        Query points.
    x1 : float
        Left boundary of the rectangle.
    x2 : float
        Right boundary of the rectangle.
    y1 : float
        Bottom boundary of the rectangle.
    y2 : float
        Top boundary of the rectangle.

    Returns
    -------
    d : ndarray, shape (N,)
        Signed distance (negative inside, positive outside).
    """
    p = np.asarray(p, dtype=float)
    return -np.minimum(
        np.minimum(np.minimum(-y1 + p[:, 1], y2 - p[:, 1]),
                   -x1 + p[:, 0]),
        x2 - p[:, 0]
    )
