"""
Set difference of two distance functions.
Copyright (C) 2004-2012 Per-Olof Persson.
"""

import numpy as np


def ddiff(d1, d2):
    """
    Compute set difference of two signed distance fields.

    Returns max(d1, -d2) element-wise, which represents the region
    inside d1 but outside d2.

    Parameters
    ----------
    d1 : array_like, shape (N,)
        Signed distance values for the first region.
    d2 : array_like, shape (N,)
        Signed distance values for the region to subtract.

    Returns
    -------
    d : ndarray, shape (N,)
        Signed distance for the set difference d1 \\ d2.
    """
    d1 = np.asarray(d1, dtype=float).ravel()
    d2 = np.asarray(d2, dtype=float).ravel()
    return np.maximum(d1, -d2)
