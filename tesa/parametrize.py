"""
Parametrization of a polyline based on edge lengths.
Author: David Legland, INRA
"""

import numpy as np


def parametrize(pts, normalize=False):
    """
    Arc-length parametrization of a polyline.

    Parameters
    ----------
    pts       : (N, D) array of vertex coordinates (typically D=2)
    normalize : if True, rescale result so last element is 1.0

    Returns
    -------
    par : (N,) cumulative arc-length from first vertex
    """
    pts = np.asarray(pts, dtype=float)

    if pts.shape[1] == 2:
        # 2D fast path
        par = np.concatenate([[0.0],
                              np.cumsum(np.hypot(np.diff(pts[:, 0]),
                                                 np.diff(pts[:, 1])))])
    else:
        # arbitrary dimension
        par = np.concatenate([[0.0],
                              np.cumsum(np.sqrt(np.sum(np.diff(pts, axis=0)**2,
                                                       axis=1)))])

    if normalize and par[-1] > 0:
        par = par / par[-1]

    return par
