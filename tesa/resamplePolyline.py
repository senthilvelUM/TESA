"""
Distribute N points equally spaced on a polyline.
Author: David Legland, INRA
"""

import numpy as np
from .parametrize import parametrize


def resamplePolyline(poly, n):
    """
    Resample a polyline with N equally-spaced points.

    Distributes n points at uniform arc-length intervals along the
    polyline using linear interpolation between original vertices.

    Parameters
    ----------
    poly : ndarray, shape (M, D)
        Original polyline vertex coordinates (D dimensions).
    n : int
        Number of output points.

    Returns
    -------
    poly2 : ndarray, shape (n, D)
        Resampled polyline with equally-spaced vertices.
    """
    poly = np.asarray(poly, dtype=float)
    dim = poly.shape[1]

    # Parametrization of the curve
    s = parametrize(poly)

    # Distribute N points equally spaced
    Lmax = s[-1]
    pos = np.linspace(0, Lmax, n)

    poly2 = np.zeros((n, dim))
    for i in range(n):
        # Index of surrounding vertices before and after
        ind0 = np.searchsorted(s, pos[i], side='right') - 1
        ind0 = max(ind0, 0)
        ind1_arr = np.where(s >= pos[i])[0]
        ind1 = ind1_arr[0] if len(ind1_arr) > 0 else len(s) - 1

        if ind0 == ind1:
            poly2[i, :] = poly[ind0, :]
            continue

        # Positions of surrounding vertices
        pt0 = poly[ind0, :]
        pt1 = poly[ind1, :]

        # Weights associated to each neighbor
        l0 = pos[i] - s[ind0]
        l1 = s[ind1] - pos[i]

        # Linear interpolation (handle zero-length segments from duplicate points)
        denom = l0 + l1
        if denom == 0:
            poly2[i, :] = pt0
        else:
            poly2[i, :] = (pt0 * l1 + pt1 * l0) / denom

    return poly2
