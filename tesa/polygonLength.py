"""
Perimeter of a polygon.
Author: David Legland, INRA
"""

import numpy as np


def polygonLength(poly):
    """
    Compute the perimeter of a closed polygon.

    Automatically closes the polygon by connecting the last vertex
    to the first.

    Parameters
    ----------
    poly : ndarray, shape (N, 2) or list of ndarray
        Polygon vertex coordinates. If a list, computes the total
        perimeter of all polygons.

    Returns
    -------
    length : float
        Total perimeter length.
    """
    # Multi-polygon case
    if isinstance(poly, list):
        return sum(polygonLength(p) for p in poly)

    poly = np.asarray(poly, dtype=float)

    if poly.shape[0] < 2:
        return 0.0

    # Close the polygon: append first vertex
    closed = np.vstack([poly, poly[0:1, :]])
    dp = np.diff(closed, axis=0)

    if poly.shape[1] == 2:
        return float(np.sum(np.hypot(dp[:, 0], dp[:, 1])))
    else:
        return float(np.sum(np.sqrt(np.sum(dp**2, axis=1))))
