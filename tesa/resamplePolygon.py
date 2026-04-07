"""
Distribute N points equally spaced on a polygon.
Author: David Legland, INRA
"""

import numpy as np
from .resamplePolyline import resamplePolyline


def resamplePolygon(poly, n):
    """
    Resample a closed polygon with N equally-spaced points.

    Closes the polygon, resamples the resulting polyline with n+1
    equally-spaced points, and drops the duplicate closing vertex.

    Parameters
    ----------
    poly : ndarray, shape (M, D)
        Polygon vertex coordinates (not closed; D dimensions).
    n : int
        Number of output vertices.

    Returns
    -------
    poly2 : ndarray, shape (n, D)
        Resampled polygon with equally-spaced vertices.
    """
    poly = np.asarray(poly, dtype=float)
    # Close the polygon, resample with n+1 points, drop the duplicate last
    closed = np.vstack([poly, poly[0:1, :]])
    poly2 = resamplePolyline(closed, n + 1)
    poly2 = poly2[:-1, :]
    return poly2
