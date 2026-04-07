"""
Length of a polyline given as a list of points.
Author: David Legland, INRA
"""

import numpy as np


def polylineLength(poly, closed=False):
    """
    Compute the total arc length of a polyline.

    Parameters
    ----------
    poly : ndarray, shape (N, D)
        Vertex coordinates of the polyline (D dimensions).
    closed : bool, optional
        If True, include the closing segment from the last vertex
        back to the first. Default False.

    Returns
    -------
    length : float
        Total arc length of the polyline.
    """
    poly = np.asarray(poly, dtype=float)

    if poly.shape[0] < 2:
        return 0.0

    if closed:
        # Close the polygon: append first vertex
        pts = np.vstack([poly, poly[0:1, :]])
    else:
        pts = poly

    dp = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt(np.sum(dp**2, axis=1))))
