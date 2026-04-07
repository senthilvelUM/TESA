"""
Calculate normals of 2D line points using weighted central differences.
Author: D. Kroon, University of Twente (August 2011)
"""

import numpy as np


def LineNormals2D(Vertices, Lines=None):
    """
    Compute normals at each vertex of a 2D polyline.

    Uses weighted central differences of tangent vectors, where the
    weight is inversely proportional to the squared segment length.

    Parameters
    ----------
    Vertices : ndarray, shape (M, 2)
        Vertex coordinates of the polyline.
    Lines : ndarray, shape (N, 2) or None, optional
        Line segment connectivity as pairs of 0-based vertex indices.
        If None, assumes sequential connectivity [0-1, 1-2, ..., (M-2)-(M-1)].

    Returns
    -------
    N : ndarray, shape (M, 2)
        Unit normal vector at each vertex.
    """
    Vertices = np.asarray(Vertices, dtype=float)
    M = Vertices.shape[0]

    if Lines is None:
        Lines = np.column_stack([np.arange(M - 1), np.arange(1, M)])
    else:
        Lines = np.asarray(Lines, dtype=int)

    # Calculate tangent vectors
    DT = Vertices[Lines[:, 0], :] - Vertices[Lines[:, 1], :]

    # Make influence of tangent vector 1/Distance (weighted central differences)
    LL = np.sqrt(DT[:, 0]**2 + DT[:, 1]**2)
    DT[:, 0] = DT[:, 0] / np.maximum(LL**2, np.finfo(float).eps)
    DT[:, 1] = DT[:, 1] / np.maximum(LL**2, np.finfo(float).eps)

    D1 = np.zeros((M, 2))
    D1[Lines[:, 0], :] = DT
    D2 = np.zeros((M, 2))
    D2[Lines[:, 1], :] = DT
    D = D1 + D2

    # Normalize the normal
    LL2 = np.sqrt(D[:, 0]**2 + D[:, 1]**2)
    LL2 = np.maximum(LL2, np.finfo(float).eps)  # avoid division by zero
    N = np.column_stack([-D[:, 1] / LL2, D[:, 0] / LL2])

    return N
