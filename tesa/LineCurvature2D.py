"""
Calculate curvature of a 2D line by fitting quadratic polynomials.
Author: D. Kroon, University of Twente (August 2011)
"""

import numpy as np


def _inverse3(M):
    """
    Compute the inverse of an array of 3x3 matrices.

    Uses the adjugate-determinant formula for efficient batched inversion.

    Parameters
    ----------
    M : ndarray, shape (K, 9)
        Each row is a flattened 3x3 matrix in row-major order.

    Returns
    -------
    Minv : ndarray, shape (K, 3, 3)
        Inverse of each 3x3 matrix.
    """
    # Adjugate matrix elements (cofactors transposed)
    adjM = np.zeros((M.shape[0], 3, 3))
    adjM[:, 0, 0] =   M[:, 4] * M[:, 8] - M[:, 7] * M[:, 5]
    adjM[:, 0, 1] = -(M[:, 3] * M[:, 8] - M[:, 6] * M[:, 5])
    adjM[:, 0, 2] =   M[:, 3] * M[:, 7] - M[:, 6] * M[:, 4]
    adjM[:, 1, 0] = -(M[:, 1] * M[:, 8] - M[:, 7] * M[:, 2])
    adjM[:, 1, 1] =   M[:, 0] * M[:, 8] - M[:, 6] * M[:, 2]
    adjM[:, 1, 2] = -(M[:, 0] * M[:, 7] - M[:, 6] * M[:, 1])
    adjM[:, 2, 0] =   M[:, 1] * M[:, 5] - M[:, 4] * M[:, 2]
    adjM[:, 2, 1] = -(M[:, 0] * M[:, 5] - M[:, 3] * M[:, 2])
    adjM[:, 2, 2] =   M[:, 0] * M[:, 4] - M[:, 3] * M[:, 1]

    detM = (M[:, 0] * M[:, 4] * M[:, 8] - M[:, 0] * M[:, 7] * M[:, 5]
          - M[:, 3] * M[:, 1] * M[:, 8] + M[:, 3] * M[:, 7] * M[:, 2]
          + M[:, 6] * M[:, 1] * M[:, 5] - M[:, 6] * M[:, 4] * M[:, 2])

    # Broadcast divide: adjM / detM
    Minv = adjM / detM[:, np.newaxis, np.newaxis]
    return Minv


def LineCurvature2D(Vertices, Lines=None):
    """
    Compute curvature at each vertex of a 2D polyline.

    Fits a local quadratic polynomial through each vertex and its two
    neighbors, then evaluates the curvature of the parametric curve at
    the vertex. Positive curvature indicates the curve bends to the left.

    Parameters
    ----------
    Vertices : ndarray, shape (M, 2)
        Vertex coordinates of the polyline.
    Lines : ndarray, shape (N, 2) or None, optional
        Line segment connectivity as pairs of 0-based vertex indices.
        If None, assumes sequential connectivity [0-1, 1-2, ..., (M-2)-(M-1)].

    Returns
    -------
    k : ndarray, shape (M,)
        Curvature value at each vertex.
    """
    Vertices = np.asarray(Vertices, dtype=float)
    nv = Vertices.shape[0]

    if Lines is None:
        Lines = np.column_stack([np.arange(nv - 1), np.arange(1, nv)])
    else:
        Lines = np.asarray(Lines, dtype=int)

    # Get left and right neighbor of each point
    # Use -1 as sentinel for "no neighbor" (MATLAB uses 0 because indices are 1-based;
    # in Python index 0 is valid, so we must use a different sentinel)
    Na = -np.ones(nv, dtype=int)
    Nb = -np.ones(nv, dtype=int)
    Na[Lines[:, 0]] = Lines[:, 1]
    Nb[Lines[:, 1]] = Lines[:, 0]

    # Check for end-of-line points without a left or right neighbor
    checkNa = (Na == -1)
    checkNb = (Nb == -1)
    Naa = Na.copy()
    Nbb = Nb.copy()
    Naa[checkNa] = np.where(checkNa)[0]
    Nbb[checkNb] = np.where(checkNb)[0]

    # If no left neighbor, use two right neighbors, and vice versa
    Na[checkNa] = Nbb[Nbb[checkNa]]
    Nb[checkNb] = Naa[Naa[checkNb]]

    # Correct for sampling differences
    Ta = -np.sqrt(np.sum((Vertices - Vertices[Na, :])**2, axis=1))
    Tb =  np.sqrt(np.sum((Vertices - Vertices[Nb, :])**2, axis=1))

    # If no left neighbor, flip sign; same for right
    Ta[checkNa] = -Ta[checkNa]
    Tb[checkNb] = -Tb[checkNb]

    # Fit quadratic polynomials: x = a3*t^2 + a2*t + a1, y = b3*t^2 + b2*t + b1
    # t=0 at vertex, t=Ta at left neighbor, t=Tb at right neighbor
    x = np.column_stack([Vertices[Na, 0], Vertices[:, 0], Vertices[Nb, 0]])
    y = np.column_stack([Vertices[Na, 1], Vertices[:, 1], Vertices[Nb, 1]])

    ones = np.ones_like(Tb)
    zeros = np.zeros_like(Tb)
    # M is (nv, 9): flattened 3x3 matrix per vertex
    # Row-major: [1, -Ta, Ta^2, 1, 0, 0, 1, -Tb, Tb^2]
    # MATLAB: [ones -Ta Ta.^2  ones zeros zeros  ones -Tb Tb.^2]
    # Note MATLAB stores column-major, but the matrix structure is:
    #   [ 1   -Ta  Ta^2 ]      row for left neighbor (t=Ta, but note sign)
    #   [ 1    0    0   ]      row for vertex itself (t=0)
    #   [ 1   -Tb  Tb^2 ]      row for right neighbor (t=Tb, but note sign)
    # Wait — re-reading MATLAB code: the M matrix rows are interleaved differently.
    # M = [ones(size(Tb)) -Ta Ta.^2 ones(size(Tb)) zeros(size(Tb)) zeros(size(Tb)) ones(size(Tb)) -Tb Tb.^2];
    # This is 9 columns: the 3x3 matrix stored row-by-row:
    #   col 1-3: row 1 = [1, -Ta, Ta^2]
    #   col 4-6: row 2 = [1,  0,   0  ]
    #   col 7-9: row 3 = [1, -Tb, Tb^2]
    M = np.column_stack([ones, -Ta, Ta**2,
                         ones, zeros, zeros,
                         ones, -Tb, Tb**2])

    invM = _inverse3(M)

    # a coefficients (for x)
    a = np.zeros((nv, 3))
    a[:, 0] = invM[:, 0, 0] * x[:, 0] + invM[:, 1, 0] * x[:, 1] + invM[:, 2, 0] * x[:, 2]
    a[:, 1] = invM[:, 0, 1] * x[:, 0] + invM[:, 1, 1] * x[:, 1] + invM[:, 2, 1] * x[:, 2]
    a[:, 2] = invM[:, 0, 2] * x[:, 0] + invM[:, 1, 2] * x[:, 1] + invM[:, 2, 2] * x[:, 2]

    # b coefficients (for y)
    b = np.zeros((nv, 3))
    b[:, 0] = invM[:, 0, 0] * y[:, 0] + invM[:, 1, 0] * y[:, 1] + invM[:, 2, 0] * y[:, 2]
    b[:, 1] = invM[:, 0, 1] * y[:, 0] + invM[:, 1, 1] * y[:, 1] + invM[:, 2, 1] * y[:, 2]
    b[:, 2] = invM[:, 0, 2] * y[:, 0] + invM[:, 1, 2] * y[:, 1] + invM[:, 2, 2] * y[:, 2]

    # Curvature from fitted polynomial
    k = 2.0 * (a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]) / \
        (a[:, 1]**2 + b[:, 1]**2)**(3.0 / 2.0)

    return k
