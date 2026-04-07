"""
Simplex volume (area for 2D, volume for 3D).
Copyright (C) 2004-2012 Per-Olof Persson.
"""

import numpy as np
from math import factorial


def simpvol(p, t):
    """
    Signed simplex volume/area.

    Parameters
    ----------
    p : (N, dim) node coordinates
    t : (M, dim+1) simplex connectivity (0-based)

    Returns
    -------
    v : (M,) signed volumes/areas
    """
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=int)
    dim = p.shape[1]

    if dim == 1:
        d12 = p[t[:, 1], :] - p[t[:, 0], :]
        v = d12.ravel()
    elif dim == 2:
        d12 = p[t[:, 1], :] - p[t[:, 0], :]
        d13 = p[t[:, 2], :] - p[t[:, 0], :]
        v = (d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0]) / 2.0
    elif dim == 3:
        d12 = p[t[:, 1], :] - p[t[:, 0], :]
        d13 = p[t[:, 2], :] - p[t[:, 0], :]
        d14 = p[t[:, 3], :] - p[t[:, 0], :]
        # dot(cross(d12, d13), d14) / 6
        cr = np.cross(d12, d13)
        v = np.sum(cr * d14, axis=1) / 6.0
    else:
        nelem = t.shape[0]
        v = np.zeros(nelem)
        for ii in range(nelem):
            A = np.ones((dim + 1, dim + 1))
            for jj in range(dim + 1):
                A[jj, 1:] = p[t[ii, jj], :]
            v[ii] = np.linalg.det(A)
        v = v / factorial(dim)

    return v
