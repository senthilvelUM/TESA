"""
Simplex quality metric.
Copyright (C) 2004-2012 Per-Olof Persson.
"""

import numpy as np
from .simpvol import simpvol


def simpqual(p, t, type=1):
    """
    Simplex quality.

    Parameters
    ----------
    p    : (N, dim) node coordinates
    t    : (M, dim+1) simplex connectivity (0-based)
    type : 1 = radius ratio (default), 2 = approximate formula

    Returns
    -------
    q : (M,) quality values in [0, 1], 1 = equilateral
    """
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=int)
    dim = p.shape[1]

    if type == 1:
        # RADIUS RATIO
        if dim == 1:
            q = np.ones(t.shape[0])
        elif dim == 2:
            a = np.sqrt(np.sum((p[t[:, 1], :] - p[t[:, 0], :])**2, axis=1))
            b = np.sqrt(np.sum((p[t[:, 2], :] - p[t[:, 0], :])**2, axis=1))
            c = np.sqrt(np.sum((p[t[:, 2], :] - p[t[:, 1], :])**2, axis=1))
            r = 0.5 * np.sqrt(np.clip((b + c - a) * (c + a - b) * (a + b - c)
                                      / (a + b + c), 0, None))
            R = a * b * c / np.sqrt(np.clip(
                (a + b + c) * (b + c - a) * (c + a - b) * (a + b - c),
                1e-30, None))
            q = 2.0 * r / R
        elif dim == 3:
            d12 = p[t[:, 1], :] - p[t[:, 0], :]
            d13 = p[t[:, 2], :] - p[t[:, 0], :]
            d14 = p[t[:, 3], :] - p[t[:, 0], :]
            d23 = p[t[:, 2], :] - p[t[:, 1], :]
            d24 = p[t[:, 3], :] - p[t[:, 1], :]
            d34 = p[t[:, 3], :] - p[t[:, 2], :]
            v = np.abs(np.sum(np.cross(d12, d13) * d14, axis=1)) / 6.0
            s1 = np.sqrt(np.sum(np.cross(d12, d13)**2, axis=1)) / 2.0
            s2 = np.sqrt(np.sum(np.cross(d12, d14)**2, axis=1)) / 2.0
            s3 = np.sqrt(np.sum(np.cross(d13, d14)**2, axis=1)) / 2.0
            s4 = np.sqrt(np.sum(np.cross(d23, d24)**2, axis=1)) / 2.0
            p1 = np.sqrt(np.sum(d12**2, axis=1)) * np.sqrt(np.sum(d34**2, axis=1))
            p2 = np.sqrt(np.sum(d23**2, axis=1)) * np.sqrt(np.sum(d14**2, axis=1))
            p3 = np.sqrt(np.sum(d13**2, axis=1)) * np.sqrt(np.sum(d24**2, axis=1))
            q = (216.0 * v**2 / (s1 + s2 + s3 + s4) /
                 np.sqrt(np.clip((p1 + p2 + p3) * (p1 + p2 - p3) *
                                 (p1 + p3 - p2) * (p2 + p3 - p1), 1e-30, None)))
        else:
            raise NotImplementedError("Dimension not implemented.")

    elif type == 2:
        # APPROXIMATE FORMULA
        if dim == 1:
            q = np.ones(t.shape[0])
        elif dim == 2:
            d12 = np.sum((p[t[:, 1], :] - p[t[:, 0], :])**2, axis=1)
            d13 = np.sum((p[t[:, 2], :] - p[t[:, 0], :])**2, axis=1)
            d23 = np.sum((p[t[:, 2], :] - p[t[:, 1], :])**2, axis=1)
            q = 4.0 * np.sqrt(3.0) * np.abs(simpvol(p, t)) / (d12 + d13 + d23)
        elif dim == 3:
            d12 = np.sum((p[t[:, 1], :] - p[t[:, 0], :])**2, axis=1)
            d13 = np.sum((p[t[:, 2], :] - p[t[:, 0], :])**2, axis=1)
            d14 = np.sum((p[t[:, 3], :] - p[t[:, 0], :])**2, axis=1)
            d23 = np.sum((p[t[:, 2], :] - p[t[:, 1], :])**2, axis=1)
            d24 = np.sum((p[t[:, 3], :] - p[t[:, 1], :])**2, axis=1)
            d34 = np.sum((p[t[:, 3], :] - p[t[:, 2], :])**2, axis=1)
            q = (216.0 * np.abs(simpvol(p, t)) / np.sqrt(3.0) /
                 (d12 + d13 + d14 + d23 + d24 + d34)**(3.0 / 2.0))
        else:
            raise NotImplementedError("Dimension not implemented.")
    else:
        raise ValueError("Incorrect type.")

    return q
