"""
Mirror hemisphere data to create a full sphere.

Takes surface data and coordinates from the lower half of a unit sphere
(as created by sphere(n) and truncated) and mirrors them across z=0,
rotating the upper half by 180 degrees.

"""

import numpy as np


def mirrorsphere(x, y, z, d):
    """
    Mirror hemisphere coordinates and data to a full sphere.

    Parameters
    ----------
    x, y, z : (M, N) arrays
        Hemisphere coordinates in matrix form (from sphere(n), truncated).
    d : (M, N) array
        Data values on the hemisphere (e.g., wave speeds).

    Returns
    -------
    xn, yn, zn : (2*M, N) arrays
        Full sphere coordinates.
    dn : (2*M, N) array
        Full sphere data values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    d = np.asarray(d, dtype=float)

    # Mirror coordinates: flip and negate x, y; flip and negate z
    xn = np.vstack([x, -np.flip(x, axis=0)])
    yn = np.vstack([y, -np.flip(y, axis=0)])
    zn = np.vstack([z, -1.0 * np.flip(z, axis=0)])

    # Mirror data: flip vertically
    dn = np.vstack([d, np.flipud(d)])

    return xn, yn, zn, dn
