"""
Lambert azimuthal equal-area projection and its inverse.

Projects points on the unit sphere to 2D and back. Used for
crystallographic orientation visualization (pole figures).

"""

import numpy as np


def lambert_azimuthal_projection(x, y, z):
    """
    Forward Lambert azimuthal equal-area projection.

    Maps points on the unit sphere (x, y, z) to 2D (X, Y).

    Parameters
    ----------
    x, y, z : array-like
        Cartesian coordinates on the unit sphere.

    Returns
    -------
    X : ndarray
        Projected x-coordinates.
    Y : ndarray
        Projected y-coordinates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    # Lambert projection: scale = sqrt(2 / (1 - z))
    # Handle z = 1 (north pole singularity) by clamping denominator
    denom = np.maximum(1.0 - z, 1e-30)
    scale = np.sqrt(2.0 / denom)
    X = scale * x
    Y = scale * y

    # Set exact zero at the pole (z = 1)
    pole = (z >= 1.0)
    X[pole] = 0.0
    Y[pole] = 0.0

    return X, Y


def inverse_lambert_azimuthal_projection(X, Y):
    """
    Inverse Lambert azimuthal equal-area projection.

    Maps 2D projected coordinates (X, Y) back to the unit sphere (x, y, z).

    Parameters
    ----------
    X, Y : array-like
        Projected 2D coordinates.

    Returns
    -------
    x : ndarray
        Cartesian x-coordinate on the unit sphere.
    y : ndarray
        Cartesian y-coordinate on the unit sphere.
    z : ndarray
        Cartesian z-coordinate on the unit sphere.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    r2 = X ** 2 + Y ** 2
    scale = np.sqrt(np.maximum(1.0 - r2 / 4.0, 0.0))
    x = scale * X
    y = scale * Y
    z = -1.0 + r2 / 2.0

    return x, y, z
