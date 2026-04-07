"""
Compute the directional thermal expansion coefficient.

For each direction theta, rotates the thermal expansion vector using
the strain Bond matrix N, then extracts the alpha_11 component in
the rotated frame.

Note: this is a 2D (in-plane) rotation only — theta is the in-plane
azimuth angle. The rotation is about the z-axis.

"""

import numpy as np


def compute_directional_thermal_expansion_coefficient(alpha, theta):
    """
    Compute directional thermal expansion coefficient.

    Parameters
    ----------
    alpha : (6, 1) or (6,) array
        Thermal expansion coefficients in Voigt notation.
    theta : array-like
        Azimuth angles (radians) for in-plane rotation.

    Returns
    -------
    alphan : (nDirections,) array
        Directional thermal expansion coefficient at each theta.
    """
    alpha = np.asarray(alpha, dtype=float).reshape(6, 1)
    theta = np.asarray(theta, dtype=float).ravel()
    nDir = len(theta)

    # Compute direction cosine matrices for each point (rotation about z)
    a = np.zeros((3, 3, nDir))
    a[0, 0, :] = np.cos(theta)
    a[0, 1, :] = np.sin(theta)
    a[0, 2, :] = 0.0
    a[1, 0, :] = -np.sin(theta)
    a[1, 1, :] = np.cos(theta)
    a[1, 2, :] = 0.0
    a[2, 0, :] = 0.0
    a[2, 1, :] = 0.0
    a[2, 2, :] = 1.0

    # Compute strain Bond matrix N
    Mul = a ** 2
    Mur = 2.0 * np.stack([
        a[:, 1, :] * a[:, 2, :],
        a[:, 2, :] * a[:, 0, :],
        a[:, 0, :] * a[:, 1, :],
    ], axis=1)
    Mll = np.stack([
        a[1, :, :] * a[2, :, :],
        a[2, :, :] * a[0, :, :],
        a[0, :, :] * a[1, :, :],
    ], axis=0)
    Mlr = np.zeros((3, 3, nDir))
    Mlr[0, 0, :] = a[1, 1, :] * a[2, 2, :] + a[1, 2, :] * a[2, 1, :]
    Mlr[0, 1, :] = a[1, 0, :] * a[2, 2, :] + a[1, 2, :] * a[2, 0, :]
    Mlr[0, 2, :] = a[1, 1, :] * a[2, 0, :] + a[1, 0, :] * a[2, 1, :]
    Mlr[1, 0, :] = a[0, 1, :] * a[2, 2, :] + a[0, 2, :] * a[2, 1, :]
    Mlr[1, 1, :] = a[0, 2, :] * a[2, 0, :] + a[0, 0, :] * a[2, 2, :]
    Mlr[1, 2, :] = a[0, 0, :] * a[2, 1, :] + a[0, 1, :] * a[2, 0, :]
    Mlr[2, 0, :] = a[0, 1, :] * a[1, 2, :] + a[0, 2, :] * a[1, 1, :]
    Mlr[2, 1, :] = a[0, 2, :] * a[1, 0, :] + a[0, 0, :] * a[1, 2, :]
    Mlr[2, 2, :] = a[0, 0, :] * a[1, 1, :] + a[0, 1, :] * a[1, 0, :]

    # N = [Mul 0.5*Mur; 2*Mll Mlr]
    N = np.zeros((6, 6, nDir))
    N[0:3, 0:3, :] = Mul
    N[0:3, 3:6, :] = 0.5 * Mur
    N[3:6, 0:3, :] = 2.0 * Mll
    N[3:6, 3:6, :] = Mlr

    # Compute transformed thermal expansion: alphap = N @ alpha
    alphap = np.einsum('ijk,jl->ilk', N, alpha)  # (6, 1, nDir)

    # Return the alpha_11 component (first element)
    alphan = alphap[0, 0, :].ravel()

    return alphan
