"""
Compute the directional Young's modulus for a given stiffness matrix.

For each direction (theta, psi), rotates the stiffness tensor to align
the 1-direction with the specified direction, computes the compliance,
and extracts E = 1/S11.

"""

import numpy as np


def compute_directional_youngs_modulus(C, theta, psi):
    """
    Compute directional Young's modulus.

    Parameters
    ----------
    C : (6, 6) array
        Elastic stiffness matrix (Voigt notation, Pa).
    theta : array-like
        Azimuth angles (radians).
    psi : array-like
        Inclination angles (radians).

    Returns
    -------
    En : (nDirections,) array
        Directional Young's modulus at each (theta, psi) direction (Pa).
    """
    theta = np.asarray(theta, dtype=float).ravel()
    psi = np.asarray(psi, dtype=float).ravel()
    nDir = len(theta)

    # Reshape for 3D batch computation
    th = theta
    ps = psi

    # Compute direction cosine matrices for each point
    a = np.zeros((3, 3, nDir))
    a[0, 0, :] = np.cos(th) * np.cos(ps)
    a[0, 1, :] = np.sin(th) * np.cos(ps)
    a[0, 2, :] = np.sin(ps)
    a[1, 0, :] = np.sin(th)
    a[1, 1, :] = -np.cos(th)
    a[1, 2, :] = 0.0
    a[2, 0, :] = np.cos(th) * np.sin(ps)
    a[2, 1, :] = np.sin(th) * np.sin(ps)
    a[2, 2, :] = -np.cos(ps)

    # Compute Bond matrices M for tensor transformation
    M = np.zeros((6, 6, nDir))
    # Row 1
    M[0, 0, :] = a[0, 0, :] ** 2
    M[0, 1, :] = a[0, 1, :] ** 2
    M[0, 2, :] = a[0, 2, :] ** 2
    M[0, 3, :] = 2.0 * a[0, 1, :] * a[0, 2, :]
    M[0, 4, :] = 2.0 * a[0, 2, :] * a[0, 0, :]
    M[0, 5, :] = 2.0 * a[0, 0, :] * a[0, 1, :]
    # Row 2
    M[1, 0, :] = a[1, 0, :] ** 2
    M[1, 1, :] = a[1, 1, :] ** 2
    M[1, 2, :] = a[1, 2, :] ** 2
    M[1, 3, :] = 2.0 * a[1, 1, :] * a[1, 2, :]
    M[1, 4, :] = 2.0 * a[1, 2, :] * a[1, 0, :]
    M[1, 5, :] = 2.0 * a[1, 0, :] * a[1, 1, :]
    # Row 3
    M[2, 0, :] = a[2, 0, :] ** 2
    M[2, 1, :] = a[2, 1, :] ** 2
    M[2, 2, :] = a[2, 2, :] ** 2
    M[2, 3, :] = 2.0 * a[2, 1, :] * a[2, 2, :]
    M[2, 4, :] = 2.0 * a[2, 2, :] * a[2, 0, :]
    M[2, 5, :] = 2.0 * a[2, 0, :] * a[2, 1, :]
    # Row 4
    M[3, 0, :] = a[1, 0, :] * a[2, 0, :]
    M[3, 1, :] = a[1, 1, :] * a[2, 1, :]
    M[3, 2, :] = a[1, 2, :] * a[2, 2, :]
    M[3, 3, :] = a[1, 1, :] * a[2, 2, :] + a[1, 2, :] * a[2, 1, :]
    M[3, 4, :] = a[1, 0, :] * a[2, 2, :] + a[1, 2, :] * a[2, 0, :]
    M[3, 5, :] = a[1, 1, :] * a[2, 0, :] + a[1, 0, :] * a[2, 1, :]
    # Row 5
    M[4, 0, :] = a[2, 0, :] * a[0, 0, :]
    M[4, 1, :] = a[2, 1, :] * a[0, 1, :]
    M[4, 2, :] = a[2, 2, :] * a[0, 2, :]
    M[4, 3, :] = a[0, 1, :] * a[2, 2, :] + a[0, 2, :] * a[2, 1, :]
    M[4, 4, :] = a[0, 2, :] * a[2, 0, :] + a[0, 0, :] * a[2, 2, :]
    M[4, 5, :] = a[0, 0, :] * a[2, 1, :] + a[0, 1, :] * a[2, 0, :]
    # Row 6
    M[5, 0, :] = a[0, 0, :] * a[1, 0, :]
    M[5, 1, :] = a[0, 1, :] * a[1, 1, :]
    M[5, 2, :] = a[0, 2, :] * a[1, 2, :]
    M[5, 3, :] = a[0, 1, :] * a[1, 2, :] + a[0, 2, :] * a[1, 1, :]
    M[5, 4, :] = a[0, 2, :] * a[1, 0, :] + a[0, 0, :] * a[1, 2, :]
    M[5, 5, :] = a[0, 0, :] * a[1, 1, :] + a[0, 1, :] * a[1, 0, :]

    # Compute transformed stiffness: Cp = M @ C @ M^T
    MC = np.einsum('ijk,jl->ilk', M, C)         # (6, 6, nDir)
    Cp = np.einsum('ijk,ljk->ilk', MC, M)        # (6, 6, nDir)

    # Compute compliance and extract E = 1/S11
    En = np.zeros(nDir)
    for i in range(nDir):
        Sp = np.linalg.solve(Cp[:, :, i], np.eye(6))
        En[i] = 1.0 / Sp[0, 0]

    return En
