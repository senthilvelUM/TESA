"""
Compute adjusted thermal properties at data points.

Rotates thermal expansion alpha and computes stress-temperature moduli beta
at each data point using Euler angles and EBSD correction matrix.

alphaData = NStar @ N @ alpha(phase)
betaData  = C_rotated @ alphaData

where N is the strain Bond matrix, NStar = inv(MStar^T),
and C_rotated = MStar @ M @ C @ M^T @ MStar^T.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np


def compute_data_point_adjusted_thermal_properties(
        data_point_euler_angles,
        data_point_phase,
        phase_elastic_stiffness_matrix,
        phase_thermal_expansion_matrix,
        data_coordinate_system_correction_matrix):
    """
    Compute rotated thermal expansion and stress-temperature moduli.

    Builds the strain Bond matrix N from Euler angles, rotates the phase
    thermal expansion via ``alpha_rot = NStar @ N @ alpha(phase)``, then
    computes the stress-temperature moduli as
    ``beta = C_rotated @ alpha_rot``.

    Parameters
    ----------
    data_point_euler_angles : ndarray, shape (n_data_points, 3)
        Euler angles (phi1, Phi, phi2) in radians.
    data_point_phase : ndarray, shape (n_data_points,), dtype int
        Phase ID for each data point (1-based).
    phase_elastic_stiffness_matrix : list of ndarray
        Stiffness matrix for each phase; each entry has shape (6, 6).
    phase_thermal_expansion_matrix : list of ndarray
        Thermal expansion coefficients for each phase; each entry has
        shape (6, 1) or (6,).
    data_coordinate_system_correction_matrix : ndarray, shape (6, 6)
        EBSD coordinate system correction (Bond) matrix MStar.

    Returns
    -------
    betaData : ndarray, shape (6, 1, n_data_points)
        Stress-temperature moduli ``beta = C_rotated @ alpha_rotated``.
    alphaData : ndarray, shape (6, 1, n_data_points)
        Rotated thermal expansion coefficients.
    """
    # Local declarations
    phi1 = data_point_euler_angles[:, 0]
    PHI  = data_point_euler_angles[:, 1]
    phi2 = data_point_euler_angles[:, 2]
    nDataPoints = len(phi1)
    phase = np.asarray(data_point_phase, dtype=int)
    nPhases = int(np.max(phase))

    # Store stiffness and thermal expansion as 3D arrays
    C = np.zeros((6, 6, nPhases))
    A = np.zeros((6, 1, nPhases))
    for i in range(nPhases):
        C[:, :, i] = phase_elastic_stiffness_matrix[i]
        alpha_i = np.asarray(phase_thermal_expansion_matrix[i]).ravel()
        A[:, 0, i] = alpha_i

    # Compute direction cosine matrices at each data point
    # a1: rotation about z by phi1
    a1 = np.zeros((3, 3, nDataPoints))
    a1[0, 0, :] = np.cos(phi1)
    a1[0, 1, :] = -np.sin(phi1)
    a1[0, 2, :] = 0.0
    a1[1, 0, :] = np.sin(phi1)
    a1[1, 1, :] = np.cos(phi1)
    a1[1, 2, :] = 0.0
    a1[2, 0, :] = 0.0
    a1[2, 1, :] = 0.0
    a1[2, 2, :] = 1.0

    # a2: rotation about x' by Phi
    a2 = np.zeros((3, 3, nDataPoints))
    a2[0, 0, :] = 1.0
    a2[0, 1, :] = 0.0
    a2[0, 2, :] = 0.0
    a2[1, 0, :] = 0.0
    a2[1, 1, :] = np.cos(PHI)
    a2[1, 2, :] = -np.sin(PHI)
    a2[2, 0, :] = 0.0
    a2[2, 1, :] = np.sin(PHI)
    a2[2, 2, :] = np.cos(PHI)

    # a3: rotation about z'' by phi2
    a3 = np.zeros((3, 3, nDataPoints))
    a3[0, 0, :] = np.cos(phi2)
    a3[0, 1, :] = -np.sin(phi2)
    a3[0, 2, :] = 0.0
    a3[1, 0, :] = np.sin(phi2)
    a3[1, 1, :] = np.cos(phi2)
    a3[1, 2, :] = 0.0
    a3[2, 0, :] = 0.0
    a3[2, 1, :] = 0.0
    a3[2, 2, :] = 1.0

    # a = a1 * a2 * a3
    a = np.einsum('ijk,jlk->ilk', a1, a2)
    a = np.einsum('ijk,jlk->ilk', a, a3)

    # Build Bond matrices M and N
    # Upper left: Mul = a.^2
    Mul = a ** 2

    # Upper right: Mur
    Mur = 2.0 * np.stack([
        a[:, 1, :] * a[:, 2, :],
        a[:, 2, :] * a[:, 0, :],
        a[:, 0, :] * a[:, 1, :],
    ], axis=1)

    # Lower left: Mll
    Mll = np.stack([
        a[1, :, :] * a[2, :, :],
        a[2, :, :] * a[0, :, :],
        a[0, :, :] * a[1, :, :],
    ], axis=0)

    # Lower right: Mlr
    Mlr = np.zeros((3, 3, nDataPoints))
    Mlr[0, 0, :] = a[1, 1, :] * a[2, 2, :] + a[1, 2, :] * a[2, 1, :]
    Mlr[0, 1, :] = a[1, 0, :] * a[2, 2, :] + a[1, 2, :] * a[2, 0, :]
    Mlr[0, 2, :] = a[1, 1, :] * a[2, 0, :] + a[1, 0, :] * a[2, 1, :]
    Mlr[1, 0, :] = a[0, 1, :] * a[2, 2, :] + a[0, 2, :] * a[2, 1, :]
    Mlr[1, 1, :] = a[0, 2, :] * a[2, 0, :] + a[0, 0, :] * a[2, 2, :]
    Mlr[1, 2, :] = a[0, 0, :] * a[2, 1, :] + a[0, 1, :] * a[2, 0, :]
    Mlr[2, 0, :] = a[0, 1, :] * a[1, 2, :] + a[0, 2, :] * a[1, 1, :]
    Mlr[2, 1, :] = a[0, 2, :] * a[1, 0, :] + a[0, 0, :] * a[1, 2, :]
    Mlr[2, 2, :] = a[0, 0, :] * a[1, 1, :] + a[0, 1, :] * a[1, 0, :]

    # M = [Mul Mur ; Mll Mlr]  (stress Bond matrix)
    M = np.zeros((6, 6, nDataPoints))
    M[0:3, 0:3, :] = Mul
    M[0:3, 3:6, :] = Mur
    M[3:6, 0:3, :] = Mll
    M[3:6, 3:6, :] = Mlr

    # N = [Mul 0.5*Mur ; 2*Mll Mlr]  (strain Bond matrix)
    N = np.zeros((6, 6, nDataPoints))
    N[0:3, 0:3, :] = Mul
    N[0:3, 3:6, :] = 0.5 * Mur
    N[3:6, 0:3, :] = 2.0 * Mll
    N[3:6, 3:6, :] = Mlr

    MStar = data_coordinate_system_correction_matrix
    # NStar = inv(MStar^T)
    NStar = np.linalg.solve(MStar.T, np.eye(6))

    # Transform thermal expansion: alphaData = NStar @ N @ alpha(phase)
    phase_idx = (phase - 1).astype(int)
    A_dp = A[:, :, phase_idx]  # (6, 1, nDataPoints)

    # N @ alpha  (batch: (6,6,N) @ (6,1,N) -> (6,1,N))
    NA = np.einsum('ijk,jlk->ilk', N, A_dp)
    # NStar @ (N @ alpha)
    alphaData = np.einsum('ij,jlk->ilk', NStar, NA)  # (6, 1, nDataPoints)

    # Compute rotated stiffness: Ci = MStar @ M @ C @ M^T @ MStar^T
    C_dp = C[:, :, phase_idx]  # (6, 6, nDataPoints)
    MC = np.einsum('ijk,jlk->ilk', M, C_dp)           # M @ C
    MCMt = np.einsum('ijk,ljk->ilk', MC, M)            # (M @ C) @ M^T
    tmp = np.einsum('ij,jkn->ikn', MStar, MCMt)        # MStar @ MCMt
    Ci = np.einsum('ijn,kj->ikn', tmp, MStar)           # tmp @ MStar^T

    # betaData = Ci @ alphaData
    betaData = np.einsum('ijk,jlk->ilk', Ci, alphaData)  # (6, 1, nDataPoints)

    return betaData, alphaData
