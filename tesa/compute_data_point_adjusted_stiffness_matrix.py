"""
Compute adjusted stiffness matrices at data points.

Rotates the phase stiffness matrices by the Euler angles at each data point,
then applies the EBSD coordinate system correction matrix.

DData = MStar @ M @ C @ M^T @ MStar^T

where M is the 6x6 stress Bond matrix built from the Euler angles,
MStar is the EBSD correction matrix, and C is the phase stiffness.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np


def compute_data_point_adjusted_stiffness_matrix(
        data_point_euler_angles,
        data_point_phase,
        phase_elastic_stiffness_matrix,
        data_coordinate_system_correction_matrix):
    """
    Compute rotated stiffness matrix at each data point.

    Builds the 6x6 stress Bond matrix M from the Bunge Euler angles
    (phi1, Phi, phi2), then computes
    ``DData = MStar @ M @ C(phase) @ M^T @ MStar^T``
    where MStar is the EBSD coordinate system correction matrix and C is
    the phase stiffness.

    Parameters
    ----------
    data_point_euler_angles : ndarray, shape (n_data_points, 3)
        Euler angles (phi1, Phi, phi2) in radians at each data point.
    data_point_phase : ndarray, shape (n_data_points,), dtype int
        Phase ID for each data point (1-based).
    phase_elastic_stiffness_matrix : list of ndarray
        Stiffness matrix for each phase; each entry has shape (6, 6).
        Phase *i* is at index ``i - 1``.
    data_coordinate_system_correction_matrix : ndarray, shape (6, 6)
        EBSD coordinate system correction (Bond) matrix MStar.

    Returns
    -------
    DData : ndarray, shape (6, 6, n_data_points)
        Rotated stiffness matrix at each data point.
    dataPointStressBondMatrix : ndarray, shape (6, 6, n_data_points)
        Stress Bond matrix M at each data point (before applying MStar
        and stiffness rotation). Retained for downstream compliance
        computation.
    """
    # Local declarations
    phi1 = data_point_euler_angles[:, 0]
    PHI  = data_point_euler_angles[:, 1]
    phi2 = data_point_euler_angles[:, 2]
    nDataPoints = len(phi1)

    # Compute the direction cosine matrices at each data point
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

    # a = a1 * a2 * a3  (batch 3x3 multiply)
    a = np.einsum('ijk,jlk->ilk', a1, a2)   # a = a1 @ a2
    a = np.einsum('ijk,jlk->ilk', a, a3)     # a = a @ a3

    # Build 6x6 stress Bond matrices M = [Mul Mur; Mll Mlr]
    # Upper left: Mul = a.^2 (element-wise square)
    Mul = a ** 2  # (3, 3, nDataPoints)

    # Upper right: Mur = 2*[a(:,2,:).*a(:,3,:)  a(:,3,:).*a(:,1,:)  a(:,1,:).*a(:,2,:)]
    Mur = 2.0 * np.stack([
        a[:, 1, :] * a[:, 2, :],
        a[:, 2, :] * a[:, 0, :],
        a[:, 0, :] * a[:, 1, :],
    ], axis=1)  # (3, 3, nDataPoints)

    # Lower left: Mll = [a(2,:,:).*a(3,:,:); a(3,:,:).*a(1,:,:); a(1,:,:).*a(2,:,:)]
    Mll = np.stack([
        a[1, :, :] * a[2, :, :],
        a[2, :, :] * a[0, :, :],
        a[0, :, :] * a[1, :, :],
    ], axis=0)  # (3, 3, nDataPoints)

    # Lower right: Mlr (3x3xN)
    Mlr = np.zeros((3, 3, nDataPoints))
    # Row 1: [a(2,2)*a(3,3)+a(2,3)*a(3,2), a(2,1)*a(3,3)+a(2,3)*a(3,1), a(2,2)*a(3,1)+a(2,1)*a(3,2)]
    Mlr[0, 0, :] = a[1, 1, :] * a[2, 2, :] + a[1, 2, :] * a[2, 1, :]
    Mlr[0, 1, :] = a[1, 0, :] * a[2, 2, :] + a[1, 2, :] * a[2, 0, :]
    Mlr[0, 2, :] = a[1, 1, :] * a[2, 0, :] + a[1, 0, :] * a[2, 1, :]
    # Row 2: [a(1,2)*a(3,3)+a(1,3)*a(3,2), a(1,3)*a(3,1)+a(1,1)*a(3,3), a(1,1)*a(3,2)+a(1,2)*a(3,1)]
    Mlr[1, 0, :] = a[0, 1, :] * a[2, 2, :] + a[0, 2, :] * a[2, 1, :]
    Mlr[1, 1, :] = a[0, 2, :] * a[2, 0, :] + a[0, 0, :] * a[2, 2, :]
    Mlr[1, 2, :] = a[0, 0, :] * a[2, 1, :] + a[0, 1, :] * a[2, 0, :]
    # Row 3: [a(1,2)*a(2,3)+a(1,3)*a(2,2), a(1,3)*a(2,1)+a(1,1)*a(2,3), a(1,1)*a(2,2)+a(1,2)*a(2,1)]
    Mlr[2, 0, :] = a[0, 1, :] * a[1, 2, :] + a[0, 2, :] * a[1, 1, :]
    Mlr[2, 1, :] = a[0, 2, :] * a[1, 0, :] + a[0, 0, :] * a[1, 2, :]
    Mlr[2, 2, :] = a[0, 0, :] * a[1, 1, :] + a[0, 1, :] * a[1, 0, :]

    # Assemble DData = [Mul Mur ; Mll Mlr]  — this is M (Bond matrix) at each data point
    DData = np.zeros((6, 6, nDataPoints))
    DData[0:3, 0:3, :] = Mul
    DData[0:3, 3:6, :] = Mur
    DData[3:6, 0:3, :] = Mll
    DData[3:6, 3:6, :] = Mlr

    # Save the Bond matrices before applying correction and stiffness
    dataPointStressBondMatrix = DData.copy()

    # Build phase stiffness 3D array C(:,:,phase)
    max_phase = int(np.max(data_point_phase))
    C = np.zeros((6, 6, max_phase))
    for iPhase in range(1, max_phase + 1):
        C[:, :, iPhase - 1] = phase_elastic_stiffness_matrix[iPhase - 1]

    # Compute adjusted stiffness:
    # DData = MStar @ M @ C(:,:,phase) @ M^T @ MStar^T
    # where MStar = data_coordinate_system_correction_matrix
    MStar = data_coordinate_system_correction_matrix  # (6, 6)

    # Index phase stiffness for each data point: C_dp(6,6,N)
    phase_idx = (data_point_phase - 1).astype(int)  # 0-based
    C_dp = C[:, :, phase_idx]  # (6, 6, nDataPoints)

    # M @ C @ M^T  (batch multiply)
    MC = np.einsum('ijk,jlk->ilk', DData, C_dp)        # M @ C
    MCMt = np.einsum('ijk,ljk->ilk', MC, DData)         # (M @ C) @ M^T

    # MStar @ (M @ C @ M^T) @ MStar^T
    tmp = np.einsum('ij,jkn->ikn', MStar, MCMt)         # MStar @ MCMt
    DData = np.einsum('ijn,kj->ikn', tmp, MStar)         # tmp @ MStar^T

    return DData, dataPointStressBondMatrix
