"""
Compute adjusted compliance matrices at data points.

Converts the stress Bond matrix M to the strain Bond matrix N,
inverts phase stiffness to get compliance S = C^-1, then rotates:

SData = NStar @ N @ S @ N^T @ NStar^T

where N is the strain Bond matrix (derived from M), NStar = inv(MStar^T),
and S is the phase compliance.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np


def compute_data_point_adjusted_compliance_matrix(
        data_point_phase,
        phase_elastic_stiffness_matrix,
        data_coordinate_system_correction_matrix,
        data_point_stress_bond_matrix):
    """
    Compute rotated compliance matrix at each data point.

    Converts the stress Bond matrix M to the strain Bond matrix N, inverts
    each phase stiffness to get compliance S = C^-1, then computes
    ``SData = NStar @ N @ S(phase) @ N^T @ NStar^T``
    where ``NStar = inv(MStar^T)``.

    Parameters
    ----------
    data_point_phase : ndarray, shape (n_data_points,), dtype int
        Phase ID for each data point (1-based).
    phase_elastic_stiffness_matrix : list of ndarray
        Stiffness matrix for each phase; each entry has shape (6, 6).
    data_coordinate_system_correction_matrix : ndarray, shape (6, 6)
        EBSD coordinate system correction (Bond) matrix MStar.
    data_point_stress_bond_matrix : ndarray, shape (6, 6, n_data_points)
        Stress Bond matrix M at each data point, as returned by
        ``compute_data_point_adjusted_stiffness_matrix``.

    Returns
    -------
    SData : ndarray, shape (6, 6, n_data_points)
        Rotated compliance matrix at each data point.
    """
    # Local declarations
    nPhases = int(np.max(data_point_phase))
    data_point_phase = np.asarray(data_point_phase, dtype=int)

    # Compute the strain bond matrices N from the stress bond matrices M
    # N = M with upper-right block halved and lower-left block doubled
    SData = data_point_stress_bond_matrix.copy()
    SData[0:3, 3:6, :] = SData[0:3, 3:6, :] * 0.5
    SData[3:6, 0:3, :] = SData[3:6, 0:3, :] * 2.0

    # Compute phase compliance matrices: S = C^-1
    S = np.zeros((6, 6, nPhases))
    for iPhase in range(nPhases):
        S[:, :, iPhase] = np.linalg.solve(
            phase_elastic_stiffness_matrix[iPhase], np.eye(6))

    # NStar = inv(MStar^T)
    NStar = np.linalg.solve(
        data_coordinate_system_correction_matrix.T, np.eye(6))

    # Index compliance for each data point
    phase_idx = (data_point_phase - 1).astype(int)  # 0-based
    S_dp = S[:, :, phase_idx]  # (6, 6, nDataPoints)

    # SData = NStar @ N @ S @ N^T @ NStar^T  (batch multiply)
    NS = np.einsum('ijk,jlk->ilk', SData, S_dp)          # N @ S
    NSNt = np.einsum('ijk,ljk->ilk', NS, SData)           # (N @ S) @ N^T
    tmp = np.einsum('ij,jkn->ikn', NStar, NSNt)           # NStar @ NSNt
    SData = np.einsum('ijn,kj->ikn', tmp, NStar)           # tmp @ NStar^T

    return SData
