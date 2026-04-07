"""
Compute adjusted thermal conductivity matrices at data points.

Rotates the 3x3 phase thermal conductivity matrix by the Euler angles
at each data point, then applies the EBSD coordinate system correction.

kappaData = aC @ a @ kappa(phase) @ a^T @ aC^T

where a is the 3x3 direction cosines matrix from Euler angles,
aC is the 3x3 EBSD correction rotation, and kappa is the phase
thermal conductivity.

Note: unlike stiffness (which uses 6x6 Bond matrices), thermal
conductivity is a 3x3 tensor rotated directly with the 3x3 direction
cosines matrix.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np


def compute_data_point_adjusted_thermal_conductivity_matrix(
        data_point_euler_angles,
        data_point_phase,
        phase_thermal_conductivity_matrix,
        data_coordinate_system_correction_angle):
    """
    Compute rotated thermal conductivity at each data point.

    Builds the 3x3 direction cosines matrix from Euler angles, then computes
    ``kappaData = aC @ a @ kappa(phase) @ a^T @ aC^T``
    where aC is the EBSD correction rotation and a is the combined
    Euler-angle rotation. Unlike stiffness (6x6 Bond matrices), thermal
    conductivity is a rank-2 tensor rotated with the 3x3 direction cosines.

    Parameters
    ----------
    data_point_euler_angles : ndarray, shape (n_data_points, 3)
        Euler angles (phi1, Phi, phi2) in radians.
    data_point_phase : ndarray, shape (n_data_points,), dtype int
        Phase ID for each data point (1-based).
    phase_thermal_conductivity_matrix : list of ndarray
        Thermal conductivity matrix for each phase; each entry has
        shape (3, 3).
    data_coordinate_system_correction_angle : float
        EBSD correction angle in radians.

    Returns
    -------
    kappaData : ndarray, shape (3, 3, n_data_points)
        Rotated thermal conductivity matrix at each data point.
    """
    # Local declarations
    phi1 = data_point_euler_angles[:, 0]
    PHI  = data_point_euler_angles[:, 1]
    phi2 = data_point_euler_angles[:, 2]
    nDataPoints = len(phi1)
    dataPointPhase = np.asarray(data_point_phase, dtype=int)
    nPhases = int(np.max(dataPointPhase))

    # Store thermal conductivities as 3D array
    K = np.zeros((3, 3, nPhases))
    for i in range(nPhases):
        K[:, :, i] = phase_thermal_conductivity_matrix[i]

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

    # EBSD coordinate system correction rotation (3x3)
    theta = data_coordinate_system_correction_angle
    aC = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])

    # Index phase conductivity for each data point
    phase_idx = (dataPointPhase - 1).astype(int)
    K_dp = K[:, :, phase_idx]  # (3, 3, nDataPoints)

    # kappaData = aC @ a @ kappa @ a^T @ aC^T  (batch multiply)
    aK = np.einsum('ijk,jlk->ilk', a, K_dp)             # a @ kappa
    aKat = np.einsum('ijk,ljk->ilk', aK, a)             # (a @ kappa) @ a^T
    tmp = np.einsum('ij,jkn->ikn', aC, aKat)            # aC @ aKat
    kappaData = np.einsum('ijn,kj->ikn', tmp, aC)       # tmp @ aC^T

    return kappaData
