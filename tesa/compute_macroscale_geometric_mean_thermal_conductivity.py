"""
Compute the geometric mean estimate for effective thermal conductivity.

Rotates phase thermal conductivity by Euler angles at every EBSD data point,
takes the matrix logarithm of each rotated kappa, averages in log space,
then takes the matrix exponential:

kappaGeo = expm( (1/N) * sum_i logm( aC @ a_i @ kappa(phase_i) @ a_i^T @ aC^T ) )

The EBSD data is processed in batches of ~N/6 to conserve memory.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from scipy.linalg import logm, expm


def compute_macroscale_geometric_mean_thermal_conductivity(
        data_point_euler_angles,
        data_point_phase,
        phase_thermal_conductivity_matrix,
        data_coordinate_system_correction_angle):
    """
    Compute geometric mean estimate for effective thermal conductivity.

    Parameters
    ----------
    data_point_euler_angles : (nDataPoints, 3) array
        Euler angles (phi1, Phi, phi2) in radians.
    data_point_phase : (nDataPoints,) array of int
        Phase ID for each data point (1-based).
    phase_thermal_conductivity_matrix : list of (3, 3) arrays
        Thermal conductivity for each phase.
    data_coordinate_system_correction_angle : float
        EBSD correction angle in radians.

    Returns
    -------
    kappaGeometricMean : (3, 3) array
        Geometric mean estimate of effective thermal conductivity.
    """
    # Local declarations
    nDataPoints = len(data_point_phase)
    nPhases = int(np.max(data_point_phase))

    # EBSD correction 3x3 rotation
    theta = data_coordinate_system_correction_angle
    aC = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])

    # Store phase conductivities
    kappaPhase = np.zeros((3, 3, nPhases))
    for iPhase in range(nPhases):
        kappaPhase[:, :, iPhase] = phase_thermal_conductivity_matrix[iPhase]

    # Split data points into 6 groups
    nSplit = round(nDataPoints / 6)
    ri = [0, nSplit, 2 * nSplit, 3 * nSplit, 4 * nSplit, 5 * nSplit]
    rf = [nSplit, 2 * nSplit, 3 * nSplit, 4 * nSplit, 5 * nSplit, nDataPoints]

    # Loop through all 6 groups
    kappaGeo_sum = np.zeros((3, 3))

    for iSplit in range(6):
        # Setup angle and phase info
        phi1 = data_point_euler_angles[ri[iSplit]:rf[iSplit], 0]
        PHI  = data_point_euler_angles[ri[iSplit]:rf[iSplit], 1]
        phi2 = data_point_euler_angles[ri[iSplit]:rf[iSplit], 2]
        phase = data_point_phase[ri[iSplit]:rf[iSplit]].astype(int)
        nPts = len(phi1)

        # Build direction cosines matrices
        a1 = np.zeros((3, 3, nPts))
        a1[0, 0, :] = np.cos(phi1)
        a1[0, 1, :] = -np.sin(phi1)
        a1[0, 2, :] = 0.0
        a1[1, 0, :] = np.sin(phi1)
        a1[1, 1, :] = np.cos(phi1)
        a1[1, 2, :] = 0.0
        a1[2, 0, :] = 0.0
        a1[2, 1, :] = 0.0
        a1[2, 2, :] = 1.0

        a2 = np.zeros((3, 3, nPts))
        a2[0, 0, :] = 1.0
        a2[0, 1, :] = 0.0
        a2[0, 2, :] = 0.0
        a2[1, 0, :] = 0.0
        a2[1, 1, :] = np.cos(PHI)
        a2[1, 2, :] = -np.sin(PHI)
        a2[2, 0, :] = 0.0
        a2[2, 1, :] = np.sin(PHI)
        a2[2, 2, :] = np.cos(PHI)

        a3 = np.zeros((3, 3, nPts))
        a3[0, 0, :] = np.cos(phi2)
        a3[0, 1, :] = -np.sin(phi2)
        a3[0, 2, :] = 0.0
        a3[1, 0, :] = np.sin(phi2)
        a3[1, 1, :] = np.cos(phi2)
        a3[1, 2, :] = 0.0
        a3[2, 0, :] = 0.0
        a3[2, 1, :] = 0.0
        a3[2, 2, :] = 1.0

        a = np.einsum('ijk,jlk->ilk', a1, a2)
        a = np.einsum('ijk,jlk->ilk', a, a3)

        # Rotate: aC @ a @ kappa(phase) @ a^T @ aC^T
        phase_idx = (phase - 1).astype(int)
        kappa_dp = kappaPhase[:, :, phase_idx]  # (3, 3, nPts)

        # a @ kappa
        aK = np.einsum('ijk,jlk->ilk', a, kappa_dp)
        # (a @ kappa) @ a^T
        aKat = np.einsum('ijk,ljk->ilk', aK, a)
        # aC @ aKat
        tmp = np.einsum('ij,jkn->ikn', aC, aKat)
        # tmp @ aC^T
        kappa_rot = np.einsum('ijn,kj->ikn', tmp, aC)  # (3, 3, nPts)

        # Take logm of each rotated kappa and sum (vectorized via eigendecomposition)
        # For symmetric positive definite matrices: logm(A) = V @ diag(log(λ)) @ V^T
        kappa_batch = kappa_rot.transpose(2, 0, 1)  # (nPts, 3, 3)
        eigvals, eigvecs = np.linalg.eigh(kappa_batch)
        log_eigvals = np.log(np.maximum(eigvals, 1e-30))
        log_kappa = np.einsum('...ij,...j,...kj->...ik', eigvecs, log_eigvals, eigvecs)
        kappaGeo_sum += np.sum(log_kappa, axis=0)

    # Average and exponentiate
    kappaGeometricMean = expm((1.0 / nDataPoints) * kappaGeo_sum).real

    return kappaGeometricMean
