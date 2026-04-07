"""
Compute Voigt, Reuss, and Hill estimates for effective thermal conductivity.

Rotates phase thermal conductivity by Euler angles at every EBSD data point
using 3x3 direction cosines (not Bond matrices), then computes:
  - Voigt: arithmetic mean of rotated kappa
  - Reuss: harmonic mean (arithmetic mean of inv(kappa), then invert)
  - Hill:  average of Voigt and Reuss

The EBSD data is processed in batches of ~N/6 to conserve memory.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np


def compute_macroscale_voigt_reuss_hill_thermal_conductivity(
        data_point_euler_angles,
        data_point_phase,
        phase_thermal_conductivity_matrix,
        data_coordinate_system_correction_angle):
    """
    Compute VRH estimates for effective thermal conductivity.

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
    kappaVoigt : (3, 3) array
        Voigt estimate of effective thermal conductivity.
    kappaReuss : (3, 3) array
        Reuss estimate of effective thermal conductivity.
    kappaHill : (3, 3) array
        Hill estimate (average of Voigt and Reuss).
    """
    # Local declarations
    nDataPoints = len(data_point_phase)
    nPhases = int(np.max(data_point_phase))

    # Store phase conductivities and their inverses
    kappaPhase = np.zeros((3, 3, nPhases))
    invkappaPhase = np.zeros((3, 3, nPhases))
    for iPhase in range(nPhases):
        kappaPhase[:, :, iPhase] = phase_thermal_conductivity_matrix[iPhase]
        invkappaPhase[:, :, iPhase] = np.linalg.solve(
            phase_thermal_conductivity_matrix[iPhase], np.eye(3))

    # Split data points into 6 groups
    nSplit = round(nDataPoints / 6)
    ri = [0, nSplit, 2 * nSplit, 3 * nSplit, 4 * nSplit, 5 * nSplit]
    rf = [nSplit, 2 * nSplit, 3 * nSplit, 4 * nSplit, 5 * nSplit, nDataPoints]

    # Loop through all 6 groups
    kappaVoigt_sum = np.zeros((3, 3))
    kappaReuss_sum = np.zeros((3, 3))

    for iSplit in range(6):
        # Setup angle and phase info for this group
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

        # Index phase arrays
        phase_idx = (phase - 1).astype(int)

        # Voigt: sum of a @ kappa(phase) @ a^T
        kappa_dp = kappaPhase[:, :, phase_idx]  # (3, 3, nPts)
        aK = np.einsum('ijk,jlk->ilk', a, kappa_dp)
        aKat = np.einsum('ijk,ljk->ilk', aK, a)
        kappaVoigt_sum += np.sum(aKat, axis=2)

        # Reuss: sum of a @ inv(kappa)(phase) @ a^T
        invkappa_dp = invkappaPhase[:, :, phase_idx]  # (3, 3, nPts)
        aiK = np.einsum('ijk,jlk->ilk', a, invkappa_dp)
        aiKat = np.einsum('ijk,ljk->ilk', aiK, a)
        kappaReuss_sum += np.sum(aiKat, axis=2)

    # Apply EBSD correction and normalize
    theta = data_coordinate_system_correction_angle
    aC = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])

    kappaVoigt = (1.0 / nDataPoints) * aC @ kappaVoigt_sum @ aC.T
    kappaReuss_inv = (1.0 / nDataPoints) * aC @ kappaReuss_sum @ aC.T

    # Symmetrize Voigt
    kappaVoigt = 0.5 * (kappaVoigt + kappaVoigt.T)

    # Reuss: invert the averaged resistivity, then symmetrize
    kappaReuss = np.linalg.solve(kappaReuss_inv, np.eye(3))
    kappaReuss = 0.5 * (kappaReuss + kappaReuss.T)

    # Hill: average of Voigt and Reuss
    kappaHill = 0.5 * (kappaVoigt + kappaReuss)

    return kappaVoigt, kappaReuss, kappaHill
