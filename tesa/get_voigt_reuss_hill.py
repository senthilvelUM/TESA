"""
Compute Voigt, Reuss, and Hill estimates for effective stiffness.

Rotates phase stiffness matrices by Euler angles at every EBSD data point,
then computes:
  - Voigt: arithmetic mean of rotated stiffness C
  - Reuss: harmonic mean (arithmetic mean of compliance S, then invert)
  - Hill:  average of Voigt and Reuss

The EBSD data is processed in batches of ~N/5 to conserve memory for
large maps.

"""

import numpy as np


def get_voigt_reuss_hill(
        original_data_euler_angle,
        original_data_phase,
        phase_stiffness_matrix,
        ebsd_correction_matrix):
    """
    Compute Voigt, Reuss, and Hill estimates for effective stiffness.

    Parameters
    ----------
    original_data_euler_angle : (nDataPoints, 3) array
        Euler angles (phi1, Phi, phi2) in radians at each EBSD point.
    original_data_phase : (nDataPoints,) array of int
        Phase ID for each EBSD data point (1-based).
    phase_stiffness_matrix : list of (6, 6) arrays
        Stiffness matrix for each phase.
    ebsd_correction_matrix : (6, 6) array
        EBSD coordinate system correction (Bond) matrix MStar.

    Returns
    -------
    CVoigt : (6, 6) array
        Voigt estimate of effective stiffness.
    CReuss : (6, 6) array
        Reuss estimate of effective stiffness.
    CHill : (6, 6) array
        Hill estimate (average of Voigt and Reuss).
    """
    # Store stiffness and compliance as 3D arrays
    nPhases = int(np.max(original_data_phase))
    C = np.zeros((6, 6, nPhases))
    S = np.zeros((6, 6, nPhases))
    for i in range(nPhases):
        C[:, :, i] = phase_stiffness_matrix[i]
        S[:, :, i] = np.linalg.solve(phase_stiffness_matrix[i], np.eye(6))

    # Split EBSD points into 5 groups to conserve memory
    nTotal = len(original_data_phase)
    n = round(nTotal / 5)
    ri = [0, n, 2 * n, 3 * n, 4 * n]
    rf = [n, 2 * n, 3 * n, 4 * n, nTotal]

    CVoigt_sum = np.zeros((6, 6))
    SReuss_sum = np.zeros((6, 6))

    # Loop through all 5 groups
    for g in range(5):
        # Setup angle and phase info for this group
        phi1 = original_data_euler_angle[ri[g]:rf[g], 0]
        PHI  = original_data_euler_angle[ri[g]:rf[g], 1]
        phi2 = original_data_euler_angle[ri[g]:rf[g], 2]
        phase = original_data_phase[ri[g]:rf[g]].astype(int)
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

        # Build stress Bond matrix M
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
        Mlr = np.zeros((3, 3, nPts))
        Mlr[0, 0, :] = a[1, 1, :] * a[2, 2, :] + a[1, 2, :] * a[2, 1, :]
        Mlr[0, 1, :] = a[1, 0, :] * a[2, 2, :] + a[1, 2, :] * a[2, 0, :]
        Mlr[0, 2, :] = a[1, 1, :] * a[2, 0, :] + a[1, 0, :] * a[2, 1, :]
        Mlr[1, 0, :] = a[0, 1, :] * a[2, 2, :] + a[0, 2, :] * a[2, 1, :]
        Mlr[1, 1, :] = a[0, 2, :] * a[2, 0, :] + a[0, 0, :] * a[2, 2, :]
        Mlr[1, 2, :] = a[0, 0, :] * a[2, 1, :] + a[0, 1, :] * a[2, 0, :]
        Mlr[2, 0, :] = a[0, 1, :] * a[1, 2, :] + a[0, 2, :] * a[1, 1, :]
        Mlr[2, 1, :] = a[0, 2, :] * a[1, 0, :] + a[0, 0, :] * a[1, 2, :]
        Mlr[2, 2, :] = a[0, 0, :] * a[1, 1, :] + a[0, 1, :] * a[1, 0, :]

        # M = [Mul Mur ; Mll Mlr]
        M = np.zeros((6, 6, nPts))
        M[0:3, 0:3, :] = Mul
        M[0:3, 3:6, :] = Mur
        M[3:6, 0:3, :] = Mll
        M[3:6, 3:6, :] = Mlr

        # Voigt: sum of M @ C(phase) @ M^T
        phase_idx = (phase - 1).astype(int)
        C_dp = C[:, :, phase_idx]  # (6, 6, nPts)
        MC = np.einsum('ijk,jlk->ilk', M, C_dp)
        MCMt = np.einsum('ijk,ljk->ilk', MC, M)
        CVoigt_sum += np.sum(MCMt, axis=2)

        # Strain Bond matrix N = [Mul 0.5*Mur; 2*Mll Mlr]
        N = np.zeros((6, 6, nPts))
        N[0:3, 0:3, :] = Mul
        N[0:3, 3:6, :] = 0.5 * Mur
        N[3:6, 0:3, :] = 2.0 * Mll
        N[3:6, 3:6, :] = Mlr

        # Reuss: sum of N @ S(phase) @ N^T
        S_dp = S[:, :, phase_idx]  # (6, 6, nPts)
        NS = np.einsum('ijk,jlk->ilk', N, S_dp)
        NSNt = np.einsum('ijk,ljk->ilk', NS, N)
        SReuss_sum += np.sum(NSNt, axis=2)

    # Apply EBSD correction and normalize
    MC_corr = ebsd_correction_matrix
    NC_corr = np.linalg.solve(MC_corr.T, np.eye(6))
    vf = 1.0 / nTotal

    CVoigt = vf * MC_corr @ CVoigt_sum @ MC_corr.T
    CReuss = np.linalg.solve(vf * NC_corr @ SReuss_sum @ NC_corr.T, np.eye(6))
    CHill = 0.5 * (CVoigt + CReuss)

    # Symmetrize
    CVoigt = 0.5 * (CVoigt + CVoigt.T)
    CReuss = 0.5 * (CReuss + CReuss.T)
    CHill = 0.5 * (CHill + CHill.T)

    return CVoigt, CReuss, CHill
