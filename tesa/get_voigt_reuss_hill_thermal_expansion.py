"""
Compute Voigt, Reuss, and Hill estimates for effective thermal expansion.

Rotates phase stiffness and thermal expansion by Euler angles at every EBSD
data point, then computes:
  - Reuss: arithmetic mean of rotated alpha
  - Voigt: inv(<C>) @ <C @ alpha>  (volume-averaged beta, then alpha = S_Voigt @ beta)
  - Hill:  average of Voigt and Reuss

Note: unlike the stiffness VRH, the Voigt estimate for alpha requires the
Voigt stiffness, so both C and alpha rotations are performed.

"""

import numpy as np


def get_voigt_reuss_hill_thermal_expansion(
        original_data_euler_angle,
        original_data_phase,
        phase_stiffness_matrix,
        phase_thermal_expansion_matrix,
        ebsd_correction_matrix):
    """
    Compute Voigt, Reuss, Hill estimates for effective thermal expansion.

    Parameters
    ----------
    original_data_euler_angle : (nDataPoints, 3) array
        Euler angles (phi1, Phi, phi2) in radians.
    original_data_phase : (nDataPoints,) array of int
        Phase ID for each EBSD data point (1-based).
    phase_stiffness_matrix : list of (6, 6) arrays
        Stiffness matrix for each phase.
    phase_thermal_expansion_matrix : list of (6, 1) or (6,) arrays
        Thermal expansion for each phase.
    ebsd_correction_matrix : (6, 6) array
        EBSD correction matrix MStar.

    Returns
    -------
    AVoigt : (6, 1) array
        Voigt estimate of effective thermal expansion.
    AReuss : (6, 1) array
        Reuss estimate of effective thermal expansion.
    AHill : (6, 1) array
        Hill estimate (average of Voigt and Reuss).
    """
    # Store stiffness and alpha as 3D arrays
    nPhases = int(np.max(original_data_phase))
    C = np.zeros((6, 6, nPhases))
    Alpha = np.zeros((6, 1, nPhases))
    for i in range(nPhases):
        C[:, :, i] = phase_stiffness_matrix[i]
        alpha_i = np.asarray(phase_thermal_expansion_matrix[i]).ravel()
        Alpha[:, 0, i] = alpha_i

    # Setup angle and phase info
    phi1 = original_data_euler_angle[:, 0]
    PHI  = original_data_euler_angle[:, 1]
    phi2 = original_data_euler_angle[:, 2]
    phase = original_data_phase.astype(int)
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

    # Build Bond matrices M and N
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

    # M = [Mul Mur ; Mll Mlr]  (stress Bond matrix)
    M = np.zeros((6, 6, nPts))
    M[0:3, 0:3, :] = Mul
    M[0:3, 3:6, :] = Mur
    M[3:6, 0:3, :] = Mll
    M[3:6, 3:6, :] = Mlr

    # N = [Mul 0.5*Mur; 2*Mll Mlr]  (strain Bond matrix)
    N = np.zeros((6, 6, nPts))
    N[0:3, 0:3, :] = Mul
    N[0:3, 3:6, :] = 0.5 * Mur
    N[3:6, 0:3, :] = 2.0 * Mll
    N[3:6, 3:6, :] = Mlr

    MStar = ebsd_correction_matrix
    NStar = np.linalg.solve(MStar.T, np.eye(6))
    vf = 1.0 / nPts

    # Index phase arrays for each data point
    phase_idx = (phase - 1).astype(int)
    Alpha_dp = Alpha[:, :, phase_idx]  # (6, 1, nPts)
    C_dp = C[:, :, phase_idx]          # (6, 6, nPts)

    # Reuss estimate: arithmetic mean of rotated alpha
    # Alphai = NStar @ N @ Alpha(phase)
    N_Alpha = np.einsum('ijk,jlk->ilk', N, Alpha_dp)      # (6, 1, nPts)
    Alphai = np.einsum('ij,jlk->ilk', NStar, N_Alpha)     # (6, 1, nPts)
    AlphaReuss = vf * np.sum(Alphai, axis=2, keepdims=True)  # (6, 1, 1)
    AlphaReuss = AlphaReuss[:, :, 0]  # (6, 1)

    # Compute rotated stiffness: Ci = MStar @ M @ C(phase) @ M^T @ MStar^T
    MC = np.einsum('ijk,jlk->ilk', M, C_dp)               # (6, 6, nPts)
    MCMt = np.einsum('ijk,ljk->ilk', MC, M)                # (6, 6, nPts)
    tmp = np.einsum('ij,jkn->ikn', MStar, MCMt)            # (6, 6, nPts)
    Ci = np.einsum('ijn,kj->ikn', tmp, MStar)               # (6, 6, nPts)

    # Voigt stiffness: arithmetic mean of rotated C
    CVoigt = vf * np.sum(Ci, axis=2)  # (6, 6)

    # Voigt beta: arithmetic mean of Ci @ Alphai
    Ci_Alphai = np.einsum('ijk,jlk->ilk', Ci, Alphai)     # (6, 1, nPts)
    BetaVoigt = vf * np.sum(Ci_Alphai, axis=2, keepdims=True)  # (6, 1, 1)
    BetaVoigt = BetaVoigt[:, :, 0]  # (6, 1)

    # Voigt alpha: inv(CVoigt) @ BetaVoigt
    AlphaVoigt = np.linalg.solve(CVoigt, np.eye(6)) @ BetaVoigt  # (6, 1)

    # Hill estimate: average of Voigt and Reuss
    AHill = 0.5 * (AlphaVoigt + AlphaReuss)

    AVoigt = AlphaVoigt
    AReuss = AlphaReuss

    return AVoigt, AReuss, AHill
