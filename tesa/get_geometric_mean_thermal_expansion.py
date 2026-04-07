"""
Compute the geometric mean estimate for effective thermal expansion.

Rotates alpha to 3x3 tensor form, computes beta = C_rotated @ alpha_rotated
in tensor form, takes matrix logarithm, averages in log space, takes matrix
exponential, then converts back to Voigt notation and applies inv(CGeo).

AGeo = inv(CGeo) @ expm( (1/N) * sum_i logm( beta_tensor_i ) )

where beta_tensor_i is the stress-temperature moduli in 3x3 tensor form
at each data point.

"""

import numpy as np
from scipy.linalg import logm, expm


def get_geometric_mean_thermal_expansion(
        original_data_euler_angle,
        original_data_phase,
        phase_stiffness_matrix,
        phase_thermal_expansion_matrix,
        ebsd_correction_matrix,
        ebsd_correction_angle,
        CGeo):
    """
    Compute geometric mean estimate for effective thermal expansion.

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
        EBSD correction matrix MStar (6x6 Bond matrix).
    ebsd_correction_angle : float
        EBSD correction angle in radians.
    CGeo : (6, 6) array
        Geometric mean effective stiffness (from get_geometric_mean).

    Returns
    -------
    AGeo : (6, 1) array
        Geometric mean estimate of effective thermal expansion.
    """
    # Store stiffness and alpha as 3D arrays
    nPhases = int(np.max(original_data_phase))
    C = np.zeros((6, 6, nPhases))
    A = np.zeros((3, 3, nPhases))
    for i in range(nPhases):
        C[:, :, i] = phase_stiffness_matrix[i]
        alpha_i = np.asarray(phase_thermal_expansion_matrix[i]).ravel()
        # Convert Voigt notation (6,) to 3x3 tensor
        # Engineering shear: alpha_4 = 2*eps_23, etc.
        A[0, 0, i] = alpha_i[0]
        A[0, 1, i] = 0.5 * alpha_i[5]
        A[0, 2, i] = 0.5 * alpha_i[4]
        A[1, 0, i] = 0.5 * alpha_i[5]
        A[1, 1, i] = alpha_i[1]
        A[1, 2, i] = 0.5 * alpha_i[3]
        A[2, 0, i] = 0.5 * alpha_i[4]
        A[2, 1, i] = 0.5 * alpha_i[3]
        A[2, 2, i] = alpha_i[2]

    # EBSD correction angle — 3x3 direction cosines
    theta = ebsd_correction_angle
    aC = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])

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

    # Rotate alpha in 3x3 tensor form: At = aC @ a @ A(phase) @ a^T @ aC^T
    phase_idx = (phase - 1).astype(int)
    A_dp = A[:, :, phase_idx]  # (3, 3, nPts)

    # aC @ a
    aCa = np.einsum('ij,jkn->ikn', aC, a)  # (3, 3, nPts)
    # aCa @ A(phase)
    aCaA = np.einsum('ijk,jlk->ilk', aCa, A_dp)  # (3, 3, nPts)
    # aCaA @ a^T
    aCaAat = np.einsum('ijk,ljk->ilk', aCaA, a)  # (3, 3, nPts)
    # aCaAat @ aC^T
    At = np.einsum('ijn,kj->ikn', aCaAat, aC)  # (3, 3, nPts)

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

    M = np.zeros((6, 6, nPts))
    M[0:3, 0:3, :] = Mul
    M[0:3, 3:6, :] = Mur
    M[3:6, 0:3, :] = Mll
    M[3:6, 3:6, :] = Mlr

    # Compute rotated stiffness: Ci = MC @ M @ C(phase) @ M^T @ MC^T
    MC_corr = ebsd_correction_matrix
    C_dp = C[:, :, phase_idx]
    MC_local = np.einsum('ijk,jlk->ilk', M, C_dp)
    MCMt = np.einsum('ijk,ljk->ilk', MC_local, M)
    tmp = np.einsum('ij,jkn->ikn', MC_corr, MCMt)
    Ci = np.einsum('ijn,kj->ikn', tmp, MC_corr)  # (6, 6, nPts)

    # Sum contributions: convert At to beta tensor form, take logm (vectorized)
    vf = 1.0 / nPts

    # Convert rotated alpha (3x3 tensor) to Voigt (6, nPts) for all points at once
    Ai_voigt = np.zeros((6, nPts))
    Ai_voigt[0, :] = At[0, 0, :]
    Ai_voigt[1, :] = At[1, 1, :]
    Ai_voigt[2, :] = At[2, 2, :]
    Ai_voigt[3, :] = 2.0 * At[1, 2, :]
    Ai_voigt[4, :] = 2.0 * At[0, 2, :]
    Ai_voigt[5, :] = 2.0 * At[0, 1, :]

    # Compute beta = Ci @ alpha (Voigt) for all points: (6,6,nPts) @ (6,nPts) → (6,nPts)
    beta_voigt = np.einsum('ijk,jk->ik', Ci, Ai_voigt)

    # Convert beta (Voigt 6,nPts) back to 3x3 tensor (nPts, 3, 3) for batch eigendecomposition
    beta_mat = np.zeros((nPts, 3, 3))
    beta_mat[:, 0, 0] = beta_voigt[0, :]
    beta_mat[:, 1, 1] = beta_voigt[1, :]
    beta_mat[:, 2, 2] = beta_voigt[2, :]
    beta_mat[:, 1, 2] = beta_mat[:, 2, 1] = beta_voigt[3, :] / 2.0
    beta_mat[:, 0, 2] = beta_mat[:, 2, 0] = beta_voigt[4, :] / 2.0
    beta_mat[:, 0, 1] = beta_mat[:, 1, 0] = beta_voigt[5, :] / 2.0

    # Vectorized matrix logarithm via eigendecomposition: logm(A) = V @ diag(log(λ)) @ V^T
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eigvals, eigvecs = np.linalg.eigh(beta_mat)
        # Clamp small/negative eigenvalues to avoid log(0) or log(negative)
        eigvals_safe = np.where(np.abs(eigvals) > 1e-30, eigvals, 1e-30)
        log_eigvals = np.log(np.abs(eigvals_safe))
        log_beta = np.einsum('...ij,...j,...kj->...ik', eigvecs, log_eigvals, eigvecs)

    At_sum = np.sum(log_beta, axis=0)

    # Average in log space and exponentiate
    AGeo_mat = expm(vf * At_sum).real

    # Symmetrize
    AGeo_mat = 0.5 * (AGeo_mat + AGeo_mat.T)

    # Convert 3x3 tensor back to Voigt (6, 1)
    AGeo = np.array([
        [AGeo_mat[0, 0]],
        [AGeo_mat[1, 1]],
        [AGeo_mat[2, 2]],
        [2.0 * AGeo_mat[1, 2]],
        [2.0 * AGeo_mat[0, 2]],
        [2.0 * AGeo_mat[0, 1]]
    ])

    # Apply inv(CGeo) @ AGeo
    AGeo = np.linalg.solve(CGeo, np.eye(6)) @ AGeo

    return AGeo
