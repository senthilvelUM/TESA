"""
Compute the geometric mean estimate for effective stiffness.

Rotates phase stiffness by Euler angles at every EBSD data point, takes
the matrix logarithm of each rotated stiffness, averages in log space,
then takes the matrix exponential:

CGeo = expm( (1/N) * sum_i  logm( MStar @ M_i @ C(phase_i) @ M_i^T @ MStar^T ) )

The EBSD data is processed in batches of ~N/5 to conserve memory.

"""

import numpy as np
from scipy.linalg import logm, expm


def get_geometric_mean(
        original_data_euler_angle,
        original_data_phase,
        phase_stiffness_matrix,
        ebsd_correction_matrix):
    """
    Compute geometric mean estimate for effective stiffness.

    Parameters
    ----------
    original_data_euler_angle : (nDataPoints, 3) array
        Euler angles (phi1, Phi, phi2) in radians.
    original_data_phase : (nDataPoints,) array of int
        Phase ID for each EBSD data point (1-based).
    phase_stiffness_matrix : list of (6, 6) arrays
        Stiffness matrix for each phase.
    ebsd_correction_matrix : (6, 6) array
        EBSD correction matrix MStar.

    Returns
    -------
    CGeo : (6, 6) array
        Geometric mean estimate of effective stiffness.
    """
    # Store stiffness as 3D array
    nPhases = int(np.max(original_data_phase))
    C = np.zeros((6, 6, nPhases))
    for i in range(nPhases):
        C[:, :, i] = phase_stiffness_matrix[i]

    # Split EBSD points into 5 groups to conserve memory
    nTotal = len(original_data_phase)
    n = round(nTotal / 5)
    ri = [0, n, 2 * n, 3 * n, 4 * n]
    rf = [n, 2 * n, 3 * n, 4 * n, nTotal]

    MC = ebsd_correction_matrix
    CGeo = np.zeros((6, 6))

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

        M = np.zeros((6, 6, nPts))
        M[0:3, 0:3, :] = Mul
        M[0:3, 3:6, :] = Mur
        M[3:6, 0:3, :] = Mll
        M[3:6, 3:6, :] = Mlr

        # Rotate each point's stiffness: Ct = MC @ M @ C(phase) @ M^T @ MC^T
        phase_idx = (phase - 1).astype(int)
        C_dp = C[:, :, phase_idx]  # (6, 6, nPts)

        # M @ C(phase) @ M^T
        MC_local = np.einsum('ijk,jlk->ilk', M, C_dp)
        MCMt = np.einsum('ijk,ljk->ilk', MC_local, M)

        # MC @ MCMt @ MC^T
        tmp = np.einsum('ij,jkn->ikn', MC, MCMt)
        Ct = np.einsum('ijn,kj->ikn', tmp, MC)  # (6, 6, nPts)

        # Take matrix logarithm of each rotated stiffness and accumulate
        # Vectorized via eigendecomposition: logm(A) = V @ diag(log(λ)) @ V^T
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ct_batch = Ct.transpose(2, 0, 1)  # (nPts, 6, 6)
            eigvals, eigvecs = np.linalg.eigh(Ct_batch)
            log_eigvals = np.log(np.maximum(eigvals, 1e-30))
            log_Ct = np.einsum('...ij,...j,...kj->...ik', eigvecs, log_eigvals, eigvecs)
            CGeo += np.sum(log_Ct, axis=0)

    # Average and exponentiate
    vf = 1.0 / nTotal
    CGeo = expm(vf * CGeo).real

    # Symmetrize
    CGeo = 0.5 * (CGeo + CGeo.T)

    return CGeo
