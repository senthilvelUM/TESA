"""
Compute per-phase Voigt, Reuss, and Hill anisotropy matrices.

Unlike getVoigtReussHill which computes a single effective stiffness for
the whole polycrystal, this function computes separate VRH estimates for
each phase — averaging only over data points belonging to that phase.

This gives the single-crystal anisotropy contribution of each phase,
useful for understanding the texture-dependent properties of individual
mineral phases.

"""

import numpy as np


def get_anisotropy_matrices(
        data_euler_angle,
        data_phase,
        phase_stiffness_matrix,
        ebsd_correction_matrix,
        number_phases):
    """
    Compute per-phase VRH anisotropy matrices.

    Parameters
    ----------
    data_euler_angle : (nDataPoints, 3) array
        Euler angles (phi1, Phi, phi2) in radians at each data point.
    data_phase : (nDataPoints,) array of int
        Phase ID for each data point (1-based).
    phase_stiffness_matrix : list of (6, 6) arrays
        Stiffness matrix for each phase.
    ebsd_correction_matrix : (6, 6) array
        EBSD correction matrix MStar (6x6 Bond matrix).
    number_phases : int
        Number of phases.

    Returns
    -------
    AVoigt : (6, 6, nPhases) array
        Per-phase Voigt estimate of stiffness.
    AReuss : (6, 6, nPhases) array
        Per-phase Reuss estimate of stiffness.
    AHill : (6, 6, nPhases) array
        Per-phase Hill estimate (average of Voigt and Reuss).
    """
    nPhases = number_phases

    # Store stiffness and compliance as 3D arrays
    C = np.zeros((6, 6, nPhases))
    S = np.zeros((6, 6, nPhases))
    for i in range(nPhases):
        C[:, :, i] = phase_stiffness_matrix[i]
        S[:, :, i] = np.linalg.inv(phase_stiffness_matrix[i])

    AVoigt = np.zeros((6, 6, nPhases))
    AReuss = np.zeros((6, 6, nPhases))

    # Split EBSD points into 5 groups to conserve memory
    nTotal = len(data_phase)
    n = round(nTotal / 5)
    ri = [0, n, 2 * n, 3 * n, 4 * n]
    rf = [n, 2 * n, 3 * n, 4 * n, nTotal]

    MC = ebsd_correction_matrix
    NC = np.linalg.solve(MC.T, np.eye(6))

    for g in range(5):
        # Setup angle and phase info for this group
        phi1 = data_euler_angle[ri[g]:rf[g], 0]
        PHI  = data_euler_angle[ri[g]:rf[g], 1]
        phi2 = data_euler_angle[ri[g]:rf[g], 2]
        phase = data_phase[ri[g]:rf[g]].astype(int)
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

        M = np.zeros((6, 6, nPts))
        M[0:3, 0:3, :] = Mul
        M[0:3, 3:6, :] = Mur
        M[3:6, 0:3, :] = Mll
        M[3:6, 3:6, :] = Mlr

        N = np.zeros((6, 6, nPts))
        N[0:3, 0:3, :] = Mul
        N[0:3, 3:6, :] = 0.5 * Mur
        N[3:6, 0:3, :] = 2.0 * Mll
        N[3:6, 3:6, :] = Mlr

        # Accumulate per-phase Voigt and Reuss
        for i in range(nPhases):
            mask = (phase == i + 1)
            if not np.any(mask):
                continue
            n_phase = np.sum(mask)

            # Voigt: MC @ M @ C(i) @ M^T @ MC^T for points of phase i
            M_ph = M[:, :, mask]  # (6, 6, n_phase)
            C_ph = np.tile(C[:, :, i:i+1], (1, 1, n_phase))
            MC_M = np.einsum('ij,jkn->ikn', MC, M_ph)
            MC_M_C = np.einsum('ijk,jlk->ilk', MC_M, C_ph)
            MC_M_C_Mt = np.einsum('ijk,ljk->ilk', MC_M_C, M_ph)
            AV = np.einsum('ijn,kj->ikn', MC_M_C_Mt, MC)
            AVoigt[:, :, i] += np.sum(AV, axis=2)

            # Reuss: NC @ N @ S(i) @ N^T @ NC^T for points of phase i
            N_ph = N[:, :, mask]
            S_ph = np.tile(S[:, :, i:i+1], (1, 1, n_phase))
            NC_N = np.einsum('ij,jkn->ikn', NC, N_ph)
            NC_N_S = np.einsum('ijk,jlk->ilk', NC_N, S_ph)
            NC_N_S_Nt = np.einsum('ijk,ljk->ilk', NC_N_S, N_ph)
            SV = np.einsum('ijn,kj->ikn', NC_N_S_Nt, NC)
            AReuss[:, :, i] += np.sum(SV, axis=2)

    # Normalize per phase
    for i in range(nPhases):
        n_phase_total = np.sum(data_phase == i + 1)
        if n_phase_total > 0:
            vf = 1.0 / n_phase_total
            AReuss[:, :, i] = np.linalg.solve(vf * AReuss[:, :, i], np.eye(6))
            AVoigt[:, :, i] = vf * AVoigt[:, :, i]

    AHill = 0.5 * (AVoigt + AReuss)

    return AVoigt, AReuss, AHill
