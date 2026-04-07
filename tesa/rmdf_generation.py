"""
Random microstructure generation using Random Distance Metric Functions (RDMFs).

Generates a synthetic 2-phase periodic microstructure on a square domain
by computing a weighted sum of Gaussian radial basis functions at random
seed points, then thresholding to achieve a target volume fraction.
Includes a cleanup step to remove stranded single pixels.

Assumes a square (1x1) representative material element (RME).

"""

import numpy as np
from scipy.interpolate import interp1d


def rmdf_generation(N=27, vf=0.6, NI=200, min_nb=5, seed=None):
    """
    Generate a random 2-phase periodic microstructure.

    Parameters
    ----------
    N : int
        Number of seed points per dimension (should be divisible by 3).
    vf : float or array-like
        Target volume fraction(s) of phase 1. Scalar or 1D array.
    NI : int
        Resolution of the output image (NI x NI pixels).
    min_nb : int
        Minimum number of same-phase neighbors in a 5x5 window.
        Pixels with fewer neighbors are flipped (stranded pixel removal).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    results : list of dict
        One dict per volume fraction, each containing:
        - 'vf_target': requested volume fraction
        - 'vf_actual': achieved volume fraction
        - 'phase_map': (NI, NI) array of int (0 or 1)
        - 'phase_image': (NI, NI, 3) array of uint8 (RGB image)
    """
    if seed is not None:
        np.random.seed(seed)

    vf = np.atleast_1d(np.asarray(vf, dtype=float))

    # Create random periodic microstructure of coordinates with weights
    X_grid, Y_grid = np.meshgrid(np.arange(N, 2 * N + 1),
                                  np.arange(N, 2 * N + 1))
    C = np.random.rand(N + 1, N + 1)

    # Enforce periodicity on weights
    C[-1, 0] = C[0, 0]
    C[-1, -1] = C[0, 0]
    C[0, -1] = C[0, 0]
    C[-1, :] = C[0, :]
    C[:, -1] = C[:, 0]

    # Distance normalization factor
    w = 3 * N / np.sqrt((3 * N + 1) ** 2)

    # Flatten for computation
    X = X_grid.ravel()
    Y = Y_grid.ravel()
    C_flat = C.ravel().copy()

    # Halve weights for overlapping boundary points
    C_flat[(Y == N) | (Y == 2 * N)] *= 0.5
    C_flat[(X == N) | (X == 2 * N)] *= 0.5

    # Periodic offsets
    dx = np.array([-N, 0, N, -N, 0, N, -N, 0, N])
    dy = np.array([-N, -N, -N, 0, 0, 0, N, N, N])

    # Create evaluation grid
    XI_grid, YI_grid = np.meshgrid(np.linspace(0, NI, NI),
                                    np.linspace(0, NI, NI))
    XI_grid = (XI_grid / NI) * N + N
    YI_grid = (YI_grid / NI) * N + N

    # Compute the function value at all grid points
    FI = np.zeros_like(XI_grid)
    w2 = w ** 2
    for i in range(XI_grid.shape[0]):
        for ii in range(len(X)):
            for j in range(9):
                d = (XI_grid[i, :] - X[ii] + dx[j]) ** 2 + \
                    (YI_grid[i, :] - Y[ii] + dy[j]) ** 2
                FI[i, :] += C_flat[ii] * np.exp(-d / w2)

    # Normalize to [0, 1]
    XI_flat = XI_grid.ravel()
    YI_flat = YI_grid.ravel()
    FI_flat = FI.ravel()
    FI_flat = (FI_flat - np.min(FI_flat)) / (np.max(FI_flat) - np.min(FI_flat))

    # Reshape and enforce periodicity
    XI_2d = XI_flat.reshape(NI, NI) - N
    YI_2d = YI_flat.reshape(NI, NI) - N
    FI_2d = FI_flat.reshape(NI, NI)
    FI_2d[-1, 0] = FI_2d[0, 0]
    FI_2d[-1, -1] = FI_2d[0, 0]
    FI_2d[0, -1] = FI_2d[0, 0]
    FI_2d[-1, :] = FI_2d[0, :]
    FI_2d[:, -1] = FI_2d[:, 0]
    FI_flat = FI_2d.ravel()

    # Find volume fraction curve for phase 1
    FO = np.arange(0, 1.01, 0.01)
    vo = np.zeros(len(FO))
    for i in range(len(FO)):
        vo[i] = np.sum(FI_flat > FO[i]) / len(FI_flat)

    # Remove duplicate vo values for interpolation
    _, unique_idx = np.unique(vo, return_index=True)
    unique_idx = np.sort(unique_idx)
    vo_unique = vo[unique_idx]
    FO_unique = FO[unique_idx]

    # Interpolate to find threshold for each target volume fraction
    interp_func = interp1d(vo_unique, FO_unique, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    FC = interp_func(vf)

    # Generate microstructures for each volume fraction
    results = []
    eps_val = np.sqrt(np.finfo(float).eps)

    for i in range(len(vf)):
        # Threshold to create binary phase map
        FB = np.zeros(len(FI_flat))
        FB[FI_flat >= FC[i] + eps_val] = 1.0

        # Reshape for stranded pixel removal
        FB_2d = FB.reshape(NI, NI)

        # Pad with periodic boundary for neighborhood check
        FB_pad = np.vstack([FB_2d[-4:, :], FB_2d, np.flipud(FB_2d[:4, :])])
        FB_pad = np.hstack([FB_pad[:, -4:], FB_pad, np.fliplr(FB_pad[:, :4])])

        # Iteratively remove stranded pixels
        new_sum = np.sum(FB_pad)
        old_sum = 0
        while new_sum != old_sum:
            old_sum = new_sum
            for m in range(2, FB_pad.shape[0] - 3):
                for n in range(2, FB_pad.shape[1] - 3):
                    cb = FB_pad[m, n]
                    fb = FB_pad[m - 2:m + 3, n - 2:n + 3]
                    nb = np.sum(fb == cb)
                    if nb < min_nb:
                        FB_pad[m, n] = 1.0 - cb
            new_sum = np.sum(FB_pad)

        # Remove padding
        FB_2d = FB_pad[4:-4, 4:-4]

        # Enforce periodicity
        FB_2d[:, -1] = FB_2d[:, 0]
        FB_2d[-1, :] = FB_2d[0, :]

        # Create RGB image (phase 0 = red, phase 1 = blue)
        FB_flat = FB_2d.ravel()
        C_img = np.zeros((len(FB_flat), 3), dtype=np.uint8)
        C_img[FB_flat == 0, 0] = 255  # red
        C_img[FB_flat == 1, 2] = 255  # blue
        C_img = C_img.reshape(NI, NI, 3)

        # Compute actual volume fraction
        vf_actual = np.sum(FB_flat == 1) / len(FB_flat)

        results.append({
            'vf_target': vf[i],
            'vf_actual': vf_actual,
            'phase_map': FB_2d.astype(int),
            'phase_image': C_img,
        })

    return results
