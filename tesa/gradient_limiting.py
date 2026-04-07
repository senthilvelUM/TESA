"""
Apply gradient limiting to a mesh density function.

Vectorized with NumPy for performance (replaces the original per-cell Python loop
with array operations). Produces identical results to the original.
"""

import numpy as np


def gradient_limiting(xx, yy, hho, g, verbose=False):
    """
    Apply gradient limiting to mesh density function.

    Parameters
    ----------
    xx  : (ny, nx) meshgrid X coordinates
    yy  : (ny, nx) meshgrid Y coordinates
    hho : (ny, nx) initial mesh density function
    g   : gradient limit parameter
    verbose : bool
        Print convergence progress.

    Returns
    -------
    hh : (ny, nx) gradient-limited mesh density function
    """
    xx = np.asarray(xx, dtype=float)
    yy = np.asarray(yy, dtype=float)
    hho = np.asarray(hho, dtype=float)

    # Grid spacing — extract directly from regular meshgrid (avoids expensive unique/sort)
    dx = abs(xx[0, 1] - xx[0, 0])
    dy = abs(yy[1, 0] - yy[0, 0])
    dt = min(dx, dy) / 2.0

    # Initial condition
    hh = hho.copy()

    tol = 1e-5
    max_delh = 2 * tol

    while True:
        if verbose:
            print(f"  Gradient Error = {max_delh:.6g}")

        # Adjust for periodic boundaries
        hh[2, :] = np.minimum(hh[2, :], hh[-3, :])
        hh[0, :] = hh[2, :]
        hh[-1, :] = hh[2, :]
        hh[1, :] = hh[2, :]
        hh[-2, :] = hh[2, :]
        hh[:, 2] = np.minimum(hh[:, 2], hh[:, -3])
        hh[:, 0] = hh[:, 2]
        hh[:, -1] = hh[:, 2]
        hh[:, 1] = hh[:, 2]
        hh[:, -2] = hh[:, 2]

        if max_delh < tol:
            break

        # Vectorized computation on interior points (rows 2:-2, cols 2:-2)
        # Extract the interior slab and its neighbors
        h_c = hh[2:-2, 2:-2]       # center
        h_l = hh[2:-2, 1:-3]       # left  (j-1)
        h_r = hh[2:-2, 3:-1]       # right (j+1)
        h_d = hh[1:-3, 2:-2]       # down  (i-1)
        h_u = hh[3:-1, 2:-2]       # up    (i+1)

        # Finite differences (matching the original loop exactly)
        Dminusx = (h_c - h_d) / dx    # (hh[i,j] - hh[i-1,j]) / dx
        Dplusx  = (h_u - h_c) / dx    # (hh[i+1,j] - hh[i,j]) / dx
        Dminusy = (h_c - h_l) / dy    # (hh[i,j] - hh[i,j-1]) / dy
        Dplusy  = (h_r - h_c) / dy    # (hh[i,j+1] - hh[i,j]) / dy

        # Upwind gradient magnitude
        del_plus = np.sqrt(
            np.maximum(Dminusx, 0)**2 + np.minimum(Dplusx, 0)**2 +
            np.maximum(Dminusy, 0)**2 + np.minimum(Dplusy, 0)**2
        )

        # Update: nhh = hh + dt * (min(del_plus, g) - del_plus)
        nhh_interior = h_c + dt * (np.minimum(del_plus, g) - del_plus)

        # Compute max change for convergence check
        max_delh = float(np.max(np.abs(nhh_interior - h_c)))

        # Update mesh size grid (interior only)
        hh[2:-2, 2:-2] = nhh_interior

    return hh
