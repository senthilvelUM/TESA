"""
Interpolate mesh size from a matrix grid.
Copyright (C) 2004-2012 Per-Olof Persson.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def hmatrix(p, xx, yy, dd, hh, *args):
    """
    Interpolate mesh size function from grid.

    Parameters
    ----------
    p  : (N, 2) query points
    xx : (ny, nx) meshgrid X coordinates
    yy : (ny, nx) meshgrid Y coordinates
    dd : unused (kept for signature compatibility)
    hh : (ny, nx) mesh size values on grid

    Returns
    -------
    h : (N,) interpolated mesh sizes
    """
    p = np.asarray(p, dtype=float)
    xx = np.asarray(xx, dtype=float)
    yy = np.asarray(yy, dtype=float)
    hh = np.asarray(hh, dtype=float)

    # Extract 1D coordinate vectors from meshgrid
    # MATLAB interp2(xx, yy, hh, px, py) uses xx as column coords, yy as row coords
    x_vec = xx[0, :]      # 1D x coordinates
    y_vec = yy[:, 0]      # 1D y coordinates

    interp = RegularGridInterpolator(
        (y_vec, x_vec), hh,
        method='linear', bounds_error=False, fill_value=None
    )

    # Query: (y, x) order for RegularGridInterpolator
    h = interp(np.column_stack([p[:, 1], p[:, 0]]))
    return h
