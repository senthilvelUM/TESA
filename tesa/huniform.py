"""
Uniform mesh size function.
Copyright (C) 2004-2012 Per-Olof Persson.
"""

import numpy as np


def huniform(p, *args, **kwargs):
    """
    Return uniform mesh size of 1.0 at every query point.

    Parameters
    ----------
    p : array_like, shape (N, 2)
        Query points (coordinates are unused).
    *args : tuple
        Ignored (accepted for signature compatibility).
    **kwargs : dict
        Ignored (accepted for signature compatibility).

    Returns
    -------
    h : ndarray, shape (N,)
        Array of ones.
    """
    p = np.asarray(p, dtype=float)
    return np.ones(p.shape[0])
