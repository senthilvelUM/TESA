"""
Remove duplicated/unused nodes and fix element orientation.
Copyright (C) 2004-2012 Per-Olof Persson.
"""

import numpy as np
from .simpvol import simpvol


def fixmesh(p, t=None, ptol=1024 * np.finfo(float).eps):
    """
    Remove duplicated/unused nodes and fix element orientation.

    Parameters
    ----------
    p    : (N, dim) node coordinates
    t    : (M, dim+1) element connectivity (0-based), or None
    ptol : tolerance for snapping/merging near-duplicate nodes

    Returns
    -------
    p   : cleaned node coordinates
    t   : cleaned element connectivity (if t was provided)
    pix : mapping from new node indices to original indices
    """
    p = np.asarray(p, dtype=float).copy()

    if t is not None:
        t = np.asarray(t, dtype=int).copy()
        if p.size == 0 or t.size == 0:
            return p, t, np.arange(p.shape[0])

    # Snap coordinates to grid defined by tolerance
    snap = np.max(np.max(p, axis=0) - np.min(p, axis=0)) * ptol
    if snap == 0:
        snap = ptol

    # Round to snap grid, then find unique rows (preserving order)
    rounded = np.round(p / snap) * snap
    _, ix, jx = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    # 'stable' sort: sort ix to preserve original order
    order = np.argsort(ix)
    ix_sorted = ix[order]
    # Build reverse mapping: for each unique row (in order), find its new index
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    jx_stable = inv_order[jx]

    p = p[ix_sorted, :]

    if t is not None:
        # Remap element indices
        t = jx_stable[t.ravel()].reshape(t.shape)

        # Remove unused nodes
        used = np.unique(t.ravel())
        remap = np.full(p.shape[0], -1, dtype=int)
        remap[used] = np.arange(len(used))
        t = remap[t.ravel()].reshape(t.shape)
        p = p[used, :]
        pix = ix_sorted[used]

        # Fix orientation: if signed volume < 0, swap first two vertices
        if t.shape[1] == p.shape[1] + 1:
            flip = simpvol(p, t) < 0
            t[flip, 0], t[flip, 1] = t[flip, 1].copy(), t[flip, 0].copy()

        return p, t, pix
    else:
        pix = ix_sorted
        return p, pix
