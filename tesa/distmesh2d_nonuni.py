"""
2-D Mesh Generator using Distance Functions (non-uniform variant).
Copyright (C) 2004-2012 Per-Olof Persson.
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from .fixmesh import fixmesh
from .simpqual import simpqual


def distmesh2d_nonuni(fd, fh, h0, bbox, pfix, *args):
    """
    2-D mesh generator using distance functions (non-uniform).

    Parameters
    ----------
    fd    : callable, distance function fd(p, *args)
    fh    : callable, scaled edge length function fh(p, *args)
    h0    : float, initial edge length
    bbox  : (2, 2) bounding box [[xmin, ymin], [xmax, ymax]]
    pfix  : (nfix, 2) fixed node positions or empty array
    *args : additional parameters passed to fd and fh

    Returns
    -------
    p : (N, 2) node positions
    t : (NT, 3) triangle indices (0-based)
    """
    bbox = np.asarray(bbox, dtype=float)
    pfix = np.asarray(pfix, dtype=float).reshape(-1, 2) if pfix is not None and len(pfix) > 0 else np.zeros((0, 2))

    dptol = 0.001
    ttol = 0.1
    Fscale = 1.2
    deltat = 0.2
    geps = 0.001 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0
    densityctrlfreq = 30
    max_iter = 208

    # 1. Create initial distribution in bounding box (equilateral triangles)
    x_vals = np.arange(bbox[0, 0], bbox[1, 0] + h0 / 2, h0)
    y_vals = np.arange(bbox[0, 1], bbox[1, 1] + h0 * np.sqrt(3) / 4, h0 * np.sqrt(3) / 2)
    xx, yy = np.meshgrid(x_vals, y_vals)
    # Shift even rows
    xx[1::2, :] = xx[1::2, :] + h0 / 2
    p = np.column_stack([xx.ravel(), yy.ravel()])

    # 2. Remove points outside the region, apply the rejection method
    p = p[fd(p, *args) < geps, :]
    r0 = 1.0 / fh(p, *args) ** 2
    p = p[np.random.rand(p.shape[0]) < r0 / np.max(r0), :]

    if pfix.shape[0] > 0:
        # Remove duplicated nodes (those in pfix)
        pfix_set = set(map(tuple, np.round(pfix, 12)))
        mask = np.array([tuple(np.round(row, 12)) not in pfix_set for row in p])
        p = p[mask]

    pfix = np.unique(pfix, axis=0)
    nfix = pfix.shape[0]
    p = np.vstack([pfix, p])
    N = p.shape[0]

    count = 0
    pold = np.full_like(p, np.inf)

    print("  Progress: Refining the FE mesh...")

    while True:
        count += 1
        print(f"  Progress: {count}/{max_iter} iterations")

        # 3. Retriangulation by the Delaunay algorithm
        if np.max(np.sqrt(np.sum((p - pold) ** 2, axis=1)) / h0) > ttol:
            pold = p.copy()
            t = Delaunay(p).simplices  # 0-based
            pmid = (p[t[:, 0], :] + p[t[:, 1], :] + p[t[:, 2], :]) / 3.0
            t = t[fd(pmid, *args) < -geps, :]
            # 4. Describe each bar by a unique pair of nodes
            bars = np.vstack([t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]])
            bars = np.sort(bars, axis=1)
            bars = np.unique(bars, axis=0)

        # 6. Move mesh points based on bar lengths L and forces F
        barvec = p[bars[:, 0], :] - p[bars[:, 1], :]
        L = np.sqrt(np.sum(barvec ** 2, axis=1))
        hbars = fh((p[bars[:, 0], :] + p[bars[:, 1], :]) / 2.0, *args)
        L0 = hbars * Fscale * np.sqrt(np.sum(L ** 2) / np.sum(hbars ** 2))

        # Density control - remove points that are too close
        if count % densityctrlfreq == 0 and np.any(L0 > 2 * L):
            remove_idx = np.unique(bars[L0 > 2 * L, :].ravel())
            remove_idx = np.setdiff1d(remove_idx, np.arange(nfix))
            if remove_idx.size > 0:
                p = np.delete(p, remove_idx, axis=0)
                N = p.shape[0]
                pold = np.full_like(p, np.inf)
                continue

        F = np.maximum(L0 - L, 0)
        Fvec = (F / L)[:, None] * barvec

        # Build force matrix using sparse accumulation
        rows = np.concatenate([bars[:, 0], bars[:, 0], bars[:, 1], bars[:, 1]])
        cols = np.concatenate([np.zeros(len(F), dtype=int), np.ones(len(F), dtype=int),
                               np.zeros(len(F), dtype=int), np.ones(len(F), dtype=int)])
        vals = np.concatenate([Fvec[:, 0], Fvec[:, 1], -Fvec[:, 0], -Fvec[:, 1]])
        Ftot = np.zeros((N, 2))
        np.add.at(Ftot, (rows, cols), vals)

        Ftot[:nfix, :] = 0  # Force = 0 at fixed points
        p = p + deltat * Ftot  # Update node positions

        # 7. Bring outside points back to the boundary
        d = fd(p, *args)
        ix = d > 0
        if np.any(ix):
            dgradx = (fd(np.column_stack([p[ix, 0] + deps, p[ix, 1]]), *args) - d[ix]) / deps
            dgrady = (fd(np.column_stack([p[ix, 0], p[ix, 1] + deps]), *args) - d[ix]) / deps
            dgrad2 = dgradx ** 2 + dgrady ** 2
            dgrad2[dgrad2 == 0] = 1.0  # avoid division by zero
            p[ix, :] = p[ix, :] - np.column_stack([d[ix] * dgradx / dgrad2, d[ix] * dgrady / dgrad2])

        # 8. Termination criterion
        interior = d < -geps
        if interior.any():
            max_move = np.max(np.sqrt(np.sum((deltat * Ftot[interior, :]) ** 2, axis=1)) / h0)
        else:
            max_move = 0.0
        if max_move < dptol or count >= max_iter:
            break

    # Clean up and report final mesh
    p, t, _ = fixmesh(p, t)
    q = simpqual(p, t)
    print(f"Minimum element quality = {np.min(q):.6g}")
    return p, t
