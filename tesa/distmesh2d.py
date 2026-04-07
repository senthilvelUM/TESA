"""
2-D Mesh Generator using Distance Functions (for initial mesh creation).
Copyright (C) 2004-2012 Per-Olof Persson.
"""

import numpy as np
from scipy.spatial import Delaunay
from .simpqual import simpqual


def distmesh2d(ms, fd, fdu, fh, h0, bbox, pfix, pinit, type_, Grains, Holes, *args):
    """
    2-D mesh generator using distance functions.

    Parameters
    ----------
    ms     : object, microstructure state (used for GrainColors; MeshingState checks removed)
    fd     : callable, distance function fd(p, *args)
    fdu    : callable, distance function for union fd(p, *args)
    fh     : callable, scaled edge length function fh(p, *args)
    h0     : float, initial edge length
    bbox   : (2, 2) bounding box [[xmin, ymin], [xmax, ymax]]
    pfix   : (nfix, 2) fixed node positions
    pinit  : (ninit, 2) initial interior node positions
    type_  : int, mesh type (3 = periodic)
    Grains : list of (Ni, 2) arrays, grain boundary polygons
    Holes  : list, hole indices per grain
    *args  : additional parameters passed to fd and fh

    Returns
    -------
    p : (N, 2) node positions
    t : (NT, 3) triangle indices (0-based)
    """
    bbox = np.asarray(bbox, dtype=float)
    pfix = np.asarray(pfix, dtype=float).reshape(-1, 2) if pfix is not None and len(pfix) > 0 else np.zeros((0, 2))
    pinit = np.asarray(pinit, dtype=float).reshape(-1, 2) if pinit is not None and len(pinit) > 0 else np.zeros((0, 2))

    dptol = 0.05
    ttol = 0.5
    Fscale = 1.2
    deltat = 0.2
    geps = 0.001 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0
    densityctrlfreq = 30
    minx = bbox[0, 0]
    maxx = bbox[1, 0]
    miny = bbox[0, 1]
    maxy = bbox[1, 1]

    # GUI setup replaced with print
    print("  Creating initial mesh...")

    # Initial triangulation for display (removed GUI patch)
    pt = np.vstack([pfix, pinit])
    tt = Delaunay(pt).simplices
    q_init = simpqual(pt, tt)
    print(f"  {tt.shape[0]} Elements, Min. Quality = {np.min(q_init):.6g}")

    # 1. Create initial distribution
    p = np.unique(pinit, axis=0)
    if pfix.shape[0] > 0:
        pfix_set = set(map(tuple, np.round(pfix, 12)))
        mask = np.array([tuple(np.round(row, 12)) not in pfix_set for row in p])
        p = p[mask]
    pfix = np.unique(pfix, axis=0)
    nfix = pfix.shape[0]
    p = np.vstack([pfix, p])
    N = p.shape[0]
    count = 0
    qual_count = 0
    pold = np.full_like(p, np.inf)
    q = np.array([0.0])  # initialize quality

    print("  Progress: Creating initial mesh (do not press ctrl-c)")

    while True:
        count += 1

        # 3. Retriangulation by the Delaunay algorithm
        if np.max(np.sqrt(np.sum((p - pold) ** 2, axis=1)) / h0) > ttol:
            pold = p.copy()
            t = Delaunay(p).simplices
            pmid = (p[t[:, 0], :] + p[t[:, 1], :] + p[t[:, 2], :]) / 3.0
            q = simpqual(p, t[fd(pmid, *args) < -geps, :])
            t = t[fdu(pmid, *args) < -geps, :]
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

        # Build force totals
        Ftot = np.zeros((N, 2))
        np.add.at(Ftot, (bars[:, 0], 0), Fvec[:, 0])
        np.add.at(Ftot, (bars[:, 0], 1), Fvec[:, 1])
        np.add.at(Ftot, (bars[:, 1], 0), -Fvec[:, 0])
        np.add.at(Ftot, (bars[:, 1], 1), -Fvec[:, 1])

        Ftot[:nfix, :] = 0  # Force = 0 at fixed points
        if type_ == 3:
            Ftot[p[:, 0] > maxx - 2 * geps, :] = 0
            Ftot[p[:, 1] > maxy - 2 * geps, :] = 0
        p = p + deltat * Ftot
        p[:nfix, :] = pfix

        # 7. Bring outside points back to the boundary
        d = fd(p, *args)
        ix = d > 0
        if np.any(ix):
            dgradx = (fd(np.column_stack([p[ix, 0] + deps, p[ix, 1]]), *args) - d[ix]) / deps
            dgrady = (fd(np.column_stack([p[ix, 0], p[ix, 1] + deps]), *args) - d[ix]) / deps
            dgrad2 = dgradx ** 2 + dgrady ** 2
            dgrad2[dgrad2 == 0] = 1.0
            p[ix, :] = p[ix, :] - np.column_stack([d[ix] * dgradx / dgrad2, d[ix] * dgrady / dgrad2])
        p = np.vstack([pfix, p[nfix:, :]])

        # Match boundaries
        pd = p.copy()
        if count > 5:
            pl = p[p[:, 0] < minx + h0 / 5, :].copy()
            pl[:, 0] = maxx
            pb = p[p[:, 1] < miny + h0 / 5, :].copy()
            pb[:, 1] = maxy
            mask = (p[:, 0] <= maxx - h0 / 5) & (p[:, 1] <= maxy - h0 / 5)
            p = p[mask, :]
            p = np.vstack([p, pl, pb])
            p = np.unique(p, axis=0)
            p[p[:, 0] < minx + h0 / 5, 0] = 0
            p[p[:, 1] < miny + h0 / 5, 1] = 0
            # Re-prepend pfix
            pfix_set = set(map(tuple, np.round(pfix, 12)))
            mask = np.array([tuple(np.round(row, 12)) not in pfix_set for row in p])
            p = np.vstack([pfix, p[mask]])
            N = p.shape[0]
            pold = np.full_like(p, np.inf)

        # 8. Termination criterion
        d = fd(pd, *args)
        interior = d < -geps
        if interior.any():
            err_val = np.max(np.sqrt(np.sum((deltat * Ftot[interior, :]) ** 2, axis=1)) / h0)
        else:
            err_val = 0.0
        if count % 50 == 0 or count <= 5:
            print(f"  Initial mesh: iter {count}, error={err_val:.4g}, q_min={np.min(q):.4f}")

        # MeshingState == 2 check removed (was for GUI stop button)

        # Termination for type==3 based on quality thresholds
        if type_ == 3:
            def _match_boundaries():
                """Helper to apply boundary matching."""
                nonlocal p, N, pold
                pl2 = p[p[:, 0] < minx + h0 / 5, :].copy()
                pl2[:, 0] = maxx
                pb2 = p[p[:, 1] < miny + h0 / 5, :].copy()
                pb2[:, 1] = maxy
                mask2 = (p[:, 0] <= maxx - h0 / 5) & (p[:, 1] <= maxy - h0 / 5)
                p2 = p[mask2, :]
                p2 = np.vstack([p2, pl2, pb2])
                p2 = np.unique(p2, axis=0)
                p2[p2[:, 0] < minx + h0 / 5, 0] = 0
                p2[p2[:, 1] < miny + h0 / 5, 1] = 0
                return p2

            if np.min(q) > 0.5 and count > 300 and err_val < dptol:
                p = _match_boundaries()
                print(f"  {count} iterations")
                print(f"  Error = {err_val:.6g}")
                print(f"  Min. Quality = {np.min(q):.6g}")
                break
            if np.min(q) > 0.4 and count > 400:
                p = _match_boundaries()
                print(f"  {count} iterations")
                print(f"  Error = {err_val:.6g}")
                print(f"  Min. Quality = {np.min(q):.6g}")
                break
            if np.min(q) > 0.5 and count > 100 and err_val < dptol:
                p = _match_boundaries()
                print(f"  Converged: {count} iterations, q_min={np.min(q):.4f}, error={err_val:.4g}")
                break
            if np.min(q) > 0.2 and count > 200:
                p = _match_boundaries()
                print(f"  Converged: {count} iterations, q_min={np.min(q):.4f}")
                break
            # Hard cap: initial mesh is refined later by per-grain distmesh2d0
            if count >= 300:
                p = _match_boundaries()
                print(f"  Hard cap: {count} iterations, q_min={np.min(q):.4f}")
                break

    # Final retriangulation: boundary matching may have changed p,
    # making the old t stale.  Recompute Delaunay + centroid filter.
    t = Delaunay(p).simplices
    pmid = (p[t[:, 0], :] + p[t[:, 1], :] + p[t[:, 2], :]) / 3.0
    t = t[fdu(pmid, *args) < -geps, :]

    print("  Progress: Done")
    return p, t
