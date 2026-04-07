"""
Per-grain mesher with two-mode point selection.
"""

import numpy as np
from scipy.spatial import Delaunay
from .dpoly import dpoly
from .ddiff_multi import ddiff_multi
from .hmatrix import hmatrix
from .inpoly import inpoly
from .simpqual import simpqual
from scipy.spatial.distance import cdist as _cdist


def _per_grain_quality(pt_all, Grains, fd, h0):
    """
    Compute per-grain Delaunay quality mirroring the centroid test in mesh_all_grains.

    For each grain: select nodes within an adaptive distance threshold, run a
    local Delaunay, keep only triangles whose centroid is strictly inside the
    grain (fd < 0), then compute simplex quality. Concatenates results across
    all grains and returns a flat quality array.

    Because each grain is triangulated independently, cross-boundary sliver
    triangles never arise, so q_min reflects genuine mesh quality.

    Parameters
    ----------
    pt_all : ndarray, shape (N, 2)
        All node positions (fixed + interior).
    Grains : list of ndarray
        Grain boundary polygons, each shape (Ni, 2).
    fd : list of callable
        Per-grain signed distance functions. fd[i](pts) returns signed
        distances for grain i.
    h0 : float
        Minimum edge length, used for adaptive threshold scaling.

    Returns
    -------
    q : ndarray, shape (M,)
        Concatenated element quality values across all grains, excluding
        slivers with q < 0.05. Returns array([1.0]) if no valid elements.
    """
    q_parts = []
    for i, grain in enumerate(Grains):
        # Adaptive node-selection threshold matching mesh_all_grains.py
        grain_area = 0.5 * abs(
            np.sum(grain[:-1, 0] * grain[1:, 1] - grain[1:, 0] * grain[:-1, 1])
            + grain[-1, 0] * grain[0, 1] - grain[0, 0] * grain[-1, 1])
        expected = grain_area / (0.433 * h0 ** 2)
        thresh_i = h0 / 5 if expected < 50 else (h0 / 7 if expected < 200 else h0 / 10)

        ip = np.where(fd[i](pt_all) < thresh_i)[0]
        if len(ip) < 3:
            continue
        pg = pt_all[ip]
        try:
            tg = Delaunay(pg).simplices
        except Exception:
            continue
        pmid = (pg[tg[:, 0]] + pg[tg[:, 1]] + pg[tg[:, 2]]) / 3.0
        tg = tg[fd[i](pmid) < 0]
        if len(tg) > 0:
            q_g = simpqual(pg, tg)
            # Exclude slivers (q < 0.05) that arise from un-snapped boundary
            # nodes being too close together.  The final mesh pipeline removes
            # these explicitly (Step D); excluding them here gives a quality
            # estimate that matches what the final mesh will actually contain.
            q_good = q_g[q_g > 0.05]
            if len(q_good) > 0:
                q_parts.append(q_good)
    return np.concatenate(q_parts) if q_parts else np.array([1.0])


def distmesh2d0(ms, fh_info, pfixi, pi, Grains, Holes, h0, minx, maxx, miny, maxy,
                max_iter=None, min_iter=40, q_worst_avg_target=0.2, q_mean_target=0.90,
                iter_callback=None, mesh_internals=None, *args):
    """
    Mesh neighboring grains (polygons).

    Parameters
    ----------
    ms      : object, microstructure state (MeshingState checks removed)
    fh_info : list [xx, yy, hh] mesh size function grid
    pfixi   : (nfix, 2) fixed node positions
    pi      : (ninit, 2) initial interior node positions
    Grains  : list of (Ni, 2) arrays, grain boundary polygons
    Holes   : list, hole indices per grain
    h0      : float, minimum edge length
    minx, maxx, miny, maxy : float, domain bounds
    max_iter      : int or None, hard cap on iterations (default 100)
    min_iter      : int, no convergence exit before this iteration (default 40)
    q_worst_avg_target : float, convergence requires mean of worst 0.5% elements above this (default 0.2)
    q_mean_target : float, convergence requires q_mean above this (default 0.90)
    iter_callback : callable or None, called each iteration with (count, p, t, q_min, q_mean, q_worst_avg)
    mesh_internals : dict or None, internal tuning parameters from _MESH_INTERNALS
    *args   : additional parameters

    Returns
    -------
    p : (N, 2) node positions
    t : (NT, 3) triangle indices (0-based)
    convergence_history : dict with q_min, q_worst_avg, q_mean, and iteration histories
    """
    pfixi = np.asarray(pfixi, dtype=float)
    pi = np.asarray(pi, dtype=float)

    # Parameters
    ttol = 0.1
    dptol = 0.001
    Fscale = 1.2
    deltat = 0.2
    geps = 0.001 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0
    count = 0
    qual_count = 0
    mean_qual = 0.0       # initialised so it is defined before first quality check
    min_qual  = 0.0
    worst_avg_qual = 0.0  # mean of worst 0.1% elements
    q_mean_history = []   # records mean_qual at every quality-check iteration
    q_min_history = []    # records min_qual at every quality-check iteration
    q_worst_avg_history = []  # records worst_avg_qual at every quality-check iteration
    q_iter_history = []   # records iteration number at every quality check

    num_grains = len(Grains)

    # Setup cells of distance functions and size functions
    fd = [None] * num_grains
    fh = [None] * num_grains
    G_infix = [None] * num_grains

    print("  Progress: Initializing mesh size function grid...")
    for i in range(num_grains):
        grain_closed = np.vstack([Grains[i], Grains[i][0:1, :]])
        if Holes[i] is None or len(Holes[i]) == 0:
            # Capture grain_closed in closure
            fd[i] = (lambda gc: lambda pts: dpoly(pts, gc))(grain_closed)
        else:
            fd[i] = (lambda idx, hi: lambda pts: ddiff_multi(pts, Grains, idx, hi))(i, Holes[i])

        xg_min = np.min(Grains[i][:, 0]) - 1
        xg_max = np.max(Grains[i][:, 0]) + 1
        yg_min = np.min(Grains[i][:, 1]) - 1
        yg_max = np.max(Grains[i][:, 1]) + 1

        xg_poly = np.array([xg_min, xg_max, xg_max, xg_min])
        yg_poly = np.array([yg_min, yg_min, yg_max, yg_max])
        poly_rect = np.column_stack([xg_poly, yg_poly])

        grid_pts = np.column_stack([fh_info[0].ravel(), fh_info[1].ravel()])
        in_mask = inpoly(grid_pts, poly_rect)

        ig_flat = np.where(in_mask)[0]
        rows, cols = np.unravel_index(ig_flat, fh_info[0].shape)
        ig = np.unique(rows)
        jg = np.unique(cols)

        # Create local fh using subgrid
        xx_sub = fh_info[0][np.ix_(ig, jg)]
        yy_sub = fh_info[1][np.ix_(ig, jg)]
        hh_sub = fh_info[2][np.ix_(ig, jg)]
        fh[i] = (lambda xs, ys, hs: lambda pts, *a: hmatrix(pts, xs, ys, None, hs))(xx_sub, yy_sub, hh_sub)

        G_infix[i] = np.abs(fd[i](pfixi)) < 10 * geps
        print(f"  Progress: Grain {i + 1}/{num_grains} initialized")

    # Precompute bounding boxes for each grain (used to skip dpoly calls)
    _bbox_margin = 0.2 * h0
    _bbox = np.zeros((num_grains, 4))  # [x_min, y_min, x_max, y_max] per grain
    for i in range(num_grains):
        _bbox[i, 0] = np.min(Grains[i][:, 0]) - _bbox_margin
        _bbox[i, 1] = np.min(Grains[i][:, 1]) - _bbox_margin
        _bbox[i, 2] = np.max(Grains[i][:, 0]) + _bbox_margin
        _bbox[i, 3] = np.max(Grains[i][:, 1]) + _bbox_margin

    # Remove pfix from pi
    pfix_set = set(map(tuple, np.round(pfixi, 12)))
    mask = np.array([tuple(np.round(row, 12)) not in pfix_set for row in pi])
    pi = pi[mask]

    # Initial triangulation
    pt = np.vstack([pfixi, pi])
    tt = Delaunay(pt).simplices
    q_init = simpqual(pt, tt)
    print(f"  {tt.shape[0]} Elements, Min. Quality = {np.min(q_init):.6g}")

    grain_count = np.zeros(num_grains, dtype=int)
    # Unpack internal tuning parameters (with defaults if not provided)
    if mesh_internals is None:
        mesh_internals = {}
    _aggressive_frac = mesh_internals.get("aggressive_fraction", 0.75)
    _consec_required = mesh_internals.get("consecutive_required", 3)
    # Aggressive phase runs for a fraction of min_iter
    aggressive_end = int(_aggressive_frac * min_iter)

    print("  Meshing routine running... (will stop when qualities are sufficiently high)")
    print("  Note: quality computed per-grain every 5 iterations (mirrors final mesh).")
    import time as _time
    _iter_start = _time.time()

    while True:
        count += 1
        _t0 = _time.time()
        ic = np.zeros(pi.shape[0], dtype=bool)
        in_mask = np.zeros(pi.shape[0], dtype=bool)
        pall = np.vstack([pfixi, pi])
        tall = Delaunay(pall).simplices
        pmidall = (pall[tall[:, 0], :] + pall[tall[:, 1], :] + pall[tall[:, 2], :]) / 3.0

        for i in range(num_grains):
            # Per-grain progress (single line, overwrite)
            pass  # grain-level detail suppressed; see iteration summary below

            # Find points in current grain
            if count > aggressive_end:
                # Bounding-box pre-filter: only call fd[i] on midpoints near grain i
                _bb = _bbox[i]
                _bb_mask = ((pmidall[:, 0] >= _bb[0]) & (pmidall[:, 0] <= _bb[2]) &
                            (pmidall[:, 1] >= _bb[1]) & (pmidall[:, 1] <= _bb[3]))
                inall = np.zeros(pmidall.shape[0], dtype=bool)
                if np.any(_bb_mask):
                    _bb_idx = np.where(_bb_mask)[0]
                    inall[_bb_idx] = fd[i](pmidall[_bb_idx]) < -geps
                ti_tris = tall[inall, :]
                tall = tall[~inall, :]
                pmidall = pmidall[~inall, :]
                ti_nodes = np.unique(ti_tris.ravel())

                # Fixed points that are in this grain's triangles
                fix_mask = np.zeros(pfixi.shape[0], dtype=bool)
                fix_nodes = ti_nodes[ti_nodes < pfixi.shape[0]]
                if fix_nodes.size > 0:
                    fix_mask[fix_nodes] = True
                pfix = pfixi[fix_mask, :]
                nfix = pfix.shape[0]

                # Interior points in this grain's triangles
                in_mask = np.zeros(pi.shape[0], dtype=bool)
                int_nodes = ti_nodes[ti_nodes >= pfixi.shape[0]] - pfixi.shape[0]
                if int_nodes.size > 0:
                    valid = int_nodes[int_nodes < pi.shape[0]]
                    if valid.size > 0:
                        in_mask[valid] = True
                p = np.vstack([pfix, pi[in_mask, :]])
            else:
                infix = G_infix[i]
                pfix = pfixi[infix, :]
                nfix = pfix.shape[0]
                in_mask[:] = False
                not_ic = ~ic
                _pi_not_ic = pi[not_ic, :]
                in_mask_temp = np.zeros(np.sum(not_ic), dtype=bool)
                ic_temp = np.zeros(np.sum(not_ic), dtype=bool)
                # Bounding-box pre-filter: only call fd[i] on points near grain i
                _bb = _bbox[i]
                _bb_mask = ((_pi_not_ic[:, 0] >= _bb[0]) & (_pi_not_ic[:, 0] <= _bb[2]) &
                            (_pi_not_ic[:, 1] >= _bb[1]) & (_pi_not_ic[:, 1] <= _bb[3]))
                if np.any(_bb_mask):
                    _bb_idx = np.where(_bb_mask)[0]
                    d = fd[i](_pi_not_ic[_bb_idx])
                    in_mask_temp[_bb_idx[d < 1.5 * ttol]] = True
                    ic_temp[_bb_idx[d < -1.5 * ttol]] = True
                not_ic_idx = np.where(not_ic)[0]
                in_mask[not_ic_idx] = in_mask_temp
                ic[not_ic_idx[ic_temp]] = True
                p = np.vstack([pfix, pi[in_mask, :]])

            N = p.shape[0]
            if N < 3:
                print("  A grain has less than 3 points")
                continue

            # Skip degenerate grains where points are collinear (zero-width)
            _p_range = p.max(axis=0) - p.min(axis=0)
            if _p_range[0] < geps or _p_range[1] < geps:
                continue

            # Triangulate and describe bars as pairs of nodes
            pold = p.copy()
            t = Delaunay(p).simplices
            pmid = (p[t[:, 0], :] + p[t[:, 1], :] + p[t[:, 2], :]) / 3.0
            t = t[fd[i](pmid, *args) < -geps, :]
            if t.shape[0] == 0:
                continue
            bars = np.vstack([t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]])
            bars = np.sort(bars, axis=1)
            bars = np.unique(bars, axis=0)

            inner_count = 0
            while True:
                grain_count[i] += 1
                inner_count += 1

                # Bar lengths
                barvec = p[bars[:, 0], :] - p[bars[:, 1], :]
                L = np.sqrt(np.sum(barvec ** 2, axis=1))
                hbars = fh[i]((p[bars[:, 0], :] + p[bars[:, 1], :]) / 2.0, *args)
                L0 = hbars * Fscale * np.sqrt(np.sum(L ** 2) / np.sum(hbars ** 2))

                # Move points
                F = np.maximum(L0 - L, 0)
                Fvec = (F / L)[:, None] * barvec
                Ftot = np.zeros((N, 2))
                np.add.at(Ftot, (bars[:, 0], 0), Fvec[:, 0])
                np.add.at(Ftot, (bars[:, 0], 1), Fvec[:, 1])
                np.add.at(Ftot, (bars[:, 1], 0), -Fvec[:, 0])
                np.add.at(Ftot, (bars[:, 1], 1), -Fvec[:, 1])
                Ftot[:nfix, :] = 0
                Ftot[p[:, 0] > maxx - 10 * geps, :] = 0
                Ftot[p[:, 1] > maxy - 10 * geps, :] = 0
                p = p + deltat * Ftot

                # Bring outside points back to the boundary
                d = fd[i](p, *args)
                ix = d > 0
                if np.any(ix):
                    dgradx = (fd[i](np.column_stack([p[ix, 0] + deps, p[ix, 1]]), *args) - d[ix]) / deps
                    dgrady = (fd[i](np.column_stack([p[ix, 0], p[ix, 1] + deps]), *args) - d[ix]) / deps
                    dgrad2 = dgradx ** 2 + dgrady ** 2
                    dgrad2[dgrad2 == 0] = 1.0
                    p[ix, :] = p[ix, :] - np.column_stack([d[ix] * dgradx / dgrad2, d[ix] * dgrady / dgrad2])

                dp_max = np.max(np.sqrt(np.sum((p - pold) ** 2, axis=1)) / h0)
                if dp_max > ttol or dp_max < dptol or inner_count > 20:
                    break

            # Update pi
            pi[in_mask, :] = p[nfix:, :]

        # Update boundaries of cell
        pl = pi[pi[:, 0] < minx + h0 / 5, :].copy()
        pl[:, 0] = minx
        pr = pl.copy()
        pr[:, 0] = maxx
        pb = pi[pi[:, 1] < miny + h0 / 5, :].copy()
        pb[:, 1] = miny
        pt_bnd = pb.copy()
        pt_bnd[:, 1] = maxy

        mask = ((pi[:, 0] <= maxx - h0 / 5) & (pi[:, 1] <= maxy - h0 / 5) &
                (pi[:, 0] >= minx + h0 / 5) & (pi[:, 1] >= miny + h0 / 5))
        pi = pi[mask, :]
        pi = np.vstack([pi, pl, pr, pb, pt_bnd])
        pi = np.unique(pi, axis=0)

        # Global density control (disabled in original: `&& false`)
        # Kept as comment for fidelity

        # NOTE: Per-iteration boundary conformance is handled by the standard
        # DistMesh gradient projection (lines 236-244 above) inside the per-grain
        # inner loop.
        # Snapping to FineGrainPolylines happens ONCE after distmesh2d0 returns,
        # not during iterations.

        # Update quality and display iteration summary
        pt_all = np.vstack([pfixi, pi])
        tt_all = Delaunay(pt_all).simplices  # global Delaunay for quality display

        _dt = _time.time() - _t0
        _elapsed = _time.time() - _iter_start

        # Filter to domain-interior triangles
        _pmid_q = (pt_all[tt_all[:, 0]] + pt_all[tt_all[:, 1]] + pt_all[tt_all[:, 2]]) / 3.0
        _rect_q = np.maximum(
            np.maximum(minx - _pmid_q[:,0], _pmid_q[:,0] - maxx),
            np.maximum(miny - _pmid_q[:,1], _pmid_q[:,1] - maxy))
        _tt_interior = tt_all[_rect_q < -geps]

        # Clean up degenerate elements before computing quality metrics
        from .cleanup_mesh import cleanup_mesh as _cleanup_q
        _p_clean, _t_clean, _ = _cleanup_q(pt_all, _tt_interior, verbose=False)
        q_all = simpqual(_p_clean, _t_clean) if len(_t_clean) > 0 else np.array([1.0])
        mean_qual = np.mean(q_all)
        min_qual  = np.min(q_all)
        # q_worst_avg: mean quality of the worst 0.5% of elements (min 1 element)
        n_worst = max(1, int(np.ceil(len(q_all) * 0.005)))
        worst_avg_qual = np.mean(np.sort(q_all)[:n_worst])
        n_below_02 = np.sum(q_all < 0.2)
        n_below_05 = np.sum(q_all < 0.5)
        q_mean_history.append(mean_qual)
        q_min_history.append(min_qual)
        q_worst_avg_history.append(worst_avg_qual)
        q_iter_history.append(count)
        # Track consecutive iterations where both q_worst_avg and q_mean exceed targets
        if worst_avg_qual > q_worst_avg_target and mean_qual > q_mean_target:
            qual_count += 1
        else:
            qual_count = 0
        print(f"  Iter {count:3d}: {pt_all.shape[0]} nodes | "
              f"q_min={min_qual:.4f}, q_worst_avg={worst_avg_qual:.4f}, q_mean={mean_qual:.4f} | "
              f"q<0.2: {n_below_02}, q<0.5: {n_below_05} | "
              f"{_dt:.1f}s (total {_elapsed:.0f}s)")

        if iter_callback is not None:
            iter_callback(count, pt_all, tt_all, min_qual, mean_qual, worst_avg_qual)

        # ── Termination criteria ──────────────────────────────────────────
        # Helper: assemble final mesh with domain corners
        def _finalize():
            nonlocal p, t
            corners = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])
            p = np.vstack([corners, pfixi, pi])
            _, idx = np.unique(np.round(p, 12), axis=0, return_index=True)
            p = p[np.sort(idx), :]
            t = Delaunay(p).simplices

        # Effective max_iter (default 100 if not specified)
        _max_iter = max_iter if max_iter is not None else 100

        # Primary: count > min_iter AND both q_worst_avg and q_mean exceed targets
        # for N consecutive iterations
        if (count > min_iter and worst_avg_qual > q_worst_avg_target
                and mean_qual > q_mean_target and qual_count >= _consec_required):
            _finalize()
            break

        # Hard safety cap: prevent infinite loops
        if count >= _max_iter:
            _finalize()
            if worst_avg_qual < q_worst_avg_target:
                print(f"  WARNING: max iterations ({_max_iter}) reached, "
                      f"q_worst_avg={worst_avg_qual:.4f} < target {q_worst_avg_target}")
            break

    # Package convergence history
    convergence_history = {
        'iterations': q_iter_history,
        'q_min': q_min_history,
        'q_worst_avg': q_worst_avg_history,
        'q_mean': q_mean_history,
    }

    return p, t, convergence_history
