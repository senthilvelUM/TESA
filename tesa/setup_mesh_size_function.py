"""
Mesh all grains in microstructure and return triangulation.
"""

import numpy as np

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from .find_grain_holes import find_grain_holes
from .polygonLength import polygonLength
from .resamplePolygon import resamplePolygon
from .polylineLength import polylineLength
from .resamplePolyline import resamplePolyline
from .dpoly import dpoly
from .ddiff_multi import ddiff_multi
from .plfs import plfs
from .knnsearch2 import knnsearch2
from .LineCurvature2D import LineCurvature2D
from .gradient_limiting import gradient_limiting
from .inpoly import inpoly


def setup_mesh_size_function(ms, verbose=False):
    """
    Set up the mesh size function for the microstructure.

    Computes a spatially varying mesh size function based on grain boundary
    curvature and medial-axis feature size. Also resamples grain boundary
    polylines and identifies junction points. The result is stored as a
    regular grid in ms.MeshSizeFunctionGrid.

    Parameters
    ----------
    ms : Microstructure
        Microstructure state with fields: GrainsSmoothed, DisplayData,
        HoleGrains, GrainHoles, GrainMedialAxis, MeshParameters,
        GrainPolylines, FineGrainPolylines, GrainJunctionPoints,
        MeshSizeFunctionGrid, MeshSizeFunctionGridLimits.
    verbose : bool
        If True, print per-grain progress to console.

    Returns
    -------
    ms : Microstructure
        Updated with MeshSizeFunctionGrid, MeshSizeFunctionGridLimits,
        GrainJunctionPoints, FineGrainPolylines, GrainPolylines, and
        GrainsSmoothed.
    """
    import warnings
    warnings.filterwarnings('ignore')

    # Check if data exists
    Gs = ms.GrainsSmoothed
    if ms.DisplayData[0] is None or len(ms.DisplayData[0]) == 0:
        print("Error: No data exists")
        return

    # Find data limits, boundary points, and junction points
    minx = 1e6
    maxx = 0
    miny = 1e6
    maxy = 0
    for i in range(len(Gs)):
        x = Gs[i][:, 0]
        y = Gs[i][:, 1]
        minx = min(np.min(x), minx)
        miny = min(np.min(y), miny)
        maxx = max(np.max(x), maxx)
        maxy = max(np.max(y), maxy)

    bl = np.zeros((0, 2))
    bb = np.zeros((0, 2))
    apts = np.zeros((0, 2))
    eps_val = np.sqrt(np.finfo(float).eps)

    for i in range(len(Gs)):
        _, id_idx = np.unique(Gs[i], axis=0, return_index=True)
        id_idx = np.sort(id_idx)
        Gs[i] = Gs[i][id_idx, :]
        x = Gs[i][:, 0]
        y = Gs[i][:, 1]
        apts = np.vstack([apts, np.column_stack([x, y])])
        mask_l = x < minx + eps_val
        if np.any(mask_l):
            bl = np.vstack([bl, np.column_stack([x[mask_l], y[mask_l]])])
        mask_b = y < miny + eps_val
        if np.any(mask_b):
            bb = np.vstack([bb, np.column_stack([x[mask_b], y[mask_b]])])

    if bl.shape[0] > 0:
        bl = np.unique(bl, axis=0)
    if bb.shape[0] > 0:
        bb = np.unique(bb, axis=0)

    # Snap points on right/top to those on left/bottom
    for i in range(len(Gs)):
        if not np.any(Gs[i][:, 0] > maxx - eps_val):
            continue
        cg = Gs[i].copy()
        cg[Gs[i][:, 0] < maxx - eps_val, 0] = 1e10
        cb = np.column_stack([bl[:, 0] + maxx, bl[:, 1]])
        if cb.shape[0] > 0 and cg.shape[0] > 0:
            D = cdist(cb, cg)
            id_nearest = np.argmin(D, axis=0)
            d_nearest = D[id_nearest, np.arange(cg.shape[0])]
            do = d_nearest < 1e10 / 2
            Gs[i][do, :] = cb[id_nearest[do], :]
            _, id_idx = np.unique(Gs[i], axis=0, return_index=True)
            id_idx = np.sort(id_idx)
            Gs[i] = Gs[i][id_idx, :]

    for i in range(len(Gs)):
        if not np.any(Gs[i][:, 1] > maxy - eps_val):
            continue
        cg = Gs[i].copy()
        cg[Gs[i][:, 1] < maxy - eps_val, 0] = 1e10
        cb = np.column_stack([bb[:, 0], bb[:, 1] + maxy])
        if cb.shape[0] > 0 and cg.shape[0] > 0:
            D = cdist(cb, cg)
            id_nearest = np.argmin(D, axis=0)
            d_nearest = D[id_nearest, np.arange(cg.shape[0])]
            do = d_nearest < 1e10 / 2
            Gs[i][do, :] = cb[id_nearest[do], :]
            _, id_idx = np.unique(Gs[i], axis=0, return_index=True)
            id_idx = np.sort(id_idx)
            Gs[i] = Gs[i][id_idx, :]

    # Find holes
    if ms.HoleGrains is None or len(ms.HoleGrains) == 0:
        Holes, holes = find_grain_holes(Gs)
        ms.GrainHoles = Holes
        ms.HoleGrains = holes
    else:
        Holes = ms.GrainHoles
        holes = ms.HoleGrains

    # Find junction points
    apts = np.zeros((0, 2))
    for i in range(len(Gs)):
        x = Gs[i][:, 0]
        y = Gs[i][:, 1]
        apts = np.vstack([apts, np.column_stack([x, y])])

    c1 = apts[:, 0] < eps_val
    c2 = apts[:, 0] > maxx - eps_val
    c3 = apts[:, 1] < eps_val
    c4 = apts[:, 1] > maxy - eps_val
    bpts = apts[c1 | c2 | c3 | c4, :]

    # Find duplicates (points appearing more than once)
    _, id_idx = np.unique(bpts, axis=0, return_index=True)
    mask = np.ones(bpts.shape[0], dtype=bool)
    mask[id_idx] = False
    bpts = bpts[mask, :]

    corners = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])
    if bpts.shape[0] > 0:
        corner_set = set(map(tuple, np.round(corners, 12)))
        bpts_mask = np.array([tuple(np.round(row, 12)) not in corner_set for row in bpts])
        bpts = bpts[bpts_mask]
    bpts = np.vstack([np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]),
                       np.unique(bpts, axis=0)]) if bpts.shape[0] > 0 else \
        np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])

    # Triple junction points (appearing 3+ times)
    jpts = apts.copy()
    _, id_idx = np.unique(jpts, axis=0, return_index=True)
    mask = np.ones(jpts.shape[0], dtype=bool)
    mask[id_idx] = False
    jpts = jpts[mask, :]
    _, id_idx = np.unique(jpts, axis=0, return_index=True)
    mask = np.ones(jpts.shape[0], dtype=bool)
    mask[id_idx] = False
    jpts = jpts[mask, :]
    jpts = np.unique(jpts, axis=0)

    # Add boundary points and their mirrors
    blpts = bpts[bpts[:, 0] < minx + eps_val, :]
    if blpts.shape[0] > 0:
        blpts = np.vstack([blpts, np.column_stack([blpts[:, 0] + maxx, blpts[:, 1]])])
    brpts = bpts[bpts[:, 0] > maxx - eps_val, :]
    if brpts.shape[0] > 0:
        brpts = np.vstack([brpts, np.column_stack([brpts[:, 0] - maxx, brpts[:, 1]])])
    bbpts = bpts[bpts[:, 1] < miny + eps_val, :]
    if bbpts.shape[0] > 0:
        bbpts = np.vstack([bbpts, np.column_stack([bbpts[:, 0], bbpts[:, 1] + maxy])])
    btpts = bpts[bpts[:, 1] > maxy - eps_val, :]
    if btpts.shape[0] > 0:
        btpts = np.vstack([btpts, np.column_stack([btpts[:, 0], btpts[:, 1] - maxy])])

    jpts_parts = [jpts]
    for arr in [blpts, brpts, bbpts, btpts]:
        if arr.shape[0] > 0:
            jpts_parts.append(arr)
    jpts = np.unique(np.vstack(jpts_parts), axis=0) if len(jpts_parts) > 1 else jpts

    # Rearrange grains vertices such that a junction point is first
    jpts_set = set(map(tuple, np.round(jpts, 12)))
    for i in range(len(Gs)):
        member = np.array([tuple(np.round(row, 12)) in jpts_set for row in Gs[i]])
        id_found = np.where(member)[0]
        if len(id_found) == 0:
            continue
        id_first = id_found[0]
        if id_first == 0:
            continue
        Gs[i] = np.vstack([Gs[i][id_first:, :], Gs[i][:id_first, :]])

    # Find the polylines in each grain
    # IMPORTANT: Shared boundary segments between adjacent grains must use
    # identical resampled points.  We build a registry keyed by the sorted
    # (rounded) junction-point pair.  The first grain to encounter a segment
    # resamples it; the second grain retrieves the cached version (flipped
    # if the traversal direction is reversed).  This guarantees that the
    # snapping step in mesh_all_grains produces coincident nodes on shared
    # boundaries, eliminating white gaps.
    mnpl = 100
    Gfpl = [[None] * mnpl for _ in range(len(Gs))]
    Gpl = [[None] * mnpl for _ in range(len(Gs))]
    rsep = 0.05
    _seg_cache = {}  # (key_start, key_end) sorted → (Gfpl_segment, start_key, end_key)

    for i in range(len(Gs)):
        if verbose:
            print(f"  Finding grain polylines {i} {len(Gs)} complete")
        if holes is not None and i in holes:
            Gpl[i][0] = Gs[i].copy()
            length = polygonLength(Gs[i])
            Gpoly = resamplePolygon(Gs[i], max(3, round(length / rsep)))
            Gfpl[i][0] = 1e-6 * np.round(1e6 * Gpoly)
            continue

        Gs_closed = np.vstack([Gs[i], Gs[i][0:1, :]])
        member = np.array([tuple(np.round(row, 12)) in jpts_set for row in Gs_closed])
        jd = np.where(member)[0]
        for j in range(len(jd) - 1):
            Gpl[i][j] = Gs_closed[jd[j]:jd[j + 1] + 1, :]

            # Build a direction-independent cache key from the two junction points
            pt_start = tuple(np.round(Gs_closed[jd[j], :], 8))
            pt_end = tuple(np.round(Gs_closed[jd[j + 1], :], 8))
            cache_key = (min(pt_start, pt_end), max(pt_start, pt_end))

            if cache_key in _seg_cache:
                # Retrieve cached segment — flip if traversal direction is reversed
                cached_seg, cached_start, cached_end = _seg_cache[cache_key]
                if pt_start == cached_start:
                    Gfpl[i][j] = cached_seg.copy()
                else:
                    Gfpl[i][j] = cached_seg[::-1, :].copy()
            else:
                # First time seeing this segment — resample and cache
                Gpoly = Gs_closed[jd[j]:jd[j + 1] + 1, :]
                length = polylineLength(Gpoly)
                n_pts = max(2, round(length / rsep))
                Gpoly = resamplePolyline(Gpoly, n_pts)
                Gpoly[0, :] = Gs_closed[jd[j], :]
                Gpoly[-1, :] = Gs_closed[jd[j + 1], :]
                seg = np.vstack([Gpoly[0:1, :],
                                 1e-6 * np.round(1e6 * Gpoly[1:-1, :]),
                                 Gpoly[-1:, :]])
                Gfpl[i][j] = seg
                _seg_cache[cache_key] = (seg, pt_start, pt_end)

    if verbose:
        print(f"  Finding grain polylines {len(Gs)} {len(Gs)} complete")
    if verbose:
        print(f"  Polyline segment cache: {len(_seg_cache)} unique segments")
    ms.FineGrainPolylines = Gfpl
    ms.GrainPolylines = Gpl

    # Get mesh size function parameters
    K = ms.MeshParameters[0]
    R = ms.MeshParameters[1]
    g = ms.MeshParameters[2]
    l = ms.MeshParameters[3]

    # Setup mesh size function
    if ms.GrainMedialAxis is None:
        ms.GrainMedialAxis = [None] * len(Gs)

    # Grid resolution for mesh size function computation
    # Capped at 300 — larger grids don't improve mesh quality
    # but dramatically increase computation time
    igrid_sizex = 300
    igrid_sizey = round(igrid_sizex * ((maxy - miny) / (maxx - minx)))
    remx = igrid_sizex % round(maxx - minx) if round(maxx - minx) > 0 else 0
    igrid_sizex = round(igrid_sizex - remx)
    remy = igrid_sizey % round(maxy - miny) if round(maxy - miny) > 0 else 0
    igrid_sizey = round(igrid_sizey - remy)

    xsep = (maxx - minx) / igrid_sizex if igrid_sizex > 0 else 1.0
    ysep = (maxy - miny) / igrid_sizey if igrid_sizey > 0 else 1.0
    x_range = np.arange(minx - 6 * xsep, maxx + 6 * xsep + xsep / 2, xsep)
    y_range = np.arange(miny - 6 * ysep, maxy + 6 * ysep + ysep / 2, ysep)
    xx, yy = np.meshgrid(x_range, y_range)
    hhcurv = np.full(xx.shape, np.inf)
    hhlfs = np.full(xx.shape, np.inf)
    all_pma = np.zeros((0, 2))
    gamma = 1.6
    kappatol = 0.6

    if verbose:
        import time as _time_msf
        print(f"  Creating mesh size function (grid {xx.shape[0]}×{xx.shape[1]}, {len(Gs)} grains)...")

    for i in range(len(Gs)):
        if verbose:
            _t_grain_msf = _time_msf.time()
            _n_grain_pts = Gs[i].shape[0]
            print(f"\r    Grain {i + 1}/{len(Gs)} ({_n_grain_pts} bdy pts)...        ", end='', flush=True)

        # Check grain size
        grain_area = 0.5 * np.abs(np.sum(Gs[i][:-1, 0] * Gs[i][1:, 1] - Gs[i][1:, 0] * Gs[i][:-1, 1]) +
                                    Gs[i][-1, 0] * Gs[i][0, 1] - Gs[i][0, 0] * Gs[i][-1, 1])

        # Setup computation grid
        if verbose:
            pass  # detailed step logging removed (was: "Setting up computation grid")
        gminx = round(np.min(Gs[i][:, 0]))
        gmaxx = round(np.max(Gs[i][:, 0]))
        gminy = round(np.min(Gs[i][:, 1]))
        gmaxy = round(np.max(Gs[i][:, 1]))

        mask_grid = ((xx > gminx - eps_val) & (xx < gmaxx + eps_val) &
                      (yy > gminy - eps_val) & (yy < gmaxy + eps_val))
        ig_flat = np.where(mask_grid)
        if len(ig_flat[0]) == 0:
            continue
        ig_rows = ig_flat[0]
        ig_cols = ig_flat[1]
        ig = np.arange(max(0, np.min(ig_rows) - 6), min(xx.shape[0], np.max(ig_rows) + 7))
        jg = np.arange(max(0, np.min(ig_cols) - 6), min(xx.shape[1], np.max(ig_cols) + 7))

        xxg = xx[np.ix_(ig, jg)]
        yyg = yy[np.ix_(ig, jg)]

        # Compute distance function at grid points
        rows = xxg.shape[0]
        cols = xxg.shape[1]
        if Holes[i] is None or len(Holes[i]) == 0:
            fd_i = lambda pts: dpoly(pts, np.vstack([Gs[i], Gs[i][0:1, :]]))
        else:
            fd_i = lambda pts, idx=i, hi=Holes[i]: ddiff_multi(pts, Gs, idx, hi)

        # Compute distances in chunks (replacing parfor with regular for)
        xt = xxg.ravel()
        yt = yyg.ravel()
        nt = xt.shape[0]
        chunk_size = max(1, int(np.ceil(nt / 12)))
        if verbose:
            pass  # detailed step logging removed (was: "Computing distances in loop")
        ddg_parts = []
        for iPoint in range(12):
            start = iPoint * chunk_size
            end = min((iPoint + 1) * chunk_size, nt)
            if start >= nt:
                break
            ddg_parts.append(fd_i(np.column_stack([xt[start:end], yt[start:end]])))
        ddg = np.concatenate(ddg_parts)
        ddg = ddg.reshape(rows, cols)

        # Compute the adjusted curvature (applied to all grains; effect is clipped
        # to a thin strip near the grain boundary by the 2*min(dx,dy) cutoff below)
        if verbose:
            pass  # detailed step logging removed (was: "Computing curvatures")
        dx_vals = np.sort(np.unique(xxg.ravel()))
        dx = abs(dx_vals[1] - dx_vals[0]) if len(dx_vals) > 1 else 1.0
        dy_vals = np.sort(np.unique(yyg.ravel()))
        dy = abs(dy_vals[1] - dy_vals[0]) if len(dy_vals) > 1 else 1.0
        hhgcurv = np.full(xxg.shape, np.inf)

        Gsc = np.vstack([Gs[i][-1:, :], Gs[i], Gs[i][0:1, :]])
        cs = LineCurvature2D(Gsc)
        Gsc = Gsc[1:-1, :]
        cs = cs[1:-1]

        # Handle junction points
        jpts_member = np.array([tuple(np.round(row, 12)) in jpts_set for row in Gsc])
        if np.any(jpts_member):
            jpts_idx = np.where(jpts_member)[0]
            D = cdist(Gs[i], Gsc[jpts_idx, :])
            ic = np.argsort(D, axis=0)[:2, :].T
            ic_second = ic[:, 1]
            cs[jpts_idx] = cs[ic_second]

        kd = np.arange(xxg.size)

        if kd.size > 0:
            pts_kd = np.column_stack([xxg.ravel()[kd], yyg.ravel()[kd]])
            # Use cKDTree for O(N log M) nearest-neighbor instead of O(N*M) cdist
            _tree_grain = cKDTree(Gs[i])
            _, id_nearest = _tree_grain.query(pts_kd)
            hhgcurv_flat = hhgcurv.ravel().copy()
            hhgcurv_flat[kd] = np.minimum(hhgcurv_flat[kd], cs[id_nearest])
            hhgcurv = np.abs(hhgcurv_flat.reshape(rows, cols))
            hhgcurv = np.abs((1 + hhgcurv * ddg) / (K * hhgcurv)) - g * ddg
            hhgcurv_flat = hhgcurv.ravel()
            hhgcurv_flat[np.abs(ddg.ravel()) > 2 * min(dx, dy)] = np.inf
            hhgcurv = hhgcurv_flat.reshape(rows, cols)
        hhcurv[np.ix_(ig, jg)] = np.minimum(hhcurv[np.ix_(ig, jg)], hhgcurv)

        # Locate medial axis points
        if verbose:
            pass  # detailed step logging removed (was: "Locating medial axis")
        if ms.GrainMedialAxis[i] is None:
            # Use coarser grid for large grains (>200 boundary pts) — medial axis
            # only needs rough approximation for element sizing
            _plfs_res = 25 if Gs[i].shape[0] > 200 else 50
            xxtg, yytg = np.meshgrid(
                np.linspace(np.min(xxg) - 2, np.max(xxg) + 2, _plfs_res),
                np.linspace(np.min(yyg) - 2, np.max(yyg) + 2, _plfs_res))
            pma = plfs(xxtg, yytg, gamma, kappatol, Gs, i, Holes[i] if Holes[i] is not None else [])
            ms.GrainMedialAxis[i] = pma
        else:
            pma = ms.GrainMedialAxis[i]

        if pma is None or len(pma) == 0:
            pma = np.mean(Gs[i], axis=0, keepdims=True)
            pts_flat = np.column_stack([xxg.ravel(), yyg.ravel()])
            D = cdist(pts_flat, pma)
            id_nearest = np.argmin(D.ravel())
            # hhlfs(id) = 1/3 -- simplified approach
            hhlfs.ravel()[id_nearest] = 1.0 / 3.0

        if verbose:
            pass  # detailed step logging removed (was: "Computing distances to medial axis")
        all_pma = np.vstack([all_pma, pma]) if pma is not None and len(pma) > 0 else all_pma

        # Find distance from each grid point to nearest medial axis point
        grid_pts = np.column_stack([xxg.ravel(), yyg.ravel()])
        if pma is not None and len(pma) > 0:
            # Use cKDTree for O(N log M) nearest-neighbor instead of brute-force knnsearch2
            _tree_pma = cKDTree(pma)
            ddma, _ = _tree_pma.query(grid_pts)
            ddma = ddma.reshape(xxg.shape[0], xxg.shape[1])
        else:
            ddma = np.full(xxg.shape, np.inf)

        # Compute the feature size
        if verbose:
            pass  # detailed step logging removed (was: "Computing feature size")
        hhlfs[np.ix_(ig, jg)] = np.minimum(hhlfs[np.ix_(ig, jg)],
                                             (np.abs(ddg) + np.abs(ddma)) / R)

        if verbose:
            _elapsed_grain_msf = _time_msf.time() - _t_grain_msf
            print(f"\r    Grain {i + 1}/{len(Gs)} ({_n_grain_pts} bdy pts)... ({_elapsed_grain_msf:.1f}s)        ", end='', flush=True)
            if i == len(Gs) - 1:
                print()  # final newline after last grain

    # Remove most added rows and columns
    xx = xx[4:-4, 4:-4]
    yy = yy[4:-4, 4:-4]
    hhlfs = hhlfs[4:-4, 4:-4]
    hhcurv = hhcurv[4:-4, 4:-4]
    hho = np.minimum(hhcurv, hhlfs)

    # Limit mesh size overall
    hho[hho < l] = l

    # Run gradient limiting
    hh = gradient_limiting(xx, yy, hho, g)
    if verbose:
        print(f"  Min. Size Function Value = {np.min(hh):.6g}")
        print(f"  Max. Size Function Value = {np.max(hh):.6g}")

    # Save grid info
    ms.ThreeNodeCoordinateList = None
    ms.GrainJunctionPoints = np.unique(jpts, axis=0)
    ms.MeshSizeFunctionGrid = [xx, yy, hh]
    ms.MeshSizeFunctionGridLimits = np.array([minx, maxx, miny, maxy])
    ms.GrainsSmoothed = Gs

    return ms
