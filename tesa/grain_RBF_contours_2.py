"""
Grain boundary contouring using RBF interpolation.
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial.distance import cdist
from .knnsearch2 import knnsearch2
from .inpoly import inpoly
from scipy.spatial import cKDTree as _cKDTree
from .find_grain_junction_points import find_grain_junction_points
from .polygonLength import polygonLength
from .resamplePolygon import resamplePolygon
from .polylineLength import polylineLength
from .resamplePolyline import resamplePolyline


def _poly2ccw(x, y):
    """
    Ensure polygon vertices are in counter-clockwise order.

    If the signed area is negative (clockwise), reverses the vertex order.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Polygon x-coordinates.
    y : ndarray, shape (n,)
        Polygon y-coordinates.

    Returns
    -------
    x : ndarray, shape (n,)
        x-coordinates in counter-clockwise order.
    y : ndarray, shape (n,)
        y-coordinates in counter-clockwise order.
    """
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + \
           0.5 * (x[-1] * y[0] - x[0] * y[-1])
    if area < 0:
        x = x[::-1]
        y = y[::-1]
    return x, y


def grain_RBF_contours_2(ms, bsep, verbose=False):
    """
    Smooth grain boundaries using RBF-based contouring.

    For each grain, builds an inside/outside indicator field using the
    raw EBSD data, interpolates it on a fine grid via cubic griddata,
    extracts the 0.5-level contour as the smoothed boundary, resamples
    it to uniform spacing, and averages shared edges between neighbors
    for consistency.

    Parameters
    ----------
    ms : Microstructure
        Microstructure state object. Must have OriginalDataCoordinateList,
        OriginalDataEulerAngle, OriginalDataPhase, Grains, GrainColors,
        and HoleGrains populated.
    bsep : float
        Target boundary point separation (spacing between resampled
        boundary vertices).
    verbose : bool, optional
        If True, print progress messages. Default False.

    Returns
    -------
    ms : Microstructure
        The input object with GrainsSmoothed and GrainPointSeparation
        attributes set.
    """
    eps_val = np.sqrt(np.finfo(float).eps)

    # Setup data
    x = ms.OriginalDataCoordinateList[:, 0].copy()
    y = ms.OriginalDataCoordinateList[:, 1].copy()

    # Find data limits and junction points
    G = ms.Grains
    minx = 1e6
    maxx = 0
    miny = 1e6
    maxy = 0
    apts = np.zeros((0, 2))
    for i in range(len(G)):
        minx = min(np.min(G[i][:, 0]), minx)
        miny = min(np.min(G[i][:, 1]), miny)
        maxx = max(np.max(G[i][:, 0]), maxx)
        maxy = max(np.max(G[i][:, 1]), maxy)
        apts = np.vstack([apts, G[i]])

    if verbose:
        print("  Finding grains within other grains...")
    jbpts = find_grain_junction_points(ms)

    # For each grain, find boundaries using contour method
    count = 0
    Gr = [None] * len(G)

    if verbose:
        print(f"  Contouring grains ({len(G)} grains)...")
    for n in range(len(G)):
        if verbose:
            print(f"    Grain {n + 1}/{len(G)}...", end='\r', flush=True)

        # Find points inside current grain (bounding box pre-filter for speed)
        _gn = G[n]
        _gx_min, _gx_max = np.min(_gn[:, 0]) - 1, np.max(_gn[:, 0]) + 1
        _gy_min, _gy_max = np.min(_gn[:, 1]) - 1, np.max(_gn[:, 1]) + 1
        _bbox_mask = (x >= _gx_min) & (x <= _gx_max) & (y >= _gy_min) & (y <= _gy_max)
        _bbox_idx = np.where(_bbox_mask)[0]

        # Only test points within bounding box against the polygon
        id_in = np.zeros(len(x), dtype=bool)
        if len(_bbox_idx) > 0:
            _bbox_pts = np.column_stack([x[_bbox_idx], y[_bbox_idx]])
            _bbox_in = inpoly(_bbox_pts, _gn)
            id_in[_bbox_idx] = _bbox_in
        ido = id_in.copy()

        # Distance from data points to grain boundary (use cKDTree)
        _grain_tree = _cKDTree(_gn)
        _pts_in = np.column_stack([x[id_in], y[id_in]])
        d = _grain_tree.query(_pts_in)[0] if len(_pts_in) > 0 else np.array([])
        d = d.ravel()
        id_in_copy = id_in.copy()
        id_in_idx = np.where(id_in)[0]
        threshold = np.sqrt(0.5 ** 2 + 1.5 ** 2) + eps_val
        id_in[id_in_idx] = d < threshold

        # Skip grain if no interior points remain after distance threshold
        if not np.any(id_in):
            Gr[count] = G[n]
            count += 1
            continue

        # Setup current data group and add mid points
        xs_range = np.arange(np.min(x[id_in]) - 1, np.max(x[id_in]) + 2, 1.0)
        ys_range = np.arange(np.min(y[id_in]) - 1, np.max(y[id_in]) + 2, 1.0)
        XS, YS = np.meshgrid(xs_range, ys_range)
        xs = XS.ravel()
        ys = YS.ravel()

        d_grid, _ = _grain_tree.query(np.column_stack([xs, ys]))
        d_grid = d_grid.ravel()
        mask_grid = d_grid < threshold
        xs = xs[mask_grid]
        ys = ys[mask_grid]

        gs = inpoly(np.column_stack([xs, ys]), G[n]).astype(float)

        xp = x[id_in].copy()
        yp = y[id_in].copy()
        xa = np.full_like(xp, np.nan)
        ya = np.full_like(yp, np.nan)
        ca = 0

        K_val = min(xp.shape[0], 7)
        if K_val > 1:
            _xp_tree = _cKDTree(np.column_stack([xp, yp]))
            d_knn, ip_knn = _xp_tree.query(np.column_stack([xp, yp]), k=K_val)
            d_knn = d_knn[:, 1:]  # remove self-match
            ip_knn = ip_knn[:, 1:]

            rd = np.all(d_knn < np.sqrt(2) + eps_val, axis=1)
            for ii in range(xp.shape[0]):
                if not rd[ii]:
                    idj = d_knn[ii, :] < np.sqrt(2) + eps_val
                    sumidj = np.sum(idj)
                    if sumidj > 0 and ca + sumidj <= xa.shape[0]:
                        xa[ca:ca + sumidj] = (xp[ii] + xp[ip_knn[ii, idj]]) / 2.0
                        ya[ca:ca + sumidj] = (yp[ii] + yp[ip_knn[ii, idj]]) / 2.0
                        ca += sumidj

        xa = xa[~np.isnan(xa)]
        ya = ya[~np.isnan(ya)]
        xs = np.concatenate([xs, xa])
        ys = np.concatenate([ys, ya])
        gs = np.concatenate([gs, np.ones(xa.shape[0])])

        # Second round of midpoint generation for non-data points
        known_pts = np.vstack([np.column_stack([x[id_in], y[id_in]]),
                                np.column_stack([xa, ya])]) if xa.shape[0] > 0 else \
            np.column_stack([x[id_in], y[id_in]])
        known_set = set(map(tuple, np.round(known_pts, 12)))
        ipt = np.array([tuple(np.round(row, 12)) not in known_set
                         for row in np.column_stack([xs, ys])])
        xp2 = xs[ipt]
        yp2 = ys[ipt]
        xa2 = np.full_like(xp2, np.nan)
        ya2 = np.full_like(yp2, np.nan)
        ca2 = 0

        K_val2 = min(xp2.shape[0], 7)
        if K_val2 > 1:
            _xp2_tree = _cKDTree(np.column_stack([xp2, yp2]))
            d_knn2, ip_knn2 = _xp2_tree.query(np.column_stack([xp2, yp2]), k=K_val2)
            d_knn2 = d_knn2[:, 1:]
            ip_knn2 = ip_knn2[:, 1:]

            rd2 = np.all(d_knn2 < np.sqrt(2) + eps_val, axis=1)
            for ii in range(xp2.shape[0]):
                if not rd2[ii]:
                    idj2 = d_knn2[ii, :] < np.sqrt(2) + eps_val
                    sumidj2 = np.sum(idj2)
                    if sumidj2 > 0 and ca2 + sumidj2 <= xa2.shape[0]:
                        xa2[ca2:ca2 + sumidj2] = (xp2[ii] + xp2[ip_knn2[ii, idj2]]) / 2.0
                        ya2[ca2:ca2 + sumidj2] = (yp2[ii] + yp2[ip_knn2[ii, idj2]]) / 2.0
                        ca2 += sumidj2

        xa2 = xa2[~np.isnan(xa2)]
        ya2 = ya2[~np.isnan(ya2)]
        xs = np.concatenate([xs, xa2])
        ys = np.concatenate([ys, ya2])
        gs = np.concatenate([gs, np.zeros(xa2.shape[0])])

        # Remove any boundary or junction points
        p_pts = np.column_stack([xs, ys])
        jbpts_set = set(map(tuple, np.round(jbpts, 12)))
        rp = np.array([tuple(np.round(row, 12)) in jbpts_set for row in p_pts])
        p_pts = p_pts[~rp, :]
        gs = gs[~rp]
        xs = p_pts[:, 0]
        ys = p_pts[:, 1]

        # Add junction points that belong to this grain
        p_grain = G[n]
        jp_mask = np.array([tuple(np.round(row, 12)) in jbpts_set for row in p_grain])
        jp = p_grain[jp_mask, :]
        if jp.shape[0] > 0:
            xs = np.concatenate([xs, jp[:, 0]])
            ys = np.concatenate([ys, jp[:, 1]])
            gs = np.concatenate([gs, 0.5 * np.ones(jp.shape[0])])

        # Remove duplicates (stable)
        combined = np.column_stack([xs, ys])
        _, idr = np.unique(combined, axis=0, return_index=True)
        idr = np.sort(idr)
        xs = xs[idr]
        ys = ys[idr]
        gs = gs[idr]

        # Compute interpolation — MATLAB uses scatteredInterpolant(...,'natural').
        # scipy.interpolate.griddata with method='cubic' is the closest match.
        interp_points = np.column_stack([xs, ys])
        interp_values = gs

        # Resample on a finer grid
        rsep = 0.2
        xr = np.arange(np.min(x[id_in]) - 1, np.max(x[id_in]) + 1 + rsep / 2, rsep)
        yr = np.arange(np.min(y[id_in]) - 1, np.max(y[id_in]) + 1 + rsep / 2, rsep)

        # Remove original data inside this grain from x,y for future grains
        x = x[~ido]
        y = y[~ido]

        XR, YR = np.meshgrid(xr, yr)
        d_fine, _ = _grain_tree.query(np.column_stack([XR.ravel(), YR.ravel()]))
        d_fine = d_fine.ravel()
        ZR = np.zeros(XR.shape)
        mask_fine = d_fine < np.sqrt(0.5 ** 2 + 1.5 ** 2)
        pts_fine = np.column_stack([XR.ravel()[mask_fine], YR.ravel()[mask_fine]])
        ZR_flat = ZR.ravel()
        try:
            interp_vals = griddata(interp_points, interp_values, pts_fine, method='cubic')
        except Exception:
            interp_vals = griddata(interp_points, interp_values, pts_fine, method='linear')
        # Replace NaN with 0
        interp_vals[np.isnan(interp_vals)] = 0.0
        ZR_flat[mask_fine] = interp_vals
        ZR = ZR_flat.reshape(XR.shape)

        # Compute contours at level 0.5 (use Agg temporarily to avoid GUI windows)
        import matplotlib
        import matplotlib.pyplot as plt
        _prev_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        plt.switch_backend('Agg')
        fig_temp, ax_temp = plt.subplots()
        cs_contour = ax_temp.contour(XR, YR, ZR, levels=[0.5])
        plt.close(fig_temp)
        # Restore the original backend
        matplotlib.use(_prev_backend)
        plt.switch_backend(_prev_backend)

        # Extract contour paths at level 0.5.
        # When a grain contains smaller grains (holes), the contour produces
        # multiple segments: the outer boundary and inner hole boundaries.
        # We pick the segment that encloses the largest area (the outer boundary).
        # Inner hole grains have their own contours computed separately.
        if len(cs_contour.allsegs) > 0 and len(cs_contour.allsegs[0]) > 0:
            _all_segs = cs_contour.allsegs[0]
            if len(_all_segs) == 1:
                cg = _all_segs[0]
            else:
                # Pick the segment with the largest enclosed area
                def _poly_area(pts):
                    pts = np.asarray(pts)
                    if len(pts) < 3:
                        return 0.0
                    return 0.5 * abs(np.sum(pts[:-1,0]*pts[1:,1] - pts[1:,0]*pts[:-1,1]) +
                                     pts[-1,0]*pts[0,1] - pts[0,0]*pts[-1,1])
                cg = max(_all_segs, key=_poly_area)
        else:
            Gr[count] = G[n].copy()
            count += 1
            continue

        cg = np.array(cg)

        # If the contour didn't close properly (large gap between first and last
        # point), fall back to the original grain boundary. This can happen for
        # very elongated grains or grains with interior holes where the RBF
        # contour exits on one side and doesn't come back around.
        _gap = np.sqrt(np.sum((cg[0] - cg[-1]) ** 2))
        if _gap > 3 * bsep:
            Gr[count] = G[n].copy()
            count += 1
            continue

        Gr[count] = cg.copy()

        # Resample the contour
        length = polygonLength(Gr[count])
        n_pts = max(3, round(length / bsep))
        Gr[count] = resamplePolygon(Gr[count], n_pts)
        # unique stable
        _, idr_u = np.unique(Gr[count], axis=0, return_index=True)
        idr_u = np.sort(idr_u)
        Gr[count] = Gr[count][idr_u, :]

        # Snap to junction/boundary points (matches MATLAB lines 157-167)
        in_jb = inpoly(jbpts, G[n])
        if np.any(in_jb):
            gjb = jbpts[in_jb, :]
            # Pass 1: snap existing contour vertices to nearby junction points
            _gjb_tree = _cKDTree(gjb)
            d_snap, id_snap = _gjb_tree.query(Gr[count])
            d_snap = d_snap.ravel()
            id_snap = id_snap.ravel()
            snap_mask = d_snap < bsep
            if np.any(snap_mask):
                Gr[count][snap_mask, :] = gjb[id_snap[snap_mask], :]

            # Pass 2: insert any junction points that are still missing.
            # MATLAB's scatteredInterpolant('natural') produces contours close
            # enough that pass 1 catches all junction points.  Python's griddata
            # can drift further, leaving some junction points unsnapped.  For
            # each missing point, find the nearest edge of the contour polygon
            # and insert the point between the edge's two vertices.
            gr_set = set(map(tuple, np.round(Gr[count], 12)))
            missing = np.array([j for j, pt in enumerate(gjb)
                                if tuple(np.round(pt, 12)) not in gr_set])
            if len(missing) > 0:
                gr = Gr[count]
                for mi in missing:
                    jp = gjb[mi]
                    # Compute distance from jp to each edge of the polygon
                    n_pts = gr.shape[0]
                    p0 = gr
                    p1 = np.roll(gr, -1, axis=0)
                    edge_vec = p1 - p0
                    edge_len_sq = np.sum(edge_vec ** 2, axis=1)
                    # Parametric projection of jp onto each edge, clamped to [0,1]
                    t = np.sum((jp - p0) * edge_vec, axis=1)
                    safe_len_sq = np.where(edge_len_sq > 0, edge_len_sq, 1.0)
                    t = np.where(edge_len_sq > 0, t / safe_len_sq, 0.0)
                    t = np.clip(t, 0.0, 1.0)
                    proj = p0 + t[:, None] * edge_vec
                    dist_sq = np.sum((jp - proj) ** 2, axis=1)
                    best_edge = np.argmin(dist_sq)
                    # Insert jp after vertex best_edge
                    ins_idx = best_edge + 1
                    gr = np.insert(gr, ins_idx, jp, axis=0)
                Gr[count] = gr

        _, idr_u = np.unique(Gr[count], axis=0, return_index=True)
        idr_u = np.sort(idr_u)
        Gr[count] = Gr[count][idr_u, :]

        xg_c, yg_c = _poly2ccw(Gr[count][:, 0], Gr[count][:, 1])
        Gr[count] = np.column_stack([xg_c, yg_c])
        count += 1

    # Trim Gr to actual count
    Gr = Gr[:count]

    # Find grain neighbors
    gnebs = [[] for _ in range(len(G))]
    gjpts_list = [None] * len(G)
    gjptsid = [None] * len(G)
    jbpts_set = set(map(tuple, np.round(jbpts, 12)))
    for i in range(len(G)):
        member = np.array([tuple(np.round(row, 12)) in jbpts_set for row in G[i]])
        gjpts_list[i] = member

    gi_list = list(range(len(G)))
    if verbose:
        print("\n  Finding grain neighbors...")
    for i in range(len(G)):
        pass  # per-grain detail suppressed
        for k in range(len(gi_list)):
            j = gi_list[k]
            if i == j:
                continue
            if np.any(gjpts_list[i] & gjpts_list[j][:len(gjpts_list[i])] if len(gjpts_list[j]) >= len(gjpts_list[i]) else False):
                # More precise check: shared junction points
                gi_jpts = set(map(tuple, np.round(G[i][gjpts_list[i], :], 12)))
                gj_jpts = set(map(tuple, np.round(G[j][gjpts_list[j], :], 12)))
                if len(gi_jpts & gj_jpts) > 0:
                    gnebs[i].append(j)
                    gnebs[j].append(i)
        if len(gi_list) > 0:
            gi_list.pop(0)

    # Deduplicate neighbors
    for i in range(len(G)):
        gnebs[i] = list(set(gnebs[i]))

    # Match polylines between neighboring grains
    for i in range(len(G)):
        pass  # per-grain detail suppressed
        xg_r, yg_r = _poly2ccw(G[i][:, 0], G[i][:, 1])
        G[i] = np.column_stack([xg_r, yg_r])
        member = np.array([tuple(np.round(row, 12)) in jbpts_set for row in G[i]])
        id_found = np.where(member)[0]
        if len(id_found) == 0:
            continue
        id_first = id_found[0]
        if id_first == 0:
            continue
        G[i] = np.vstack([G[i][id_first:, :], G[i][:id_first, :]])

    mnpl = 100
    Gpl = [[None] * mnpl for _ in range(len(G))]
    for i in range(len(G)):
        member = np.array([tuple(np.round(row, 12)) in jbpts_set for row in G[i]])
        if not np.any(member):
            Gpl[i][0] = G[i].copy()
            continue
        G_closed = np.vstack([G[i], G[i][0:1, :]])
        member_closed = np.array([tuple(np.round(row, 12)) in jbpts_set for row in G_closed])
        jd = np.where(member_closed)[0]
        for j in range(len(jd) - 1):
            Gpl[i][j] = G_closed[jd[j]:jd[j + 1] + 1, :]

    gnpl = [None] * len(G)
    found_npl = False
    if verbose:
        print("  Setting up grain edges...")
    for i in range(len(G)):
        pass  # per-grain detail suppressed
        gnpl[i] = []
        if Gpl[i][1] is None:
            continue
        for j in range(mnpl):
            if Gpl[i][j] is None:
                break
            pl_pts = Gpl[i][j]
            # Check if on boundary
            if (np.all(pl_pts[:, 0] < minx + eps_val) or
                    np.all(pl_pts[:, 0] > maxx - eps_val) or
                    np.all(pl_pts[:, 1] < miny + eps_val) or
                    np.all(pl_pts[:, 1] > maxy - eps_val)):
                gnpl[i].append([i, j])
                continue

            for k in range(len(gnebs[i])):
                ii = gnebs[i][k]
                for jj in range(mnpl):
                    if Gpl[ii][jj] is None:
                        break
                    # Check if polylines match (reversed)
                    pl_set = set(map(tuple, np.round(Gpl[i][j], 12)))
                    npl_set = set(map(tuple, np.round(Gpl[ii][jj], 12)))
                    if pl_set == npl_set:
                        # Check if they are reversed
                        if (Gpl[i][j].shape[0] == Gpl[ii][jj].shape[0] and
                                np.allclose(Gpl[i][j], Gpl[ii][jj][::-1, :], atol=1e-10)):
                            gnpl[i].append([ii, jj])
                            found_npl = True
                            break
                if found_npl:
                    break
            if not found_npl:
                gnpl[i].append([i, j])
            found_npl = False

    # Resample grain boundaries and match polylines in contoured grains
    for i in range(len(Gr)):
        if Gr[i] is None:
            continue
        gs_first = G[i][0, :] if i < len(G) else None
        if gs_first is not None:
            member = np.array([tuple(np.round(row, 12)) == tuple(np.round(gs_first, 12)) for row in Gr[i]])
            id_found = np.where(member)[0]
            if len(id_found) > 0:
                Gr[i] = np.vstack([Gr[i][id_found[0]:, :], Gr[i][:id_found[0], :]])

    Gcpl = [[None] * mnpl for _ in range(len(Gr))]
    for i in range(len(Gr)):
        if Gr[i] is None:
            continue
        member = np.array([tuple(np.round(row, 12)) in jbpts_set for row in Gr[i]])
        if not np.any(member):
            Gcpl[i][0] = Gr[i].copy()
            continue
        Gr_closed = np.vstack([Gr[i], Gr[i][0:1, :]])
        member_closed = np.array([tuple(np.round(row, 12)) in jbpts_set for row in Gr_closed])
        jd = np.where(member_closed)[0]
        for j in range(len(jd) - 1):
            Gcpl[i][j] = Gr_closed[jd[j]:jd[j + 1] + 1, :]

    Gn = [None] * len(Gr)
    gdi = np.zeros((0, 2), dtype=int)
    gdpl = []

    if verbose:
        print("  Matching grain boundaries...")
    for i in range(len(Gn)):
        if verbose:
            print(f"    Grain {i + 1}/{len(Gn)}...", end='\r', flush=True)
        if Gcpl[i][1] is None and Gcpl[i][0] is not None:
            length = polygonLength(Gcpl[i][0])
            n_pts = max(3, int(np.ceil(length / bsep)))
            Gn[i] = resamplePolygon(Gcpl[i][0], n_pts)
            Gn[i][Gn[i][:, 0] < minx + bsep / 2, 0] = minx
            Gn[i][Gn[i][:, 0] > maxx - bsep / 2, 0] = maxx
            Gn[i][Gn[i][:, 1] < miny + bsep / 2, 1] = miny
            Gn[i][Gn[i][:, 1] > maxy - bsep / 2, 1] = maxy
            _, idr_u = np.unique(Gn[i], axis=0, return_index=True)
            idr_u = np.sort(idr_u)
            Gn[i] = Gn[i][idr_u, :]
            continue

        Gn[i] = np.zeros((0, 2))
        if gnpl[i] is None:
            continue

        for j in range(len(gnpl[i])):
            if Gcpl[i][j] is None:
                break
            gnpl_ij = gnpl[i][j]
            if gnpl_ij[0] == i and gnpl_ij[1] == j:
                # Own polyline
                length = polylineLength(Gcpl[i][j])
                n_pts = max(2, int(np.ceil(length / bsep)))
                pl = resamplePolyline(Gcpl[i][j], n_pts)
                Gn[i] = np.vstack([Gn[i], pl])
                continue

            # Check if already computed
            if gdi.shape[0] > 0:
                match = np.all(gdi == gnpl_ij, axis=1)
                if np.any(match):
                    loc = np.where(match)[0][0]
                    Gn[i] = np.vstack([Gn[i], gdpl[loc][::-1, :]])
                    continue

            # Average the two polylines
            length = polylineLength(Gcpl[i][j])
            n_pts = max(2, int(np.ceil(length / bsep)))
            pl1 = resamplePolyline(Gcpl[i][j], n_pts)
            neighbor_pl = Gcpl[gnpl_ij[0]][gnpl_ij[1]]
            if neighbor_pl is not None:
                pl2 = resamplePolyline(neighbor_pl, pl1.shape[0])[::-1, :]
                new_pl = (pl1 + pl2) / 2.0
            else:
                new_pl = pl1
            Gn[i] = np.vstack([Gn[i], new_pl])
            gdi = np.vstack([gdi, [[i, j]]])
            gdpl.append(new_pl)

        if Gn[i].shape[0] > 0:
            Gn[i][Gn[i][:, 0] < minx + bsep / 2, 0] = minx
            Gn[i][Gn[i][:, 0] > maxx - bsep / 2, 0] = maxx
            Gn[i][Gn[i][:, 1] < miny + bsep / 2, 1] = miny
            Gn[i][Gn[i][:, 1] > maxy - bsep / 2, 1] = maxy
            _, idr_u = np.unique(Gn[i], axis=0, return_index=True)
            idr_u = np.sort(idr_u)
            Gn[i] = Gn[i][idr_u, :]

    # Ensure CCW and snap to junction points
    gsize = []
    for i in range(len(Gn)):
        if Gn[i] is None or Gn[i].shape[0] == 0:
            gsize.append(0)
            continue
        xg_f, yg_f = _poly2ccw(Gn[i][:, 0], Gn[i][:, 1])
        Gn[i] = np.column_stack([xg_f, yg_f])
        member = np.array([tuple(np.round(row, 12)) in jbpts_set for row in Gn[i]])
        id_found = np.where(member)[0]
        Gn[i] = 1e-5 * np.round(1e5 * Gn[i])
        if len(id_found) > 0:
            for idx in id_found:
                best = np.argmin(np.sum((jbpts - Gn[i][idx, :]) ** 2, axis=1))
                Gn[i][idx, :] = jbpts[best, :]
        gsize.append(Gn[i].shape[0])

    # Compute total area (for reporting)
    total_area = 0
    gsize_arr = np.array(gsize)
    remaining = gsize_arr.copy()
    while True:
        if np.all(remaining < eps_val):
            break
        ng = np.argmax(remaining)
        if Gn[ng] is not None and Gn[ng].shape[0] > 2:
            if ms.HoleGrains is None or ng not in ms.HoleGrains:
                total_area += 0.5 * np.abs(np.sum(
                    Gn[ng][:-1, 0] * Gn[ng][1:, 1] - Gn[ng][1:, 0] * Gn[ng][:-1, 1]) +
                    Gn[ng][-1, 0] * Gn[ng][0, 1] - Gn[ng][0, 0] * Gn[ng][-1, 1])
        remaining[ng] = 0

    if verbose:
        print(f"\n  Total Area = {total_area:.6g} of {(maxx - minx) * (maxy - miny):.6g}")

    # Save the smoothed grains
    ms.GrainsSmoothed = Gn
    ms.GrainPointSeparation = bsep

    return ms
