"""
Type 3: Non-conforming rectangular grid.

Generates a structured rectangular grid mesh using meshgrid,
followed by Delaunay triangulation. No iterative mesh optimization — fastest
mesh type. Properties are assigned per element based on nearest EBSD data.
"""

import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors_global
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch as _Patch
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import mode as _mode_fn

from .fixmesh import fixmesh
from .cleanup_mesh import cleanup_mesh
from .simpqual import simpqual
from .triarea import triarea
from .inpoly import inpoly


def mesh_nonconforming_rectgrid(ms, job, run_dir=None, log_path=None, settings=None):
    """
    Generate a Type 3 non-conforming rectangular grid mesh.

    Creates a structured rectangular grid using meshgrid,
    Delaunay-triangulates it, and assigns element properties from nearest
    EBSD data. No iterative optimization -- near-instant mesh generation.

    Parameters
    ----------
    ms : Microstructure
        Must have EBSD data loaded (via load_ebsd).
    job : dict
        Job dictionary with target_elements, etc.
    run_dir : str or None
        Results directory for saving figures and mesh files.
    log_path : str or None
        Path to log.md file.
    settings : dict or None
        Global settings (verbose, show_figures, etc.).

    Returns
    -------
    ms : Microstructure
        Updated with rectangular grid mesh fields.
    """
    # ── Unpack settings ──────────────────────────────────────────────────
    if settings is None:
        settings = {}
    vc = settings.get("verbose_console", "medium")
    vl = settings.get("verbose_log", "medium")
    console_on = vc in ("medium", "high")
    show_figures = settings.get("show_figures", False)
    figure_pause = settings.get("figure_pause", 1.0)
    figure_dpi = settings.get("figure_dpi", 150)
    random_seed = settings.get("random_seed", 42)
    figure_title_fontsize = settings.get("figure_title_fontsize", 14)
    figure_fontsize = settings.get("figure_fontsize", 12)

    # Phase colors and grain colormap (with fallback defaults)
    if "phase_colors" not in settings:
        print("  WARNING: 'phase_colors' not in settings, using default palette")
    _phase_base_colors = settings.get("phase_colors",
        ['red', 'lime', 'tab:blue', 'tab:orange', 'tab:purple',
         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])
    if "grain_colormap" not in settings:
        print("  WARNING: 'grain_colormap' not in settings, using default 'tab20'")
    _grain_cmap_name = settings.get("grain_colormap", "tab20")
    _grain_alpha = settings.get("grain_colormap_alpha", 0.9)
    _phase_alpha = settings.get("phase_colors_alpha", 0.9)
    _overlay_alpha = settings.get("mesh_overlay_alpha", 0.5)
    _overlay_lw = settings.get("mesh_overlay_linewidth", 0.3)

    # ── Unpack job parameters ────────────────────────────────────────────
    ebsd_file = job["ebsd_file"]
    ebsd_name = os.path.basename(ebsd_file)
    target_elements = job.get("target_elements", 2000)

    # ── Ensure interactive matplotlib backend ────────────────────────────
    if show_figures:
        backend = matplotlib.get_backend()
        if backend.lower() == 'agg':
            matplotlib.use('MacOSX')

    # ── Figures directory ────────────────────────────────────────────────
    FIG_DIR = os.path.join(run_dir, "mesh") if run_dir else None
    if FIG_DIR:
        os.makedirs(FIG_DIR, exist_ok=True)
        os.makedirs(os.path.join(FIG_DIR, "diagnostics"), exist_ok=True)

    # ── Figure counter (continues from Stage 1) ─────────────────────────
    _stage1_fig_count = getattr(ms, 'fig_count', 0)
    fig_num = [_stage1_fig_count]

    def _next_fig_name(name):
        """Return filename (counter kept for internal tracking)."""
        fig_num[0] += 1
        return name

    def save(fig, name):
        """Save figure to disk, optionally display on screen.
        Diagnostic figures are saved silently without display or console output."""
        if FIG_DIR is None:
            plt.close(fig)
            return
        path = os.path.join(FIG_DIR, name)
        fig.savefig(path, dpi=figure_dpi, bbox_inches='tight')
        _is_diag = name.startswith('diagnostics')
        if show_figures and not _is_diag:
            fig.show()
            plt.pause(figure_pause)
        plt.close(fig)
        if console_on and not _is_diag:
            print(f"  Saved: {name}")

    # ── Domain bounds ────────────────────────────────────────────────────
    coords = ms.OriginalDataCoordinateList
    minx, maxx = coords[:, 0].min(), coords[:, 0].max()
    miny, maxy = coords[:, 1].min(), coords[:, 1].max()
    domain_area = (maxx - minx) * (maxy - miny)

    # Compute h0 from target element count
    # Hex grid cell area = h0² * sqrt(3)/2, each cell yields ~2 triangles
    # n_elements ≈ 4 * domain_area / (h0² * sqrt(3))
    h0 = np.sqrt(4 * domain_area / (target_elements * np.sqrt(3)))

    # Title info for figures
    nx_pts = len(np.unique(np.round(coords[:, 0], 6)))
    ny_pts = len(np.unique(np.round(coords[:, 1], 6)))
    _title_info = f"{ebsd_name} ({nx_pts}x{ny_pts})"

    if console_on:
        print(f"\n---- Rectangular grid mesh (Type 3) ----")
        print(f"  Domain: [{minx:.2f}, {maxx:.2f}] x [{miny:.2f}, {maxy:.2f}]")
        print(f"  Target elements: {target_elements}")
        print(f"  Computed h0: {h0:.4g}")

    # ── Set random seed for reproducibility ──────────────────────────────
    if random_seed is not None:
        np.random.seed(random_seed)

    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Create structured rectangular grid
    # ══════════════════════════════════════════════════════════════════════

    # Compute grid dimensions
    xs = round((maxx - minx) / h0) + 1
    ys = round((maxy - miny) / h0) + 1

    # Create regular rectangular grid (no hex offset)
    x_vals = np.linspace(minx, maxx, xs)
    y_vals = np.linspace(miny, maxy, ys)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Flatten to point list
    p_hex = np.column_stack([xx.ravel(), yy.ravel()])

    if console_on:
        print(f"  Rectangular grid: {xs} cols x {ys} rows = {p_hex.shape[0]} points")

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Identify boundary points (on the rectangular domain edge)
    # ══════════════════════════════════════════════════════════════════════

    # Boundary points: on any edge of the rectangle
    _on_left = np.isclose(p_hex[:, 0], minx)
    _on_right = np.isclose(p_hex[:, 0], maxx)
    _on_bottom = np.isclose(p_hex[:, 1], miny)
    _on_top = np.isclose(p_hex[:, 1], maxy)
    _on_boundary = _on_left | _on_right | _on_bottom | _on_top
    pbounds = p_hex[_on_boundary].copy()

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Delaunay triangulate the hex grid
    # ══════════════════════════════════════════════════════════════════════

    t_hex = Delaunay(p_hex).simplices

    # Initial fixmesh cleanup
    p_hex, t_hex, _ = fixmesh(p_hex, t_hex)

    q_initial = simpqual(p_hex, t_hex)
    if console_on:
        print(f"  Initial hex mesh: {p_hex.shape[0]} nodes, {t_hex.shape[0]} elements, "
              f"q_min={np.min(q_initial):.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: Remove boundary nodes, re-insert from pbounds
    # ══════════════════════════════════════════════════════════════════════

    # Remove nodes within ds of the boundary (keep interior only)
    ds = h0 / 5
    _interior_mask = (
        (p_hex[:, 0] > minx + ds) & (p_hex[:, 0] < maxx - ds) &
        (p_hex[:, 1] > miny + ds) & (p_hex[:, 1] < maxy - ds)
    )
    p_interior = p_hex[_interior_mask]

    # Extract unique boundary coordinates from pbounds
    pbounds_unique = np.unique(np.round(pbounds, 12), axis=0)
    xg = np.unique(pbounds_unique[:, 0])
    yg = np.unique(pbounds_unique[:, 1])

    # Re-insert boundary nodes: 4 corners + edges from hex grid + interior
    # Corners
    corners = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])

    # Bottom edge: interior x-coords at y=miny
    xg_inner = xg[(xg > minx + 1e-10) & (xg < maxx - 1e-10)]
    bottom_edge = np.column_stack([xg_inner, np.full(len(xg_inner), miny)])

    # Top edge: interior x-coords at y=maxy
    top_edge = np.column_stack([xg_inner, np.full(len(xg_inner), maxy)])

    # Left edge: interior y-coords at x=minx
    yg_inner = yg[(yg > miny + 1e-10) & (yg < maxy - 1e-10)]
    left_edge = np.column_stack([np.full(len(yg_inner), minx), yg_inner])

    # Right edge: interior y-coords at x=maxx
    right_edge = np.column_stack([np.full(len(yg_inner), maxx), yg_inner])

    # Assemble: boundary first, then interior
    boundary_pts = np.vstack([corners, bottom_edge, top_edge, left_edge, right_edge])
    p = np.vstack([boundary_pts, p_interior])

    # Remove any duplicate points
    p = np.unique(np.round(p, 12), axis=0)

    # ══════════════════════════════════════════════════════════════════════
    # Step 5: Final Delaunay triangulation on reordered nodes
    # ══════════════════════════════════════════════════════════════════════

    t = Delaunay(p).simplices

    # Cleanup
    p, t, _ = fixmesh(p, t)
    p, t, _cleanup_stats = cleanup_mesh(p, t, verbose=console_on)

    # Ensure consistent triangle orientation
    _sgn = triarea(p, t)
    _sw = _sgn < 0
    if np.any(_sw):
        t[_sw, 0], t[_sw, 1] = t[_sw, 1].copy(), t[_sw, 0].copy()

    t_mesh = time.time() - t_start

    # ── Element centroids ────────────────────────────────────────────────
    centroids = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3.0

    # ── Quality metrics ──────────────────────────────────────────────────
    q_all = simpqual(p, t)
    q_min = np.min(q_all)
    q_mean = np.mean(q_all)
    q_std = np.std(q_all)
    n_elems = t.shape[0]
    n_nodes = p.shape[0]
    elem_areas = np.abs(triarea(p, t))
    covered_area = np.sum(elem_areas)

    if console_on:
        print(f"  Final mesh: {n_nodes} nodes, {n_elems} elements")
        print(f"  Quality: q_min={q_min:.4f}, q_mean={q_mean:.4f}")
        print(f"  Mesh generation time: {t_mesh:.2f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Assign properties per element using nearest EBSD data point
    # ══════════════════════════════════════════════════════════════════════
    if console_on:
        print(f"\n---- Assigning element properties (nearest neighbor) ----")

    ebsd_coords = ms.OriginalDataCoordinateList
    ebsd_phase = ms.OriginalDataPhase
    ebsd_euler = ms.OriginalDataEulerAngle

    # For each element centroid, find the nearest EBSD data point
    _tree = cKDTree(ebsd_coords)
    _, _nearest_idx = _tree.query(centroids)

    # Assign element phase and Euler angles from nearest EBSD point
    el_phase = ebsd_phase[_nearest_idx].astype(int)
    el_angle = ebsd_euler[_nearest_idx].copy()

    # Assign elements to grains using smoothed grain boundaries (if available)
    Gs = getattr(ms, 'GrainsSmoothed', None) or ms.Grains
    n_grains = len([g for g in Gs if g is not None and np.asarray(g).size > 0])
    valid_gs = [np.asarray(g) for g in Gs if g is not None and np.asarray(g).size > 0]

    # Grain assignment: for each centroid, find which grain polygon it falls inside.
    # Uses bounding-box pre-filtering + matplotlib Path.contains_points for speed.
    from matplotlib.path import Path as _MplPath
    el_grain = np.full(n_elems, -1, dtype=int)
    cx, cy = centroids[:, 0], centroids[:, 1]

    for gi, g in enumerate(valid_gs):
        if g.shape[0] < 3:
            continue
        # Bounding-box filter: only test centroids inside this grain's bbox
        gx_min, gy_min = g[:, 0].min(), g[:, 1].min()
        gx_max, gy_max = g[:, 0].max(), g[:, 1].max()
        bbox_mask = (cx >= gx_min) & (cx <= gx_max) & (cy >= gy_min) & (cy <= gy_max)
        # Only test unassigned centroids within the bounding box
        candidates = np.where(bbox_mask & (el_grain < 0))[0]
        if len(candidates) == 0:
            continue
        path = _MplPath(g)
        inside = path.contains_points(centroids[candidates])
        el_grain[candidates[inside]] = gi

    # Any remaining unassigned: assign to nearest grain centroid
    unassigned = np.where(el_grain < 0)[0]
    if len(unassigned) > 0:
        grain_centroids = np.array([np.mean(g, axis=0) for g in valid_gs])
        _g_tree = cKDTree(grain_centroids)
        _, _g_nearest = _g_tree.query(centroids[unassigned])
        el_grain[unassigned] = _g_nearest

    # Build GrainsElements mapping
    grain_elsets = {}
    for gi in range(n_grains):
        gi_mask = np.where(el_grain == gi)[0]
        if len(gi_mask) > 0:
            grain_elsets[gi] = gi_mask

    # Per-grain phase and angles (mode of EBSD points inside each grain).
    # Uses bounding-box pre-filtering to avoid testing all EBSD points against every grain.
    grain_phases = np.zeros(n_grains, dtype=int)
    grain_angles = np.zeros((n_grains, 3))
    ex, ey = ebsd_coords[:, 0], ebsd_coords[:, 1]
    for gi, g in enumerate(valid_gs):
        if g.shape[0] < 3:
            continue
        # Bounding-box filter on EBSD coordinates
        gx_min, gy_min = g[:, 0].min(), g[:, 1].min()
        gx_max, gy_max = g[:, 0].max(), g[:, 1].max()
        bbox_mask = (ex >= gx_min) & (ex <= gx_max) & (ey >= gy_min) & (ey <= gy_max)
        candidates = np.where(bbox_mask)[0]
        if len(candidates) == 0:
            continue
        path = _MplPath(g)
        _inside = path.contains_points(ebsd_coords[candidates])
        idx_inside = candidates[_inside]
        if len(idx_inside) > 0:
            grain_phases[gi] = int(_mode_fn(ebsd_phase[idx_inside], keepdims=False).mode)
            grain_angles[gi, 0] = float(_mode_fn(ebsd_euler[idx_inside, 0], keepdims=False).mode)
            grain_angles[gi, 1] = float(_mode_fn(ebsd_euler[idx_inside, 1], keepdims=False).mode)
            grain_angles[gi, 2] = float(_mode_fn(ebsd_euler[idx_inside, 2], keepdims=False).mode)

    n_assigned = int(np.sum(el_grain >= 0))
    if console_on:
        print(f"  Elements assigned to grains: {n_assigned}/{n_elems}")
        unique_phases = np.unique(el_phase)
        print(f"  Phases in mesh: {len(unique_phases)} ({', '.join(str(ph) for ph in unique_phases)})")

    # ══════════════════════════════════════════════════════════════════════
    # 6-node (quadratic) mesh: insert midside nodes
    # ══════════════════════════════════════════════════════════════════════
    _p6_12 = (p[t[:, 0]] + p[t[:, 1]]) / 2.0
    _p6_23 = (p[t[:, 1]] + p[t[:, 2]]) / 2.0
    _p6_13 = (p[t[:, 0]] + p[t[:, 2]]) / 2.0

    _all_midside = np.vstack([_p6_12, _p6_23, _p6_13])
    _unique_midside = np.unique(_all_midside, axis=0)
    _p6 = np.vstack([p, _unique_midside])

    _p6_round = np.round(_p6, 12)
    _coord_to_idx = {tuple(row): idx for idx, row in enumerate(_p6_round)}
    _t4 = np.array([_coord_to_idx[tuple(np.round(row, 12))] for row in _p6_12], dtype=int)
    _t5 = np.array([_coord_to_idx[tuple(np.round(row, 12))] for row in _p6_23], dtype=int)
    _t6 = np.array([_coord_to_idx[tuple(np.round(row, 12))] for row in _p6_13], dtype=int)
    _t6_full = np.column_stack([t[:, 0], t[:, 1], t[:, 2], _t4, _t5, _t6])

    n_midside = _unique_midside.shape[0]
    n_nodes_6 = _p6.shape[0]

    # ══════════════════════════════════════════════════════════════════════
    # Boundary node pairing (periodic boundary conditions)
    # ══════════════════════════════════════════════════════════════════════
    from .compute_boundary_pairs import compute_boundary_pairs
    _bnd_rel = compute_boundary_pairs(_p6)

    # ── Edge lengths ─────────────────────────────────────────────────────
    _e1 = np.sqrt(np.sum((p[t[:, 0]] - p[t[:, 1]]) ** 2, axis=1))
    _e2 = np.sqrt(np.sum((p[t[:, 1]] - p[t[:, 2]]) ** 2, axis=1))
    _e3 = np.sqrt(np.sum((p[t[:, 0]] - p[t[:, 2]]) ** 2, axis=1))
    _all_edges = np.concatenate([_e1, _e2, _e3])

    # ══════════════════════════════════════════════════════════════════════
    # Summary output
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 62)
    print("  FINAL MESH SUMMARY (Type 3 — Non-conforming Rectangular)")
    print("=" * 62)
    print(f"  Domain          : [{minx:.2f}, {maxx:.2f}] x [{miny:.2f}, {maxy:.2f}]")
    print(f"  Domain area     : {domain_area:.4g}")
    print(f"  Covered area    : {covered_area:.4g}  ({100*covered_area/domain_area:.2f}%)")
    print(f"  Grains          : {n_grains}")
    print()
    print(f"  --- 3-node (linear) mesh ---")
    print(f"  Elements        : {n_elems}")
    print(f"  Corner nodes    : {n_nodes}")
    print(f"  Avg elem area   : {np.mean(elem_areas):.4g}  "
          f"(min: {np.min(elem_areas):.4g}, max: {np.max(elem_areas):.4g})")
    print(f"  Target h0       : {h0:.4g}")
    print(f"  Edge length     : mean={np.mean(_all_edges):.4g}  "
          f"min={np.min(_all_edges):.4g}  max={np.max(_all_edges):.4g}")
    print()
    print(f"  --- 6-node (quadratic) mesh ---")
    print(f"  Elements        : {n_elems}")
    print(f"  Total nodes     : {n_nodes_6}  "
          f"(corner: {n_nodes}, midside: {n_midside})")
    print()
    print(f"  --- Element quality ---")
    print(f"  Min quality     : {q_min:.6g}")
    print(f"  Mean quality    : {q_mean:.6g}")
    print(f"  Std quality     : {q_std:.6g}")
    print(f"  Slivers q<0.05  : {int(np.sum(q_all < 0.05))}")
    print(f"  Slivers q<0.10  : {int(np.sum(q_all < 0.10))}")
    print(f"  Slivers q<0.20  : {int(np.sum(q_all < 0.20))}")
    print(f"  Mesh time       : {t_mesh:.2f}s")
    print("=" * 62)

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════
    if FIG_DIR:
        _grain_cmap = plt.cm.get_cmap(_grain_cmap_name, 20)
        _grain_color_idx = getattr(ms, 'grain_color_idx', {})
        _N_CYCLE = getattr(ms, 'N_CYCLE', 20)

        # ── Final mesh figure (colored by grain) ─────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        verts = p[t]
        fcolors = [_grain_cmap(_grain_color_idx.get(el_grain[i], el_grain[i] % _N_CYCLE))
                   for i in range(n_elems)]
        pc = PolyCollection(verts, facecolors=fcolors, edgecolors='none', linewidths=0, alpha=_grain_alpha)
        ax.add_collection(pc)
        # Mesh wireframe overlay
        ax.triplot(p[:, 0], p[:, 1], t, color='k', linewidth=_overlay_lw, alpha=_overlay_alpha)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for _sp in ax.spines.values(): _sp.set_visible(False)
        ax.set_title(f'{_title_info}\n'
                     f'Final Mesh — {n_elems} elements, {n_nodes} nodes\n'
                     f'Quality: min={q_min:.4f}, mean={q_mean:.4f}  |  '
                     f'Area: {covered_area:.1f}/{domain_area:.0f}',
                     fontsize=figure_title_fontsize)
        save(fig, _next_fig_name('final_mesh.png'))

        # ── Final mesh overlaid on original grain boundaries ─────────────
        from matplotlib.patches import Polygon as _MplPoly
        from matplotlib.collections import PatchCollection as _PC_overlay
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # Layer 1: shaded original grain polygons
        _ogb_patches = []
        _ogb_colors = []
        _orig_grains = ms.Grains
        for _i, _g in enumerate(_orig_grains):
            _g_arr = np.asarray(_g) if _g is not None else np.empty((0, 2))
            if _g_arr.size > 0:
                _ogb_patches.append(_MplPoly(_g_arr, closed=True))
                _ogb_colors.append(_grain_cmap(_grain_color_idx.get(_i, _i % _N_CYCLE)))
        if _ogb_patches:
            _ogb_pc = _PC_overlay(_ogb_patches, facecolors=_ogb_colors,
                                  edgecolors='none', linewidths=0, alpha=_grain_alpha)
            ax.add_collection(_ogb_pc)
        # Layer 2: mesh wireframe (dark gray)
        ax.triplot(p[:, 0], p[:, 1], t, color='k', linewidth=_overlay_lw, alpha=_overlay_alpha)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for _sp in ax.spines.values(): _sp.set_visible(False)
        ax.set_title(f'{_title_info}\n'
                     f'Final Mesh on Original Grain Boundaries\n'
                     f'{n_elems} elements, {n_nodes} nodes',
                     fontsize=figure_title_fontsize)
        save(fig, _next_fig_name('final_mesh_on_original_GB.png'))

        # ── Final mesh overlaid on original phase map ────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # Layer 1: EBSD phase map as pixel raster
        _pm_coords = ms.OriginalDataCoordinateList
        _pm_phases = ms.OriginalDataPhase
        _pm_xs = np.unique(np.round(_pm_coords[:, 0], 6))
        _pm_ys = np.unique(np.round(_pm_coords[:, 1], 6))
        _pm_dx = _pm_xs[1] - _pm_xs[0] if len(_pm_xs) > 1 else 1.0
        _pm_dy = _pm_ys[1] - _pm_ys[0] if len(_pm_ys) > 1 else 1.0
        _pm_x_edges = np.concatenate([_pm_xs - _pm_dx/2, [_pm_xs[-1] + _pm_dx/2]])
        _pm_y_edges = np.concatenate([_pm_ys - _pm_dy/2, [_pm_ys[-1] + _pm_dy/2]])
        _pm_grid = np.full((len(_pm_ys), len(_pm_xs)), np.nan)
        for _k in range(len(_pm_coords)):
            _ix = np.searchsorted(_pm_xs, _pm_coords[_k, 0])
            _iy = np.searchsorted(_pm_ys, _pm_coords[_k, 1])
            if _ix < len(_pm_xs) and _iy < len(_pm_ys):
                _pm_grid[_iy, _ix] = _pm_phases[_k]
        _pm_unique_phases = np.unique(_pm_phases[_pm_phases > 0]).astype(int)
        from matplotlib.colors import ListedColormap as _LCM_pm
        _pm_cmap = _LCM_pm([_phase_base_colors[i % len(_phase_base_colors)]
                             for i in range(len(_pm_unique_phases))])
        ax.pcolormesh(_pm_x_edges, _pm_y_edges, _pm_grid, cmap=_pm_cmap,
                      vmin=0.5, vmax=len(_pm_unique_phases) + 0.5, alpha=_phase_alpha)
        # Layer 2: mesh wireframe with transparency
        ax.triplot(p[:, 0], p[:, 1], t, color='k', linewidth=_overlay_lw, alpha=_overlay_alpha)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for _sp in ax.spines.values(): _sp.set_visible(False)
        # Phase legend
        from matplotlib.patches import Patch as _PatchPM
        _pm_legend = []
        for _ph in sorted(_pm_unique_phases):
            _rgba = _mcolors_global.to_rgba(_phase_base_colors[(_ph - 1) % len(_phase_base_colors)])
            _ph_name = ms.PhaseName[_ph-1] if ms.PhaseName and _ph-1 < len(ms.PhaseName) and ms.PhaseName[_ph-1] else f"Phase {_ph}"
            _pm_legend.append(_PatchPM(facecolor=_rgba, edgecolor='k', label=f'Phase {_ph}: {_ph_name}'))
        ax.legend(handles=_pm_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=figure_fontsize)
        ax.set_title(f'{_title_info}\n'
                     f'Final Mesh on Phase Map\n'
                     f'{n_elems} elements, {n_nodes} nodes',
                     fontsize=figure_title_fontsize)
        save(fig, _next_fig_name('final_mesh_on_original_phase_map.png'))

        # ── Phase-colored mesh figure ────────────────────────────────────
        _phase_base_rgba = {ph+1: _mcolors_global.to_rgba(c)
                           for ph, c in enumerate(_phase_base_colors)}
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fcolors_ph = [_phase_base_rgba.get(el_phase[i], (0.5, 0.5, 0.5, 1.0))
                      for i in range(n_elems)]
        pc = PolyCollection(verts, facecolors=fcolors_ph, edgecolors='none', linewidths=0, alpha=_phase_alpha)
        ax.add_collection(pc)
        # Mesh wireframe overlay
        ax.triplot(p[:, 0], p[:, 1], t, color='k', linewidth=_overlay_lw, alpha=_overlay_alpha)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for _sp in ax.spines.values(): _sp.set_visible(False)

        # Phase legend
        _phase_legend = []
        for ph in sorted(set(el_phase)):
            rgba = _phase_base_rgba.get(ph, (0.5, 0.5, 0.5, 1.0))
            n_ph = int(np.sum(el_phase == ph))
            _ph_name = ms.PhaseName[ph-1] if ms.PhaseName and ph-1 < len(ms.PhaseName) and ms.PhaseName[ph-1] else f"Phase {ph}"
            _phase_legend.append(_Patch(facecolor=rgba, edgecolor='k',
                                        label=f'Phase {ph}: {_ph_name}'))
        ax.legend(handles=_phase_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=figure_fontsize)
        ax.set_title(f'{_title_info}\n'
                     f'Final Mesh — Phase Colored',
                     fontsize=figure_title_fontsize)
        save(fig, os.path.join('diagnostics', _next_fig_name('final_mesh_phase.png')))

    # ══════════════════════════════════════════════════════════════════════
    # Store all mesh results on ms
    # ══════════════════════════════════════════════════════════════════════

    # 3-node mesh
    ms.ThreeNodeCoordinateList = p.copy()
    ms.ThreeNodeElementIndexList = t.copy()

    # 6-node mesh (1-based indices for AEH solver compatibility)
    ms.SixNodeCoordinateList = _p6.copy()
    ms.SixNodeElementIndexList = _t6_full.copy() + 1
    ms.NumberElements = n_elems
    ms.NumberNodes = n_nodes_6

    # Boundary pairing
    ms.BoundaryNodeRelationsList = _bnd_rel.copy()

    # Per-element properties
    ms.ElementPhases = el_phase.copy()
    ms.ElementGrains = el_grain.copy()

    # Per-grain properties
    ms.GrainPhases = grain_phases.copy()
    ms.GrainAngles = grain_angles.copy()
    ms.GrainsElements = {k: v.copy() for k, v in grain_elsets.items()}

    # Overwrite DataCoordinateList/DataPhase/DataEulerAngle with per-element values
    ms.DataCoordinateList = centroids.copy()
    ms.DataPhase = el_phase.copy()
    ms.DataEulerAngle = el_angle.copy()

    # Store meshed grain boundaries
    ms.GrainsMeshed = [np.asarray(g).copy() if g is not None else None for g in valid_gs]

    # Mesh type
    ms.CurrentMeshType = 3

    print(f"\n  Stored mesh results on ms: {ms.NumberElements} elements, "
          f"{ms.NumberNodes} nodes (6-node), "
          f"{len(ms.GrainsElements)} grains")

    # ══════════════════════════════════════════════════════════════════════
    # Write to log.md
    # ══════════════════════════════════════════════════════════════════════
    if log_path and vl in ("medium", "high"):
        with open(log_path, "a") as lf:
            lf.write("## Stage 2 — Mesh Generation (Type 3: Non-conforming Rectangular)\n\n")
            lf.write(f"| {'Property':<22s} | {'Value':<36s} |\n")
            lf.write(f"|{'-'*24}|{'-'*38}|\n")
            lf.write(f"| {'Mesh type':<22s} | {'3 (non-conforming uniform)':<36s} |\n")
            lf.write(f"| {'Target h0':<22s} | {h0:<36.4g} |\n")
            lf.write(f"| {'Grid size':<22s} | {f'{xs} x {ys}':<36s} |\n")
            lf.write(f"| {'Elements':<22s} | {n_elems:<36d} |\n")
            lf.write(f"| {'Nodes (3-node)':<22s} | {n_nodes:<36d} |\n")
            lf.write(f"| {'Nodes (6-node)':<22s} | {n_nodes_6:<36d} |\n")
            lf.write(f"| {'Boundary pairs':<22s} | {int(np.sum(_bnd_rel > 0)):<36d} |\n")
            lf.write(f"| {'q_min':<22s} | {q_min:<36.6f} |\n")
            lf.write(f"| {'q_mean':<22s} | {q_mean:<36.6f} |\n")
            lf.write(f"| {'Domain area':<22s} | {domain_area:<36.4g} |\n")
            lf.write(f"| {'Covered area':<22s} | {covered_area:<36.4g} |\n")
            lf.write(f"| {'Mesh time':<22s} | {f'{t_mesh:.2f}s':<36s} |\n")
            lf.write("\n")

            # Figure list
            if FIG_DIR:
                lf.write("### Figures\n\n")
                figs = sorted([f for f in os.listdir(FIG_DIR) if f.endswith('.png')])
                for fn in figs:
                    lf.write(f"- `{fn}`\n")
                lf.write("\n")

            lf.write("---\n\n")

    # Update figure counter
    ms.fig_count = fig_num[0]

    return ms
