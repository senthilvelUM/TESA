"""
Generate Stage 1 visualization figures for raw EBSD microstructure.

Figures produced (numbered sequentially):
  1_phase_map.png                  — Phase map as filled pixels
  2_euler_angles.png               — 3-panel Euler angles with grain boundaries
  3_grains_numbered.png  — Grain boundaries with grain number labels
  4_grains_phase.png     — Grain boundaries colored by phase
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors_global
import matplotlib.patheffects as _mpe
import colorsys as _cs
from matplotlib.patches import Polygon as MplPolygon, Patch as _Patch
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
from collections import defaultdict

from .dpoly import dpoly
from .find_grain_holes import find_grain_holes




def _save_fig(fig, run_dir, name, console_on=True, show_figures=False, figure_pause=1.0, figure_dpi=150):
    """
    Save a figure to the results folder, optionally display on screen, then close.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    run_dir : str
        Directory to save the figure into.
    name : str
        Filename (e.g., 'phase_map.png').
    console_on : bool, optional
        If True, print the saved filename to the console. Default True.
    show_figures : bool, optional
        If True, display the figure on screen before closing. Default False.
    figure_pause : float, optional
        Seconds to pause when displaying on screen. Default 1.0.
    figure_dpi : int, optional
        Resolution for the saved image. Default 150.
    """
    # Remove tick marks and labels from plot axes (skip colorbars)
    for ax in fig.get_axes():
        if ax.get_label() != '<colorbar>':
            ax.set_xticks([])
            ax.set_yticks([])
    path = os.path.join(run_dir, name)
    fig.savefig(path, dpi=figure_dpi, bbox_inches='tight')
    # Display on screen if requested
    if show_figures:
        fig.show()
        plt.pause(figure_pause)
    plt.close(fig)
    if console_on:
        print(f"  Saved: {name}")


def _find_interior_label_point(g_arr, hole_arrs=None):
    """
    Find a good interior point for labeling a grain polygon.

    For grains with holes, avoids the hole regions. Returns the centroid
    if it lies inside the grain and outside all holes; otherwise samples a
    grid and returns the most-interior valid point.

    Parameters
    ----------
    g_arr : ndarray, shape (n_vertices, 2)
        Grain boundary polygon vertices (not closed).
    hole_arrs : list of ndarray or None, optional
        List of hole polygon vertex arrays, each shape (n_hole_verts, 2).
        If None, no holes are considered.

    Returns
    -------
    point : ndarray, shape (2,)
        Interior point suitable for placing a label.
    """

    # Close the grain polygon
    closed = np.vstack([g_arr, g_arr[0:1, :]])

    # Build closed polygons for holes
    hole_closed = []
    if hole_arrs:
        for h_arr in hole_arrs:
            hole_closed.append(np.vstack([h_arr, h_arr[0:1, :]]))

    def _is_valid(pts):
        """Check points are inside grain and outside all holes."""
        fd = dpoly(pts, closed)
        valid = fd < 0  # inside grain
        for hc in hole_closed:
            fd_h = dpoly(pts, hc)
            valid &= fd_h > 0  # outside hole
        return valid, fd

    # Try centroid first
    centroid = g_arr.mean(axis=0)
    valid_c, _ = _is_valid(centroid.reshape(1, -1))
    if valid_c[0]:
        return centroid

    # Sample a grid and find the best valid point (most interior)
    xmin, ymin = g_arr.min(axis=0)
    xmax, ymax = g_arr.max(axis=0)
    nx, ny = 30, 30
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(gx, gy)
    candidates = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    valid, fd_vals = _is_valid(candidates)
    if not np.any(valid):
        return centroid  # fallback

    # Pick the point most interior to the grain (most negative fd)
    best = np.argmin(fd_vals[valid])
    return candidates[valid][best]


def plot_ebsd(ms, run_dir, ebsd_file=None, log_path=None, settings=None):
    """
    Generate Stage 1 visualization figures for the raw EBSD microstructure.

    Parameters
    ----------
    ms : Microstructure
        Must have EBSD data loaded (via gui_load / read_ebsd).
    run_dir : str
        Directory for saving figures.
    ebsd_file : str or None
        Path to the EBSD file (for figure titles).
    log_path : str or None
        Path to log.md file. If provided, figure list is appended.
    settings : dict or None
        Global settings (show_figures, verbose, etc.). If None, defaults are used.

    Returns
    -------
    grain_color : callable
        Function mapping grain index to RGBA color.
    grain_color_idx : dict
        Grain index to color slot mapping.
    N_CYCLE : int
        Number of colors in the cycle.
    fig_count : int
        Number of figures saved (for continuing the counter in later stages).
    """
    # Use defaults if settings not provided
    if settings is None:
        settings = {}
    if "verbose_console" not in settings:
        print("  WARNING: 'verbose_console' not in settings, using default 'medium'")
    vc = settings.get("verbose_console", "medium")
    console_on = vc in ("medium", "high")
    if "show_figures" not in settings:
        print("  WARNING: 'show_figures' not in settings, using default False")
    show_figures = settings.get("show_figures", False)
    if "figure_pause" not in settings:
        print("  WARNING: 'figure_pause' not in settings, using default 1.0")
    figure_pause = settings.get("figure_pause", 1.0)
    if "figure_dpi" not in settings:
        print("  WARNING: 'figure_dpi' not in settings, using default 150")
    figure_dpi = settings.get("figure_dpi", 150)
    # Phase color palette from global settings (fall back to default if missing)
    if "phase_colors" not in settings:
        print("  WARNING: 'phase_colors' not in settings, using default palette")
    _phase_base_colors = settings.get("phase_colors",
        ['red', 'lime', 'tab:blue', 'tab:orange', 'tab:purple',
         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])
    # Grain colormap from global settings
    if "grain_colormap" not in settings:
        print("  WARNING: 'grain_colormap' not in settings, using default 'tab20'")
    _grain_cmap_name = settings.get("grain_colormap", "tab20")
    _grain_alpha = settings.get("grain_colormap_alpha", 0.9)
    _phase_alpha = settings.get("phase_colors_alpha", 0.9)
    figure_title_fontsize = settings.get("figure_title_fontsize", 14)
    figure_fontsize = settings.get("figure_fontsize", 12)

    if console_on:
        print("\n── Stage 1 visualization ──")

    # EBSD file name and map dimensions for figure titles
    import os as _os
    ebsd_name = _os.path.basename(ebsd_file) if ebsd_file else "EBSD"
    coords = ms.OriginalDataCoordinateList
    nx_grid = len(np.unique(coords[:, 0]))
    ny_grid = len(np.unique(coords[:, 1]))
    _title_info = f"{ebsd_name} ({nx_grid}x{ny_grid})"

    # Create Stage 1 figures subfolder
    fig_dir = _os.path.join(run_dir, "microstructure")
    _os.makedirs(fig_dir, exist_ok=True)

    # Image counter — incremented for each saved figure
    fig_num = [0]
    def _next_fig_name(name):
        fig_num[0] += 1
        return name

    # Extract data from Microstructure
    phases = ms.OriginalDataPhase
    eulers = ms.OriginalDataEulerAngle
    n_pts = len(phases)
    unique_phases = np.unique(phases)

    # Build pixel grid coordinates (shared by phase map and euler angle figures)
    xs = np.sort(np.unique(coords[:, 0]))
    ys = np.sort(np.unique(coords[:, 1]))
    dx = xs[1] - xs[0] if nx_grid > 1 else 1.0
    dy = ys[1] - ys[0] if ny_grid > 1 else 1.0
    x_edges = np.concatenate([xs - dx/2, [xs[-1] + dx/2]])
    y_edges = np.concatenate([ys - dy/2, [ys[-1] + dy/2]])

    # Build phase color map
    _phase_color_map = {ph+1: _phase_base_colors[ph % len(_phase_base_colors)]
                        for ph in range(20)}
    _phase_color_map[0] = '#999999'  # unassigned / phase 0

    # ── Figure: Phase map (filled pixels) ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    phase_grid = np.full((ny_grid, nx_grid), np.nan)
    for k in range(len(coords)):
        ix = np.searchsorted(xs, coords[k, 0])
        iy = np.searchsorted(ys, coords[k, 1])
        if ix < nx_grid and iy < ny_grid:
            phase_grid[iy, ix] = phases[k]
    cmap_ph = ListedColormap([_phase_base_colors[i % len(_phase_base_colors)]
                              for i in range(len(unique_phases[unique_phases > 0]))])
    _n_active_phases = len(unique_phases[unique_phases > 0])
    ax.pcolormesh(x_edges, y_edges, phase_grid, cmap=cmap_ph, vmin=0.5, vmax=_n_active_phases + 0.5, alpha=_phase_alpha)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_aspect('equal')
    ax.set_title(f'{_title_info}\n{len(unique_phases)} Phases', fontsize=figure_title_fontsize)
    # Add legend with point counts and fractions
    from matplotlib.patches import Patch as _PatchPM
    _pm_legend = []
    for ph in sorted(unique_phases[unique_phases > 0].astype(int)):
        c = _phase_color_map.get(ph, '#999999')
        _ph_name = ms.PhaseName[ph-1] if ms.PhaseName and ph-1 < len(ms.PhaseName) and ms.PhaseName[ph-1] else f"Phase {ph}"
        _pm_legend.append(_PatchPM(facecolor=c, edgecolor='k',
                                    label=f'Phase {ph}: {_ph_name}'))
    ax.legend(handles=_pm_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=figure_fontsize)
    _save_fig(fig, fig_dir, _next_fig_name('phase_map.png'), console_on, show_figures, figure_pause, figure_dpi)

    # ── Figure: Euler angles as filled pixels (3 panels) ───────────────
    euler_grids = [np.full((ny_grid, nx_grid), np.nan) for _ in range(3)]
    for k in range(n_pts):
        ix = np.searchsorted(xs, coords[k, 0])
        iy = np.searchsorted(ys, coords[k, 1])
        if ix < nx_grid and iy < ny_grid:
            for col in range(3):
                euler_grids[col][iy, ix] = eulers[k, col]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    euler_labels_px = ['$\\varphi_1$ (rad)', '$\\Phi$ (rad)', '$\\varphi_2$ (rad)']
    for col in range(3):
        ax = axes[col]
        # Display as filled pixel image with hsv colormap
        im = ax.imshow(euler_grids[col],
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                       origin='lower', aspect='equal', interpolation='nearest',
                       cmap='hsv')
        plt.colorbar(im, ax=ax, shrink=0.7)
        # Overlay grain boundaries as thin black lines
        for g in ms.Grains:
            if g is not None and len(g) > 0:
                g_arr = np.asarray(g)
                closed = np.vstack([g_arr, g_arr[0:1, :]])
                ax.plot(closed[:, 0], closed[:, 1], 'k-', linewidth=0.4)
        ax.set_title(f'{euler_labels_px[col]}', fontsize=figure_title_fontsize)
    fig.suptitle(f'{_title_info}\nEuler Angles', fontsize=figure_title_fontsize)
    fig.tight_layout()
    _save_fig(fig, fig_dir, _next_fig_name('euler_angles.png'), console_on, show_figures, figure_pause, figure_dpi)

    # ── Neighbor-aware grain color assignment ──────────────────────────
    _grain_cmap = plt.cm.get_cmap(_grain_cmap_name, 20)
    N_CYCLE = 10  # number of colors to cycle through

    # Build adjacency: two grains are neighbors if their boundary polygons
    # have points within 1.5 pixels of each other.
    # Uses a single cKDTree over all boundary points for O(n) instead of O(n²).
    _grain_neighbors = defaultdict(set)
    _grain_bnd_arrs = {}
    for _i, _g in enumerate(ms.Grains):
        if _g is not None and len(_g) > 0:
            _grain_bnd_arrs[_i] = np.asarray(_g)

    _adj_dist = 1.5  # pixels

    # Concatenate all boundary points and tag each with its grain index
    _all_bnd_pts = []
    _all_bnd_ids = []
    for _i, _arr in _grain_bnd_arrs.items():
        _all_bnd_pts.append(_arr)
        _all_bnd_ids.append(np.full(len(_arr), _i, dtype=int))
    _all_bnd_pts = np.vstack(_all_bnd_pts)
    _all_bnd_ids = np.concatenate(_all_bnd_ids)

    # Single tree query: find all point pairs within adjacency distance
    _global_tree = cKDTree(_all_bnd_pts)
    _pairs = _global_tree.query_pairs(r=_adj_dist)
    for _pa, _pb in _pairs:
        _gi, _gj = _all_bnd_ids[_pa], _all_bnd_ids[_pb]
        if _gi != _gj:
            _grain_neighbors[_gi].add(_gj)
            _grain_neighbors[_gj].add(_gi)

    # Print neighbor list to log
    if console_on:
        print("\n  -- Grain Neighbor List --")
        _sorted_keys_nb = sorted(_grain_bnd_arrs.keys())
        _n_total_nb = len(_sorted_keys_nb)
        for _idx_nb, _i in enumerate(_sorted_keys_nb):
            _n_neighbors = len(_grain_neighbors[_i])
            print(f"\r  Grain {_i}/{_n_total_nb} ({_n_neighbors} neighbors)        ", end='', flush=True)
        _total_pairs = sum(len(v) for v in _grain_neighbors.values()) // 2
        print(f"\r  Total neighbor pairs: {_total_pairs}                              ")
        print()

    # Greedy graph coloring: for each grain, pick the first color (starting
    # from i % N_CYCLE) that no already-colored neighbor uses
    grain_color_idx = {}
    for _i in range(len(ms.Grains)):
        if _i not in _grain_bnd_arrs:
            grain_color_idx[_i] = _i % N_CYCLE
            continue
        _used = {grain_color_idx[_nb] for _nb in _grain_neighbors[_i]
                 if _nb in grain_color_idx}
        _c = _i % N_CYCLE
        for _ in range(N_CYCLE):
            if _c not in _used:
                break
            _c = (_c + 1) % N_CYCLE
        grain_color_idx[_i] = _c

    # Grain color function
    def grain_color(i):
        return _grain_cmap(grain_color_idx.get(i, i % N_CYCLE))

    # Check for neighbor conflicts in color assignments
    _n_conflicts = 0
    _sorted_grain_keys = sorted(_grain_bnd_arrs.keys())
    _n_total_grains = len(_sorted_grain_keys)
    for _idx_g, _i in enumerate(_sorted_grain_keys):
        _ci = grain_color_idx[_i]
        _conflicts = [int(_nb) for _nb in _grain_neighbors[_i]
                      if grain_color_idx.get(_nb) == _ci]
        if _conflicts:
            if console_on:
                # Print conflict on its own line (clear the progress line first)
                print(f"\r  Grain {_i}: color slot {_ci}  *** CONFLICT with {_conflicts}              ")
            _n_conflicts += len(_conflicts)
        elif console_on:
            print(f"\r  Assigning grain colors... {_idx_g + 1}/{_n_total_grains}        ", end='', flush=True)
    if console_on:
        print(f"\r  Grain colors assigned. Neighbor conflicts: {_n_conflicts // 2}              ")
        print()

    # ── Figure: Grain boundaries with grain numbers ──────────────────

    # Find hole relationships on original (non-smoothed) grains
    import io as _io_holes, contextlib as _ctx_holes
    _grains_list = [np.asarray(g) for g in ms.Grains if g is not None and len(g) > 0]

    # Build index mapping: _grains_list[k] corresponds to ms.Grains[_gidx_map[k]]
    _gidx_map = []
    for i, g in enumerate(ms.Grains):
        if g is not None and len(g) > 0:
            _gidx_map.append(i)

    _f_holes = _io_holes.StringIO()
    with _ctx_holes.redirect_stdout(_f_holes):
        _Holes, _holes_flat = find_grain_holes(_grains_list)

    # Build a map: grain_index -> list of enclosed grain indices
    _holes_map = {}
    for k, hole_list in enumerate(_Holes):
        if hole_list:
            enc_idx = _gidx_map[k]
            _holes_map[enc_idx] = [_gidx_map[h] for h in hole_list]

    # Build grain patches (shared by grains.png and grains_numbered.png)
    _grain_polys = []
    _grain_colors_list = []
    for i, g in enumerate(ms.Grains):
        if g is not None and len(g) > 0:
            _grain_polys.append(np.asarray(g))
            _grain_colors_list.append(grain_color(i))

    # ── Figure: Grain boundaries without numbers (grains.png) ─────────
    fig, ax = plt.subplots(figsize=(9, 9))
    if _grain_polys:
        _patches_plain = [MplPolygon(g, closed=True) for g in _grain_polys]
        pc_plain = PatchCollection(_patches_plain, facecolors=_grain_colors_list,
                                   edgecolors='k', linewidths=0.4, alpha=_grain_alpha)
        ax.add_collection(pc_plain)
    ax.set_xlim(coords[:,0].min()-0.5, coords[:,0].max()+0.5)
    ax.set_ylim(coords[:,1].min()-0.5, coords[:,1].max()+0.5)
    ax.set_aspect('equal')
    ax.set_title(f'{_title_info}\n{len(ms.Grains)} Grains',
                 fontsize=figure_title_fontsize)
    _save_fig(fig, fig_dir, _next_fig_name('grains.png'), console_on, show_figures, figure_pause, figure_dpi)

    # ── Figure: Grain boundaries with grain numbers ───────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    if _grain_polys:
        _patches_num = [MplPolygon(g, closed=True) for g in _grain_polys]
        pc_num = PatchCollection(_patches_num, facecolors=_grain_colors_list,
                                 edgecolors='k', linewidths=0.4, alpha=_grain_alpha)
        ax.add_collection(pc_num)

    # Compute hole-aware label positions for each grain
    _centroids = []
    for i, g in enumerate(ms.Grains):
        if g is not None and len(g) > 0:
            g_arr = np.asarray(g)
            # Get hole polygons for this grain if any
            hole_arrs = None
            if i in _holes_map:
                hole_arrs = []
                for h_idx in _holes_map[i]:
                    h_g = ms.Grains[h_idx]
                    if h_g is not None and len(h_g) > 0:
                        hole_arrs.append(np.asarray(h_g))
            label_pt = _find_interior_label_point(g_arr, hole_arrs)
            _centroids.append((i, label_pt, g_arr))
        else:
            _centroids.append((i, None, None))

    # Place labels at interior points
    for i, cxy, g_arr in _centroids:
        if cxy is not None:
            ax.text(float(cxy[0]), float(cxy[1]), str(i), ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='white',
                    path_effects=[_mpe.withStroke(linewidth=2, foreground='black')])
    ax.set_xlim(coords[:,0].min()-0.5, coords[:,0].max()+0.5)
    ax.set_ylim(coords[:,1].min()-0.5, coords[:,1].max()+0.5)
    ax.set_aspect('equal')
    ax.set_title(f'{_title_info}\n{len(ms.Grains)} Grains',
                 fontsize=figure_title_fontsize)
    _save_fig(fig, fig_dir, _next_fig_name('grains_numbered.png'), console_on, show_figures, figure_pause, figure_dpi)

    # ── Figure 03a3: Phase-colored grain map (Euler-angle shading) ─────

    # Phase base RGBA via global _phase_base_colors
    _phase_base_rgba = {ph+1: _mcolors_global.to_rgba(c) for ph, c in enumerate(_phase_base_colors)}

    # Assign each grain a phase and Euler angle via nearest EBSD point to centroid
    _grain_phase = []
    _grain_euler = []
    for i, g in enumerate(ms.Grains):
        if g is not None and len(g) > 0:
            g_arr = np.asarray(g)
            centroid = g_arr.mean(axis=0)
            dists = np.sqrt((coords[:, 0] - centroid[0])**2 +
                            (coords[:, 1] - centroid[1])**2)
            nearest = np.argmin(dists)
            _grain_phase.append(int(phases[nearest]))
            _grain_euler.append(eulers[nearest].copy())
        else:
            _grain_phase.append(0)
            _grain_euler.append(np.zeros(3))

    # Build phase-aware colors: base color from phase, shade from Euler angles
    _colors_phase = []
    for i in range(len(ms.Grains)):
        ph = _grain_phase[i]
        ea = _grain_euler[i]
        base_rgba = _phase_base_rgba.get(ph, (0.5, 0.5, 0.5, 1.0))
        base_r, base_g, base_b = base_rgba[0], base_rgba[1], base_rgba[2]
        # Convert base RGB to HSV, then modulate sat/val with Euler angles
        base_h, base_s, base_v = _cs.rgb_to_hsv(base_r, base_g, base_b)
        # Vary saturation by +/-0.15 using Euler angle 1
        sat = base_s + ((ea[0] % np.pi) / np.pi * 0.3 - 0.15)
        sat = max(0.15, min(1.0, sat))
        # Vary brightness by +/-0.15 using Euler angle 2
        val = base_v + ((ea[1] % np.pi) / np.pi * 0.3 - 0.15)
        val = max(0.4, min(1.0, val))
        r, g_c, b = _cs.hsv_to_rgb(base_h, sat, val)
        _colors_phase.append((r, g_c, b, _phase_alpha))

    fig, ax = plt.subplots(figsize=(9, 9))
    _patches_03a3 = []
    _fcolors_03a3 = []
    for i, g in enumerate(ms.Grains):
        if g is not None and len(g) > 0:
            g_arr = np.asarray(g)
            _patches_03a3.append(MplPolygon(g_arr, closed=True))
            _fcolors_03a3.append(_colors_phase[i])
    if _patches_03a3:
        pc3 = PatchCollection(_patches_03a3, facecolors=_fcolors_03a3,
                              edgecolors='k', linewidths=0.4)
        ax.add_collection(pc3)

    # Add legend for phases with grain counts
    _phase_legend = []
    _unique_grain_phases = sorted(set(p for p in _grain_phase if p > 0))
    for ph in _unique_grain_phases:
        rgba = _phase_base_rgba.get(ph, (0.5, 0.5, 0.5, 1.0))
        _ph_name = ms.PhaseName[ph-1] if ms.PhaseName and ph-1 < len(ms.PhaseName) and ms.PhaseName[ph-1] else f"Phase {ph}"
        _phase_legend.append(_Patch(facecolor=rgba, edgecolor='k',
                                     label=f'Phase {ph}: {_ph_name}'))
    ax.legend(handles=_phase_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=figure_fontsize)

    # Reuse the hole-aware label positions from 03a2
    for i, cxy, g_arr in _centroids:
        if cxy is not None:
            ax.text(float(cxy[0]), float(cxy[1]), str(i), ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='white',
                    path_effects=[_mpe.withStroke(linewidth=2, foreground='black')])
    ax.set_xlim(coords[:,0].min()-0.5, coords[:,0].max()+0.5)
    ax.set_ylim(coords[:,1].min()-0.5, coords[:,1].max()+0.5)
    ax.set_aspect('equal')
    ax.set_title(f'{_title_info}\n{len(ms.Grains)} Grains — Phase Colored',
                 fontsize=figure_title_fontsize)
    _save_fig(fig, fig_dir, _next_fig_name('grains_phase.png'), console_on, show_figures, figure_pause, figure_dpi)

    # ── Append figure list to log.md ───────────────────────────────────
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write("### Figures\n\n")
            figs = sorted([fn for fn in os.listdir(fig_dir) if fn.endswith('.png')])
            for fn in figs:
                size_kb = os.path.getsize(os.path.join(fig_dir, fn)) / 1024
                f.write(f"- `{fn}` ({size_kb:.0f} KB)\n")
            f.write("\n---\n\n")

    if console_on:
        print(f"  Stage 1 figures saved to: {fig_dir}")

    return grain_color, grain_color_idx, N_CYCLE, fig_num[0]
