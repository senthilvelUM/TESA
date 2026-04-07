"""
Type 1: Conforming non-uniform mesh.

Generates a mesh that conforms to grain boundaries using:
  - RBF grain boundary smoothing
  - Mesh size function (curvature + medial axis)
  - Initial mesh with fixed grain boundary nodes
  - distmesh2d0 iterative refinement per grain
  - Cleanup, grain assignment, and 6-node element promotion

This code is extracted verbatim from the tested master pipeline
(test_mesh_gap_debug_master.py), with minimal changes for settings.
"""

import sys
import os
import io
import contextlib
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors_global
from matplotlib.collections import PolyCollection
from scipy.spatial import Delaunay
from collections import defaultdict as _ddict

from .grain_RBF_contours_2 import grain_RBF_contours_2
from .setup_mesh_size_function import setup_mesh_size_function
from .gradient_limiting import gradient_limiting
from .create_initial_mesh import create_initial_mesh
from .DecimatePoly import DecimatePoly
from .distmesh2d0 import distmesh2d0
from .find_grain_holes import find_grain_holes
from .dpoly import dpoly
from .triarea import triarea
from .simpqual import simpqual

# ── Internal mesh parameters (rarely need changing) ─────────────────────────
# These control the distmesh2d0 iteration behavior. Exposed here so they are
# visible and documented in one place rather than buried in the code.
_MESH_INTERNALS = {
    "aggressive_fraction": 0.6,  # Fraction of min_iter spent in aggressive mode (floor of 10 iterations)
    "consecutive_required": 3,    # Both q_min and q_mean targets must hold for N consecutive iterations
}


def mesh_conforming(ms, job, run_dir=None, log_path=None, settings=None):
    """
    Generate a Type 1 conforming non-uniform mesh with full diagnostic figures and logging.

    Parameters
    ----------
    ms : Microstructure
        Must have EBSD data and grain boundaries loaded (via load_ebsd).
    job : dict
        Job dictionary with advanced_mesh_params, bsep, etc.
    run_dir : str or None
        Results directory for saving figures and mesh files.
    log_path : str or None
        Path to log.md file.
    settings : dict or None
        Global settings (verbose, show_figures, figure_pause, random_seed).

    Returns
    -------
    ms : Microstructure
        Updated with conforming mesh fields.
    """
    # Use defaults if settings not provided
    if settings is None:
        settings = {}
    if "verbose_console" not in settings:
        print("  WARNING: 'verbose_console' not in settings, using default 'medium'")
    vc = settings.get("verbose_console", "medium")
    if "verbose_log" not in settings:
        print("  WARNING: 'verbose_log' not in settings, using default 'medium'")
    vl = settings.get("verbose_log", "medium")
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
    if "random_seed" not in settings:
        print("  WARNING: 'random_seed' not in settings, using default 42")
    random_seed = settings.get("random_seed", 42)
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
    _overlay_alpha = settings.get("mesh_overlay_alpha", 0.5)
    _overlay_lw = settings.get("mesh_overlay_linewidth", 0.3)

    # For "high" log: capture all console output to log.md via TeeWriter
    _tee = None
    if vl == "high" and log_path is not None:
        import sys as _sys
        _console_log_path = os.path.join(run_dir if run_dir else ".", '_console_output.txt')
        class _TeeWriter:
            """Write to both stdout and a log file simultaneously."""
            def __init__(self, log_path):
                self.terminal = _sys.stdout
                self.log = open(log_path, 'w')
            def write(self, msg):
                self.terminal.write(msg)
                self.log.write(msg)
                self.log.flush()
            def flush(self):
                self.terminal.flush()
                self.log.flush()
            def close(self):
                self.log.close()
        _tee = _TeeWriter(_console_log_path)
        _sys.stdout = _tee

    # Alias Part 1 variables to match the master script's variable names
    ebsd_name = os.path.basename(job["ebsd_file"])
    ebsd_path = job["ebsd_file"]
    # Build title info string matching plot_ebsd format
    _coords_mc = ms.OriginalDataCoordinateList
    _nx_mc = int(round(_coords_mc[:,0].max() - _coords_mc[:,0].min())) + 1
    _ny_mc = int(round(_coords_mc[:,1].max() - _coords_mc[:,1].min())) + 1
    _title_info = f"{ebsd_name} ({_nx_mc}x{_ny_mc})"
    # Save figures to Stage 2 subfolder (iteration figures go inside it)
    FIG_DIR = os.path.join(run_dir, "mesh") if run_dir is not None else "."
    os.makedirs(FIG_DIR, exist_ok=True)
    DIAG_DIR = os.path.join(FIG_DIR, "diagnostics")
    os.makedirs(DIAG_DIR, exist_ok=True)
    coords = ms.OriginalDataCoordinateList
    _grain_color = ms.grain_color
    _grain_color_idx = ms.grain_color_idx
    _N_CYCLE = ms.N_CYCLE
    bsep = job.get("grain_boundary_resolution", 1.0)

    # Sequential figure counter — continues from where plot_ebsd left off
    _stage1_fig_count = getattr(ms, 'fig_count', 0)
    ms._stage1_fig_count = _stage1_fig_count
    fig_num = [_stage1_fig_count]
    def _next_fig_name(name):
        fig_num[0] += 1
        return name

    # Local save function (replaces the master's global save)
    # Diagnostic figures are saved silently without display or console output
    def save(fig, name):
        path = os.path.join(FIG_DIR, name)
        fig.savefig(path, dpi=figure_dpi, bbox_inches='tight')
        _is_diag = name.startswith('diagnostics')
        if show_figures and not _is_diag:
            fig.show()
            plt.pause(figure_pause)
        plt.close(fig)
        if console_on and not _is_diag:
            print(f"  Saved: {name}")

    # Display a saved PNG on screen (master used a persistent window)
    def _show_from_file(path):
        if show_figures:
            img = plt.imread(path)
            fig_tmp, ax_tmp = plt.subplots(figsize=(10, 10))
            ax_tmp.imshow(img)
            ax_tmp.axis('off')
            fig_tmp.show()
            plt.pause(figure_pause)
            plt.close(fig_tmp)

    # ═══════════════════════════════════════════════════════════════════════
    # Everything below is extracted verbatim from master.py lines 498–2027
    # ═══════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 2: Grain boundary smoothing (RBF)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n  ---- RBF grain smoothing ----")
    # Save original grains for comparison
    grains_original = [np.asarray(g).copy() for g in ms.Grains if g is not None]
    ms = grain_RBF_contours_2(ms, bsep=bsep, verbose=console_on)
    
    valid_gs = [np.asarray(g) for g in ms.GrainsSmoothed
                if g is not None and np.asarray(g).size > 0]
    ms.GrainsSmoothed = valid_gs
    n_empty = len(ms.GrainsSmoothed) - len(valid_gs) if hasattr(ms, 'GrainsSmoothed') else 0
    print(f"  Smoothed grains: {len(valid_gs)} valid"
          f"{f' ({n_empty} empty removed)' if n_empty > 0 else ''}")
    
    # ── Overlay — original grains (shaded) with smoothed boundaries ────────
    from matplotlib.patches import Polygon as _MplPoly_ov
    from matplotlib.collections import PatchCollection as _PC_ov
    fig, ax = plt.subplots(figsize=(9, 9))

    # Shaded original grain polygons
    _patches_ov = []
    _colors_ov = []
    for i, g in enumerate(grains_original):
        _patches_ov.append(_MplPoly_ov(np.asarray(g), closed=True))
        _colors_ov.append(_grain_color(i))
    if _patches_ov:
        _pc_ov = _PC_ov(_patches_ov, facecolors=_colors_ov,
                         edgecolors='k', linewidths=0.4, alpha=_grain_alpha)
        ax.add_collection(_pc_ov)

    # Smoothed grain boundaries overlaid in black
    for i, g in enumerate(valid_gs):
        closed = np.vstack([g, g[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], '-', color='black', linewidth=1.0)
    ax.set_xlim(coords[:,0].min()-0.5, coords[:,0].max()+0.5)
    ax.set_ylim(coords[:,1].min()-0.5, coords[:,1].max()+0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{_title_info}\nOriginal Grains with Smoothed Boundaries', fontsize=12)
    fig.tight_layout()
    save(fig, os.path.join('diagnostics', _next_fig_name('grain_overlay_comparison.png')))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 3: Mesh size function
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n  ---- Mesh size function ----")
    ms = setup_mesh_size_function(ms, verbose=console_on)

    # ── Scale h(x,y) to match target_elements ───────────────────────────
    # Uses a quick Delaunay triangulation (~0.5s) to count elements at each
    # iteration, avoiding the inaccurate integral estimate. Each iteration:
    #   1. Scale h from original unscaled grid
    #   2. Apply floor clamping + junction refinement cap
    #   3. Apply gradient limiting (smooths junction zone transitions)
    #   4. Quick Delaunay count → compare to target → adjust scale
    _target_ne = job["target_elements"]
    _xx_ad, _yy_ad, _hh_ad = ms.MeshSizeFunctionGrid
    _mesh_floor_ratio = job.get("mesh_floor_ratio", 0.25)
    _junction_refine_r = job.get("junction_refine_ratio", 0.7)
    _g_rate = ms.MeshParameters[2]  # gradient rate from advanced_mesh_params

    # Imports for quick Delaunay counting
    from .drectangle import drectangle as _dr_cal
    from .hmatrix import hmatrix as _hm_cal

    _minx_cal = float(ms.MeshSizeFunctionGridLimits[0])
    _maxx_cal = float(ms.MeshSizeFunctionGridLimits[1])
    _miny_cal = float(ms.MeshSizeFunctionGridLimits[2])
    _maxy_cal = float(ms.MeshSizeFunctionGridLimits[3])

    # Helper: quick Delaunay element count for a given h(x,y) grid.
    # Generates a hex grid at h0 spacing, applies density thinning based on
    # h(x,y), triangulates, and counts interior elements.
    def _quick_delaunay_count(hh):
        _h0 = float(np.min(hh))
        _geps = 0.001 * _h0
        _xv = np.arange(_minx_cal, _maxx_cal + _h0 / 2, _h0)
        _yv = np.arange(_miny_cal, _maxy_cal + _h0 * np.sqrt(3) / 4,
                        _h0 * np.sqrt(3) / 2)
        _xg, _yg = np.meshgrid(_xv, _yv)
        _xg[1::2, :] += _h0 / 2
        _p = np.column_stack([_xg.ravel(), _yg.ravel()])
        _p = _p[_dr_cal(_p, _minx_cal, _maxx_cal, _miny_cal, _maxy_cal) < _geps]
        # Density thinning: keep points with probability proportional to 1/h(x,y)^2
        _rng = np.random.RandomState(random_seed)
        _r0 = 1.0 / _hm_cal(_p, _xx_ad, _yy_ad, None, hh) ** 2
        _p = _p[_rng.rand(_p.shape[0]) < _r0 / np.max(_r0)]
        if _p.shape[0] < 3:
            return 0
        _t = Delaunay(_p).simplices
        _pmid = (_p[_t[:, 0]] + _p[_t[:, 1]] + _p[_t[:, 2]]) / 3.0
        _t = _t[_dr_cal(_pmid, _minx_cal, _maxx_cal, _miny_cal, _maxy_cal) < -_geps]
        return _t.shape[0]

    # Initial scale from integral estimate (rough starting point)
    _dx = _xx_ad[0, 1] - _xx_ad[0, 0] if _xx_ad.shape[1] > 1 else 1.0
    _dy = _yy_ad[1, 0] - _yy_ad[0, 0] if _yy_ad.shape[0] > 1 else 1.0
    _N_integral = (2.0 / np.sqrt(3)) * np.sum(1.0 / _hh_ad**2) * _dx * _dy
    _scale = np.sqrt(_N_integral / _target_ne)

    # ── Build junction refinement mask (fixed, computed once) ──────────
    # Grid points within junction_refine_ratio × h_max of any junction
    # point get forced to h_floor (fine elements near stress concentrations).
    # The mask is computed once using the initial h_max estimate and held
    # fixed throughout the loop to avoid positive feedback.
    _junc_mask = np.zeros(_hh_ad.shape, dtype=bool)
    _n_junc_mask = 0
    _refine_radius = 0.0
    if (_junction_refine_r > 0
            and ms.GrainJunctionPoints is not None
            and len(ms.GrainJunctionPoints) > 0):
        _jpts = ms.GrainJunctionPoints
        _h_max_est = float(np.max(_hh_ad * _scale))
        _refine_radius = _junction_refine_r * _h_max_est
        for _jp in _jpts:
            _dist = np.sqrt((_xx_ad - _jp[0])**2 + (_yy_ad - _jp[1])**2)
            _junc_mask |= (_dist < _refine_radius)
        _n_junc_mask = int(np.sum(_junc_mask))

    # ── Iterative scale–clamp–count loop ──────────────────────────────
    _max_iter = 6
    _tolerance = 0.05  # 5% of target
    _n_clamped = 0
    _h_max = 0.0
    _h_floor = 0.0
    for _iter in range(_max_iter):
        # Scale from original unscaled h(x,y) each iteration
        _hh_scaled = _hh_ad * _scale
        _h_max = float(np.max(_hh_scaled))
        _h_floor = max(_mesh_floor_ratio * _h_max, 0.1)

        # Outside junction zones: raise h up to h_floor (prevent tiny elements)
        _clamp_mask = (_hh_scaled < _h_floor) & ~_junc_mask
        _n_clamped = int(np.sum(_clamp_mask))
        _hh_scaled[_clamp_mask] = _h_floor

        # Inside junction zones: cap h down to h_floor (force fine elements)
        if _n_junc_mask > 0:
            _hh_scaled[_junc_mask & (_hh_scaled > _h_floor)] = _h_floor

        # Smooth abrupt transitions at junction zone boundaries
        if _n_junc_mask > 0:
            _hh_scaled = gradient_limiting(_xx_ad, _yy_ad, _hh_scaled, _g_rate)

        # Quick Delaunay count — the actual element count for this h(x,y)
        _N_count = _quick_delaunay_count(_hh_scaled)
        _error = abs(_N_count - _target_ne) / _target_ne

        if console_on:
            print(f"    iter {_iter+1}: N={_N_count}, "
                  f"error={100*(_N_count/_target_ne - 1):+.1f}%, "
                  f"scale={_scale:.4f}")

        if _error < _tolerance:
            break

        # Adjust scale: elements ~ 1/h², so h *= sqrt(N_actual / N_target)
        _scale *= np.sqrt(_N_count / _target_ne)

    ms.MeshSizeFunctionGrid = [_xx_ad, _yy_ad, _hh_scaled]

    # Update l (MeshParameters[3]) and bsep to match the effective minimum
    _l_new = float(np.min(_hh_scaled))
    ms.MeshParameters[3] = _l_new
    job["grain_boundary_resolution"] = _l_new
    bsep = _l_new
    if console_on:
        print(f"  Auto mesh density: target={_target_ne}, actual={_N_count} "
              f"({100*(_N_count/_target_ne - 1):+.1f}%), "
              f"{_iter+1} iterations")
        print(f"    hh range = [{float(np.min(_hh_scaled)):.4f}, {_h_max:.4f}]")
        if _mesh_floor_ratio > 0:
            print(f"    mesh_floor_ratio = {_mesh_floor_ratio} "
                  f"(h_floor = {_h_floor:.4f}, "
                  f"max/min = {1.0/_mesh_floor_ratio:.1f}x, "
                  f"{_n_clamped} pts clamped)")
        if _junction_refine_r > 0 and _n_junc_mask > 0:
            print(f"    junction_refine_ratio = {_junction_refine_r} "
                  f"(radius = {_refine_radius:.4f}, "
                  f"{_n_junc_mask} pts refined)")

    xx_msf, yy_msf, hh_msf = ms.MeshSizeFunctionGrid
    lim_msf = ms.MeshSizeFunctionGridLimits
    print(f"  Grid: {xx_msf.shape}, hh range: [{hh_msf.min():.4f}, {hh_msf.max():.4f}]")
    print(f"  Limits: [{lim_msf[0]:.1f}, {lim_msf[1]:.1f}] x [{lim_msf[2]:.1f}, {lim_msf[3]:.1f}]")
    print(f"  Junction points: {ms.GrainJunctionPoints.shape[0]}")
    
    # ── Figure 04a: Mesh size function contour ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    cf = ax.contourf(xx_msf, yy_msf, hh_msf, levels=30, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Element size h')
    for g in valid_gs:
        closed = np.vstack([g, g[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], 'w-', linewidth=0.5, alpha=0.7)
    ax.set_xlim(lim_msf[0], lim_msf[1]); ax.set_ylim(lim_msf[2], lim_msf[3])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{_title_info}\nMesh Size Function (min={hh_msf.min():.3f}, max={hh_msf.max():.3f})',
                 fontsize=12)
    save(fig, _next_fig_name('mesh_size_function.png'))
    
    # ── Figure 05a: Junction points on grain boundaries ─────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    for i, g in enumerate(valid_gs):
        closed = np.vstack([g, g[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], '-', color=_grain_color(i), linewidth=0.6)
    jpts = ms.GrainJunctionPoints
    ax.plot(jpts[:, 0], jpts[:, 1], 'ro', markersize=5, zorder=5, label=f'{jpts.shape[0]} junction pts')
    ax.set_xlim(lim_msf[0], lim_msf[1]); ax.set_ylim(lim_msf[2], lim_msf[3])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'{_title_info}\nJunction Points ({jpts.shape[0]} points)', fontsize=12)
    save(fig, os.path.join('diagnostics', _next_fig_name('junction_points.png')))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 4: Create initial mesh (fixed + interior points)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n  ---- Create initial mesh ----")
    np.random.seed(random_seed)
    ms = create_initial_mesh(ms)
    
    pfix_pts = ms.InitialMesh[0]
    pinit_pts = ms.InitialMesh[1]
    print(f"  Fixed points: {pfix_pts.shape[0]}")
    print(f"  Interior points: {pinit_pts.shape[0]}")
    
    # ── Figure: Triangulated initial mesh ────────────────────────────────────
    p_all_init = np.vstack([pfix_pts, pinit_pts])
    _, idx_u = np.unique(np.round(p_all_init, 12), axis=0, return_index=True)
    p_all_init = p_all_init[np.sort(idx_u)]
    t_init = Delaunay(p_all_init).simplices
    q_init = simpqual(p_all_init, t_init)
    
    fig, ax = plt.subplots(figsize=(9, 9))
    verts_init = p_all_init[t_init]
    pc_init = PolyCollection(verts_init, facecolors='white', edgecolors=(0, 0, 0, 0.5), linewidths=0.3)
    ax.add_collection(pc_init)
    for g in valid_gs:
        closed = np.vstack([g, g[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], 'k-', linewidth=0.5)
    ax.set_xlim(lim_msf[0], lim_msf[1]); ax.set_ylim(lim_msf[2], lim_msf[3])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{_title_info}\nInitial Mesh — {t_init.shape[0]} elements, {p_all_init.shape[0]} nodes',
                 fontsize=12)
    save(fig, os.path.join('diagnostics', _next_fig_name('initial_mesh_triangulated.png')))
    
    print(f"\n  Pipeline ready: {len(ms.GrainsSmoothed)} grains, "
          f"pfix={pfix_pts.shape[0]}, pinit={pinit_pts.shape[0]}")
    print(f"  Initial mesh: {t_init.shape[0]} elems, q_min={q_init.min():.4f}, q_mean={q_init.mean():.4f}")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Replicate mesh_all_grains step by step with figures
    # ═══════════════════════════════════════════════════════════════════════════
    
    Gs = ms.GrainsSmoothed
    xx, yy, hh = ms.MeshSizeFunctionGrid
    lim = ms.MeshSizeFunctionGridLimits
    minx, maxx, miny, maxy = lim
    h0 = np.min(hh)
    geps = 0.01 * h0
    eps_val = np.sqrt(np.finfo(float).eps)
    num_grains = len(Gs)
    
    K, R, g, l = ms.MeshParameters
    print("\n" + "=" * 60)
    print("MESH PARAMETERS SUMMARY")
    print("=" * 60)
    print(f"  MeshParameters: K={K}, R={R}, g={g}, l={l:.6f}")
    print(f"  h0 (min element size) = min(hh) = {h0:.6f}")
    print(f"  hh range: [{np.min(hh):.4f}, {np.max(hh):.4f}]")
    print(f"  geps (centroid tolerance) = 0.01 * h0 = {geps:.6f}")
    # All thresholds are purely relative to h0 (which equals l in pixel coordinates).
    # No absolute floors: since 1 unit = 1 pixel = 1 scan step for all .ang files,
    # h0-relative values scale correctly with mesh density.
    # Exception: recover_threshold = 1.0 pixel (absolute) — tied to EBSD polyline
    # tracing gap size, which is ~1 scan step regardless of mesh density.
    print(f"  thresh (node selection)    = h0      = {h0:.6f}")
    print(f"  gsep  (boundary snap dist) = h0/2    = {h0/2:.6f}")
    print(f"  snap_tol (close-node merge)= h0/5    = {h0/5:.6f}")
    print(f"  recover_threshold (orphan) = 1.0 px  (absolute)")
    print(f"  Domain: x=[{minx:.1f}, {maxx:.1f}], y=[{miny:.1f}, {maxy:.1f}]")
    print(f"  Domain area: {(maxx-minx)*(maxy-miny):.1f}")
    print(f"  Number of grains: {num_grains}")
    print(f"  Grid size: {xx.shape}")
    print(f"  Junction points: {ms.GrainJunctionPoints.shape[0]}")
    print(f"  EBSD file: {ebsd_path}")
    print("=" * 60 + "\n")
    
    pfix_all = np.unique(np.vstack([ms.InitialMesh[0], ms.GrainJunctionPoints]), axis=0)
    pfix_set = set(map(tuple, np.round(pfix_all, 12)))
    pinit = ms.InitialMesh[1][
        np.array([tuple(np.round(row, 12)) not in pfix_set for row in ms.InitialMesh[1]])]
    
    # Holes
    if ms.GrainHoles is None or len(ms.GrainHoles) == 0:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            Holes, holes = find_grain_holes(Gs)
        ms.GrainHoles = Holes
        ms.HoleGrains = holes
    else:
        Holes = ms.GrainHoles
    
    # Decimate grains
    Grains = [None] * num_grains
    for i in range(num_grains):
        closed = np.vstack([Gs[i], Gs[i][0:1, :]])
        Grains[i], _ = DecimatePoly(closed, np.array([eps_val, 1.0]))
        Grains[i] = Grains[i][:-1, :]
    
    # ── Step A: distmesh2d0 (single pass) ────────────────────────────────────
    from scipy.spatial.distance import cdist
    
    fh_info = [xx, yy, hh]
    corners = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])
    jbpts = np.unique(ms.GrainJunctionPoints, axis=0)
    jbpts_set = set(map(tuple, np.round(jbpts, 12)))
    
    print("\nStep A: distmesh2d0...")
    t0 = time.time()
    
    # ── Iteration PNGs → compiled PDF ─────────────────────────────────────────
    from matplotlib.collections import PolyCollection as _PC
    _iter_png_dir = os.path.join(FIG_DIR, 'iterations')
    # Clear previous run's PNGs so stale files don't persist
    if os.path.isdir(_iter_png_dir):
        for _old_f in os.listdir(_iter_png_dir):
            if _old_f.endswith('.png'):
                os.remove(os.path.join(_iter_png_dir, _old_f))
    os.makedirs(_iter_png_dir, exist_ok=True)
    _iter_png_files = []
    
    _cmap_iter = plt.cm.get_cmap(_grain_cmap_name, 20)
    
    # Pre-compute per-grain signed distance functions for accurate grain assignment
    # For grains with holes (enclosed grains), use ddiff_multi so that fd is positive
    # inside the hole — the enclosed grain then wins the "most negative fd" contest.
    from tesa.ddiff_multi import ddiff_multi as _ddiff_multi_iter
    _grain_fd = []
    for _gi in range(num_grains):
        if Holes[_gi] is not None and len(Holes[_gi]) > 0:
            _grain_fd.append(
                (lambda idx, hi: lambda pts: _ddiff_multi_iter(pts, Gs, idx, hi))(_gi, Holes[_gi]))
        else:
            _gc = np.vstack([Gs[_gi], Gs[_gi][0:1, :]])
            _grain_fd.append((lambda gc: lambda pts: dpoly(pts, gc))(_gc))
    
    def _assign_grains_fd(pmid):
        """Assign triangle centroids to grain using signed distance (fd < 0 = inside)."""
        n = pmid.shape[0]
        g_assign = np.full(n, -1, dtype=int)
        best_fd = np.full(n, np.inf)
        for gi in range(num_grains):
            fd_vals = _grain_fd[gi](pmid)
            # A point is "inside" grain gi if fd < 0; use most-negative fd as assignment
            better = fd_vals < best_fd
            g_assign[better] = gi
            best_fd[better] = fd_vals[better]
        # Any unassigned points (fd > 0 for all grains) → assign to nearest grain centroid
        unassigned = g_assign < 0
        if np.any(unassigned):
            _centroids = np.array([Gs[gi].mean(axis=0) for gi in range(num_grains)])
            D = cdist(pmid[unassigned], _centroids)
            g_assign[unassigned] = np.argmin(D, axis=1)
        return g_assign
    
    _temp_iter_png = os.path.join(_iter_png_dir, '_display.png')

    # Extract convergence parameters from job dictionary (needed by callback closure)
    mesh_conv = job.get("mesh_convergence", [40, 0.2, 0.90, 100])
    min_iter             = int(mesh_conv[0])
    q_worst_avg_target   = float(mesh_conv[1])
    q_mean_target        = float(mesh_conv[2])
    max_iter             = int(mesh_conv[3])

    # Mutable container to stash the last callback's (pts, tris) for final mesh
    _last_iter_mesh = {'p': None, 't': None}

    # Convergence history for live plotting during iterations
    _live_conv = {'iters': [], 'q_mins': [], 'q_worst_avgs': [], 'q_means': []}

    def _iter_png_callback(iteration, pts, tris, q_min, q_mean, q_worst_avg):
        """Save PNG for every iteration, display mesh + convergence side-by-side on schedule."""
        # Use quality values already computed by distmesh2d0 (no duplicate computation)
        _q_min_global = float(q_min)
        _q_mean_global = float(q_mean)
        _q_worst_avg_global = float(q_worst_avg)

        # Always stash the latest mesh
        _last_iter_mesh['p'] = pts.copy()
        _last_iter_mesh['t'] = tris.copy()
        _last_iter_mesh['iter'] = iteration
        _last_iter_mesh['q_min'] = _q_min_global
        _last_iter_mesh['q_mean'] = _q_mean_global
        _last_iter_mesh['q_worst_avg'] = _q_worst_avg_global
        on_schedule = (iteration <= 5 or iteration % 5 == 0)
        _last_iter_mesh['rendered'] = on_schedule

        # Update live convergence history
        _live_conv['iters'].append(iteration)
        _live_conv['q_mins'].append(_q_min_global)
        _live_conv['q_worst_avgs'].append(_q_worst_avg_global)
        _live_conv['q_means'].append(_q_mean_global)

        # Render and save mesh PNG for every iteration
        fig_i, ax_i = plt.subplots(figsize=(9, 9))
        pmid_i = (pts[tris[:, 0]] + pts[tris[:, 1]] + pts[tris[:, 2]]) / 3.0
        g_assign = _assign_grains_fd(pmid_i)
        facecolors_i = [_cmap_iter(_grain_color_idx.get(g, g % _N_CYCLE)) for g in g_assign]
        verts_i = pts[tris]
        pc_i = _PC(verts_i, facecolors=facecolors_i, edgecolors=(0, 0, 0, 0.5),
                    linewidths=0.3, alpha=_grain_alpha)
        ax_i.add_collection(pc_i)
        ax_i.set_xlim(minx, maxx); ax_i.set_ylim(miny, maxy)
        ax_i.set_aspect('equal')
        ax_i.set_xticks([]); ax_i.set_yticks([])
        ax_i.set_title(f'Iteration {iteration} [{ebsd_name}]\n'
                        f'{tris.shape[0]} elements, {pts.shape[0]} nodes | '
                        f'q_min={_q_min_global:.4f}, q_worst_avg={_q_worst_avg_global:.4f}, '
                        f'q_mean={_q_mean_global:.4f}', fontsize=12)
        png_path = os.path.join(_iter_png_dir, f'iter_{iteration:04d}.png')
        fig_i.savefig(png_path, dpi=figure_dpi, bbox_inches='tight')
        _iter_png_files.append(png_path)
        plt.close(fig_i)

        # Display side-by-side: mesh (left) + convergence (right) on schedule
        if on_schedule and show_figures:
            fig_display, (ax_mesh, ax_conv) = plt.subplots(1, 2, figsize=(18, 8),
                                                            gridspec_kw={'width_ratios': [1.2, 1]})

            # Left panel: mesh at this iteration
            pmid_d = (pts[tris[:, 0]] + pts[tris[:, 1]] + pts[tris[:, 2]]) / 3.0
            g_assign_d = _assign_grains_fd(pmid_d)
            facecolors_d = [_cmap_iter(_grain_color_idx.get(g, g % _N_CYCLE)) for g in g_assign_d]
            pc_d = _PC(pts[tris], facecolors=facecolors_d, edgecolors=(0, 0, 0, 0.5), linewidths=0.3, alpha=_grain_alpha)
            ax_mesh.add_collection(pc_d)
            ax_mesh.set_xlim(minx, maxx); ax_mesh.set_ylim(miny, maxy)
            ax_mesh.set_aspect('equal')
            ax_mesh.set_xticks([])
            ax_mesh.set_yticks([])
            ax_mesh.set_title(f'Iteration {iteration} [{ebsd_name}]\n'
                              f'{tris.shape[0]} elements, {pts.shape[0]} nodes | '
                              f'q_min={_q_min_global:.4f}, q_worst_avg={_q_worst_avg_global:.4f}, '
                              f'q_mean={_q_mean_global:.4f}', fontsize=11)

            # Right panel: convergence history up to this iteration
            from matplotlib.ticker import MaxNLocator as _MaxNLocator_iter
            _iters = _live_conv['iters']
            _qmins = _live_conv['q_mins']
            _qworstavgs = _live_conv['q_worst_avgs']
            _qmeans = _live_conv['q_means']
            ax_conv.plot(_iters, _qmins, 'o-', color='tab:blue', markersize=4, label='q_min')
            ax_conv.plot(_iters, _qworstavgs, 'D-', color='tab:green', markersize=4, label='q_worst_avg (0.5%)')
            ax_conv.plot(_iters, _qmeans, 's-', color='tab:red', markersize=4, label='q_mean')
            ax_conv.axhline(y=q_worst_avg_target, color='tab:green', linestyle='--', linewidth=0.8,
                            alpha=0.5, label=f'q_worst_avg target = {q_worst_avg_target}')
            ax_conv.axhline(y=q_mean_target, color='tab:red', linestyle='--', linewidth=0.8,
                            alpha=0.5, label=f'q_mean target = {q_mean_target}')
            ax_conv.set_ylim(0.0, 1.0)
            ax_conv.xaxis.set_major_locator(_MaxNLocator_iter(integer=True))
            ax_conv.set_xlabel('Iteration', fontsize=11)
            ax_conv.set_ylabel('Quality', fontsize=11)
            ax_conv.set_title('Convergence History', fontsize=11)
            ax_conv.legend(fontsize=9, loc='lower left')
            ax_conv.grid(True, alpha=0.3)

            fig_display.tight_layout()
            fig_display.show()
            plt.pause(figure_pause)
            plt.close(fig_display)
    
    # NOTE: No per-iteration polyline snapping — MATLAB uses standard DistMesh
    # gradient projection (fd > 0 → project back) inside the per-grain inner loop.
    # Snapping to FineGrainPolylines happens once AFTER distmesh2d0 returns.

    p, t_distmesh, convergence_history = distmesh2d0(ms, fh_info, pfix_all, pinit, Grains, Holes,
                        h0, minx, maxx, miny, maxy,
                        max_iter=max_iter, min_iter=min_iter,
                        q_worst_avg_target=q_worst_avg_target, q_mean_target=q_mean_target,
                        iter_callback=_iter_png_callback,
                        mesh_internals=_MESH_INTERNALS)
    print(f"  Done ({time.time()-t0:.1f}s), {p.shape[0]} nodes, {t_distmesh.shape[0]} elements")
    print(f"  Saved {len(_iter_png_files)} iteration PNGs to {_iter_png_dir}")
    
    # Always render the final iteration as side-by-side (mesh + convergence)
    if _last_iter_mesh['p'] is not None:
        _fi = _last_iter_mesh['iter']
        _fp = _last_iter_mesh['p']
        _ft = _last_iter_mesh['t']
        _fqmin = _last_iter_mesh['q_min']
        _fqmean = _last_iter_mesh['q_mean']
        _fqworstavg = _last_iter_mesh.get('q_worst_avg', _fqmin)

        fig_f, (ax_mesh_f, ax_conv_f) = plt.subplots(1, 2, figsize=(18, 8),
                                                       gridspec_kw={'width_ratios': [1.2, 1]})

        # Left panel: mesh at final iteration
        _fmid = (_fp[_ft[:,0]] + _fp[_ft[:,1]] + _fp[_ft[:,2]]) / 3.0
        _fg = _assign_grains_fd(_fmid)
        _ffc = [_cmap_iter(_grain_color_idx.get(g, g % _N_CYCLE)) for g in _fg]
        pc_f = _PC(_fp[_ft], facecolors=_ffc, edgecolors=(0, 0, 0, 0.5), linewidths=0.3, alpha=_grain_alpha)
        ax_mesh_f.add_collection(pc_f)
        ax_mesh_f.set_xlim(minx, maxx); ax_mesh_f.set_ylim(miny, maxy)
        ax_mesh_f.set_aspect('equal')
        ax_mesh_f.set_xticks([])
        ax_mesh_f.set_yticks([])
        ax_mesh_f.set_title(f'Final Iteration {_fi} [{ebsd_name}]\n'
                            f'{_ft.shape[0]} elements, {_fp.shape[0]} nodes  |  '
                            f'q_min={_fqmin:.4f}, q_worst_avg={_fqworstavg:.4f}, '
                            f'q_mean={_fqmean:.4f}', fontsize=11)

        # Right panel: full convergence history
        from matplotlib.ticker import MaxNLocator as _MaxNLocator
        _iters_f = _live_conv['iters']
        _qmins_f = _live_conv['q_mins']
        _qworstavgs_f = _live_conv['q_worst_avgs']
        _qmeans_f = _live_conv['q_means']
        ax_conv_f.plot(_iters_f, _qmins_f, 'o-', color='tab:blue', markersize=4, label='q_min')
        ax_conv_f.plot(_iters_f, _qworstavgs_f, 'D-', color='tab:green', markersize=4, label='q_worst_avg (0.5%)')
        ax_conv_f.plot(_iters_f, _qmeans_f, 's-', color='tab:red', markersize=4, label='q_mean')
        ax_conv_f.axhline(y=q_worst_avg_target, color='tab:green', linestyle='--', linewidth=0.8,
                          alpha=0.5, label=f'q_worst_avg target = {q_worst_avg_target}')
        ax_conv_f.axhline(y=q_mean_target, color='tab:red', linestyle='--', linewidth=0.8,
                          alpha=0.5, label=f'q_mean target = {q_mean_target}')
        ax_conv_f.set_ylim(0.0, 1.0)
        ax_conv_f.xaxis.set_major_locator(_MaxNLocator(integer=True))
        ax_conv_f.set_xlabel('Iteration', fontsize=11)
        ax_conv_f.set_ylabel('Quality', fontsize=11)
        ax_conv_f.set_title('Convergence History', fontsize=11)
        ax_conv_f.legend(fontsize=9, loc='lower left')
        ax_conv_f.grid(True, alpha=0.3)

        fig_f.tight_layout()

    # ── Compile iteration PNGs into GIF ───────────────────────────────────────
    if _iter_png_files:
        import imageio.v3 as _iio
        _gif_name = _next_fig_name('mesh_iterations.gif')
        _gif_path = os.path.join(FIG_DIR, _gif_name)
        _frames_raw = [_iio.imread(f) for f in sorted(_iter_png_files)]
        # Resize all frames to the same dimensions (largest height × largest width)
        # to avoid ValueError when stacking frames with different bbox_inches='tight' crops
        _max_h = max(f.shape[0] for f in _frames_raw)
        _max_w = max(f.shape[1] for f in _frames_raw)
        _frames = []
        for _fr in _frames_raw:
            if _fr.shape[0] == _max_h and _fr.shape[1] == _max_w:
                _frames.append(_fr)
            else:
                # Pad with white to match the largest frame size
                _padded = np.full((_max_h, _max_w) + _fr.shape[2:], 255, dtype=_fr.dtype)
                _padded[:_fr.shape[0], :_fr.shape[1]] = _fr
                _frames.append(_padded)
        _iio.imwrite(_gif_path, _frames, duration=333, loop=1)
        print(f"  Iteration animation saved: {_gif_name} "
              f"({len(_frames)} frames, 3 fps, {os.path.getsize(_gif_path)/1024:.0f} KB)")

    # ── Save final iteration figure (after GIF so numbering is sequential) ────
    if _last_iter_mesh['p'] is not None:
        save(fig_f, os.path.join('diagnostics', _next_fig_name('final_iteration.png')))
        print(f"  Final iteration {_fi}")
    
    # ── Convergence history plot ────────────────────────────────────────────
    if convergence_history['iterations']:
        fig_conv, ax_conv_hist = plt.subplots(figsize=(8, 5))
        iters = convergence_history['iterations']
        q_mins = convergence_history['q_min']
        q_worst_avgs = convergence_history['q_worst_avg']
        q_means = convergence_history['q_mean']

        # All three quality measures on a single panel
        ax_conv_hist.plot(iters, q_means, 's-', color='tab:red', markersize=4, label='q_mean')
        ax_conv_hist.plot(iters, q_worst_avgs, 'D-', color='tab:green', markersize=4, label='q_worst_avg (0.5%)')
        ax_conv_hist.plot(iters, q_mins, 'o-', color='tab:blue', markersize=4, label='q_min')
        ax_conv_hist.axhline(y=q_mean_target, color='tab:red', linestyle='--', linewidth=0.8,
                             alpha=0.5, label=f'q_mean target = {q_mean_target}')
        ax_conv_hist.axhline(y=q_worst_avg_target, color='tab:green', linestyle='--', linewidth=0.8,
                             alpha=0.5, label=f'q_worst_avg target = {q_worst_avg_target}')
        ax_conv_hist.set_ylabel('Quality', fontsize=11)
        ax_conv_hist.set_ylim(0.0, 1.0)
        ax_conv_hist.set_xlabel('Iteration', fontsize=11)
        ax_conv_hist.set_title(f'{ebsd_name} — Mesh Quality Convergence', fontsize=12)
        ax_conv_hist.legend(fontsize=9, loc='center right')
        ax_conv_hist.grid(True, alpha=0.3)

        fig_conv.tight_layout()
        save(fig_conv, _next_fig_name('mesh_quality_convergence.png'))

    p_after_iter = p.copy()   # snapshot of nodes after distmesh2d0 iterations
    
    # Use the stashed callback mesh — this is the EXACT (pts, tris) from the last
    # iteration that was displayed, matching iter_NNNN.png pixel-for-pixel.
    # t_distmesh is a DIFFERENT Delaunay (rebuilt on a corner-augmented node set
    # at termination), so it won't match the iteration plots.
    if _last_iter_mesh['p'] is not None:
        p_after_iter = _last_iter_mesh['p']
        t_after_iter = _last_iter_mesh['t']
        print(f"  Using callback mesh: {p_after_iter.shape[0]} nodes, "
              f"{t_after_iter.shape[0]} elements (matches last iteration plot)")
    else:
        t_after_iter = t_distmesh.copy()
        print(f"  WARNING: no callback mesh available, using t_distmesh")
    
    # ── Grain assignment (fd-based) ──────────────────────────────────────────
    # Uses the clean post-iteration global Delaunay directly.  Each element is
    # assigned to the grain with the most negative fd at its centroid —
    # the same method used by the iteration callback (_assign_grains_fd).
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Grain assignment (fd-based) ──")
    
    _mb_p = p_after_iter.copy()
    _mb_t = t_after_iter.copy()  # exact triangulation from distmesh2d0 (matches iteration plot)
    
    # Fix orientation (CCW) — distmesh2d0 doesn't guarantee winding order
    _mb_sgn = triarea(_mb_p, _mb_t)
    _mb_sw = _mb_sgn < 0
    _mb_t[_mb_sw, 0], _mb_t[_mb_sw, 1] = _mb_t[_mb_sw, 1].copy(), _mb_t[_mb_sw, 0].copy()
    
    # ── Cleanup: merge coincident nodes, remove degenerates, compact ──
    from .cleanup_mesh import cleanup_mesh
    print("  Cleanup:")
    _mb_p, _mb_t, _cleanup_stats = cleanup_mesh(_mb_p, _mb_t, verbose=True)
    
    # ── Grain assignment (after cleanup) ─────────────────────────────────────
    _mb_n_grains = len(ms.Grains)
    _mb_centroids = (_mb_p[_mb_t[:,0]] + _mb_p[_mb_t[:,1]] + _mb_p[_mb_t[:,2]]) / 3.0
    
    # Assign each element to the grain with the most negative fd at its centroid
    # (reuses _assign_grains_fd already defined for the iteration callback)
    _mb_elem_grain = _assign_grains_fd(_mb_centroids)
    
    _mb_n_elems = _mb_t.shape[0]
    _mb_n_grain_ok = int(np.sum(_mb_elem_grain >= 0))
    print(f"  Element grain assignment: {_mb_n_grain_ok}/{_mb_n_elems} assigned")
    
    # Build GrainsElements-style mapping
    _mb_grain_elsets = {}
    for _gi in range(_mb_n_grains):
        _gi_mask = np.where(_mb_elem_grain == _gi)[0]
        if len(_gi_mask) > 0:
            _mb_grain_elsets[_gi] = _gi_mask
    
    # ── Write mesh_final.txt with full integrity checks ──────────────────
    _mb_path = os.path.join(FIG_DIR, 'diagnostics', 'mesh_final.txt')
    print(f"\nWriting mesh_final.txt ...")
    
    # 1-indexed arrays
    _mb_p1  = _mb_p
    _mb_t1  = _mb_t + 1
    _mb_nn  = _mb_p1.shape[0]
    _mb_ne  = _mb_t1.shape[0]
    
    # ── Integrity checks ──────────────────────────────────────────────────
    # --- Duplicate node IDs (always OK with sequential numbering)
    _mb_dup_node_ids = False
    
    # --- Duplicate node coordinates
    _mb_coord_round = np.round(_mb_p1, 12)
    _mb_coord_tuples = [tuple(row) for row in _mb_coord_round]
    _mb_seen_coords = {}
    _mb_dup_coords = []
    for _ni, _ct in enumerate(_mb_coord_tuples, start=1):
        if _ct in _mb_seen_coords:
            _mb_dup_coords.append((_ct, [_mb_seen_coords[_ct], _ni]))
        else:
            _mb_seen_coords[_ct] = _ni
    
    # --- Duplicate element IDs (always OK)
    _mb_dup_elem_ids = False
    
    # --- Duplicate element connectivity
    _mb_conn_keys = [tuple(sorted(row)) for row in _mb_t1]
    _mb_conn_seen = {}
    _mb_dup_conn = []
    for _ei, _ck in enumerate(_mb_conn_keys, start=1):
        if _ck in _mb_conn_seen:
            _mb_dup_conn.append((_ck, [_mb_conn_seen[_ck], _ei]))
        else:
            _mb_conn_seen[_ck] = _ei
    
    # --- References to non-existent nodes
    _mb_all_refs = set(_mb_t1.ravel())
    _mb_bad_refs = _mb_all_refs - set(range(1, _mb_nn + 1))
    
    # --- Degenerate elements
    _mb_degen = [_ei for _ei, row in enumerate(_mb_t1, start=1)
                 if len(set(row)) < 3]
    
    # --- Zero-area elements
    _mb_areas = triarea(_mb_p1, _mb_t)
    _mb_abs_areas = np.abs(_mb_areas)
    _mb_n_zero = int(np.sum(_mb_abs_areas < 1e-14))
    
    # --- Consistent CCW winding
    _mb_n_cw = int(np.sum(_mb_areas < -1e-14))
    
    # --- All elements in a grain elset
    _mb_missing_from_sets = [_ei + 1 for _ei in range(_mb_ne)
                             if _mb_elem_grain[_ei] < 0]
    
    # --- Each element in exactly one grain (always 1 with nearest-point)
    _mb_multi_grain_elems = []
    
    # --- All nodes referenced
    _mb_ref_nodes = set(_mb_t1.ravel())
    _mb_unreferenced = set(range(1, _mb_nn + 1)) - _mb_ref_nodes
    
    # --- Crossing edges (grid-accelerated)
    _mb_cell_sz = max(h0 * 5.0, 1.0)
    _mb_gcx = int(np.ceil((maxx - minx) / _mb_cell_sz)) + 1
    _mb_gcy = int(np.ceil((maxy - miny) / _mb_cell_sz)) + 1
    
    def _mb_seg_cross(p0, p1, p2, p3):
        d1x, d1y = p1[0]-p0[0], p1[1]-p0[1]
        d2x, d2y = p3[0]-p2[0], p3[1]-p2[1]
        denom = d1x*d2y - d1y*d2x
        if abs(denom) < 1e-15:
            return False
        t = ((p2[0]-p0[0])*d2y - (p2[1]-p0[1])*d2x) / denom
        u = ((p2[0]-p0[0])*d1y - (p2[1]-p0[1])*d1x) / denom
        return 1e-10 < t < 1-1e-10 and 1e-10 < u < 1-1e-10
    
    _mb_edge_cells = _ddict(list)
    _mb_all_edges_list = []
    for _ei in range(_mb_ne):
        n0, n1, n2 = int(_mb_t1[_ei,0]), int(_mb_t1[_ei,1]), int(_mb_t1[_ei,2])
        for _a, _b in ((n0,n1),(n1,n2),(n0,n2)):
            _ek = (min(_a,_b), max(_a,_b))
            if _ek not in _mb_edge_cells:
                _mb_all_edges_list.append(_ek)
            ax_ = _mb_p1[_a-1, 0]; ay_ = _mb_p1[_a-1, 1]
            bx_ = _mb_p1[_b-1, 0]; by_ = _mb_p1[_b-1, 1]
            cx0 = int((min(ax_, bx_) - minx) / _mb_cell_sz)
            cy0 = int((min(ay_, by_) - miny) / _mb_cell_sz)
            cx1 = int((max(ax_, bx_) - minx) / _mb_cell_sz)
            cy1 = int((max(ay_, by_) - miny) / _mb_cell_sz)
            for _cx in range(max(0,cx0), min(_mb_gcx, cx1+1)):
                for _cy in range(max(0,cy0), min(_mb_gcy, cy1+1)):
                    _mb_edge_cells[(_cx,_cy)].append(_ek)
    
    _mb_crossing_pairs = []
    _mb_checked = set()
    for _cell_key, _eklist in _mb_edge_cells.items():
        for _ii in range(len(_eklist)):
            for _jj in range(_ii+1, len(_eklist)):
                _e1, _e2 = _eklist[_ii], _eklist[_jj]
                if _e1 == _e2:
                    continue
                _pair = (_e1, _e2) if _e1 < _e2 else (_e2, _e1)
                if _pair in _mb_checked:
                    continue
                _mb_checked.add(_pair)
                if set(_e1) & set(_e2):
                    continue
                if _mb_seg_cross(_mb_p1[_e1[0]-1], _mb_p1[_e1[1]-1],
                                 _mb_p1[_e2[0]-1], _mb_p1[_e2[1]-1]):
                    _mb_crossing_pairs.append((
                        (int(_e1[0]),int(_e1[1])), (int(_e2[0]),int(_e2[1]))))
                    if len(_mb_crossing_pairs) >= 5:
                        break
            if len(_mb_crossing_pairs) >= 5:
                break
        if len(_mb_crossing_pairs) >= 5:
            break
    
    # --- T-junctions
    _mb_edge_elems = _ddict(list)
    for _ei in range(_mb_ne):
        n0, n1, n2 = int(_mb_t1[_ei,0]), int(_mb_t1[_ei,1]), int(_mb_t1[_ei,2])
        for _a, _b in ((n0,n1),(n1,n2),(n0,n2)):
            _mb_edge_elems[(min(_a,_b), max(_a,_b))].append(_ei + 1)
    _mb_tjunctions = [(ek, els) for ek, els in _mb_edge_elems.items() if len(els) > 2]
    
    # --- Bowtie (non-manifold) nodes
    _mb_node_elems_map = _ddict(set)
    for _ei in range(_mb_ne):
        for _j in range(3):
            _mb_node_elems_map[int(_mb_t1[_ei,_j])].add(_ei)
    
    _mb_bowtie_nodes = []
    for _nid, _nelems in _mb_node_elems_map.items():
        if len(_nelems) < 2:
            continue
        _adj = _ddict(set)
        for _ei in _nelems:
            _ns = [int(_mb_t1[_ei,_j]) for _j in range(3) if int(_mb_t1[_ei,_j]) != _nid]
            if len(_ns) == 2:
                _adj[_ei].add(_ns[0]); _adj[_ei].add(_ns[1])
        _visited = set(); _comps = []
        for _start in _nelems:
            if _start in _visited:
                continue
            _comp = set(); _queue = [_start]
            while _queue:
                _cur = _queue.pop()
                if _cur in _comp:
                    continue
                _comp.add(_cur)
                for _nb in _nelems:
                    if _nb not in _comp and _adj[_cur] & _adj[_nb]:
                        _queue.append(_nb)
            _comps.append(_comp); _visited |= _comp
        if len(_comps) > 1:
            _mb_bowtie_nodes.append(_nid)
    
    # --- Single connected component
    _mb_elem_adj = _ddict(set)
    for _ek, _els in _mb_edge_elems.items():
        for _ii in range(len(_els)):
            for _jj in range(_ii+1, len(_els)):
                _mb_elem_adj[_els[_ii]].add(_els[_jj])
                _mb_elem_adj[_els[_jj]].add(_els[_ii])
    
    _mb_visited_elems = set()
    _mb_queue = [1]
    while _mb_queue:
        _cur = _mb_queue.pop()
        if _cur in _mb_visited_elems:
            continue
        _mb_visited_elems.add(_cur)
        for _nb in _mb_elem_adj[_cur]:
            if _nb not in _mb_visited_elems:
                _mb_queue.append(_nb)
    _mb_disconnected_elems = sorted(set(range(1, _mb_ne+1)) - _mb_visited_elems)
    
    # --- Domain coverage (area sum)
    _mb_area_sum = float(np.sum(_mb_abs_areas))
    _mb_domain_area = (maxx - minx) * (maxy - miny)
    _mb_area_tol = 1e-6 * _mb_domain_area
    _mb_area_gap = _mb_domain_area - _mb_area_sum
    _mb_area_ok = abs(_mb_area_gap) < _mb_area_tol
    
    # --- Interior boundary edges
    _mb_eps_dom = 1e-6
    def _mb_on_dom_bnd(na, nb):
        _xa, _ya = _mb_p1[na-1]; _xb, _yb = _mb_p1[nb-1]
        return (
            (abs(_xa-minx)<_mb_eps_dom and abs(_xb-minx)<_mb_eps_dom) or
            (abs(_xa-maxx)<_mb_eps_dom and abs(_xb-maxx)<_mb_eps_dom) or
            (abs(_ya-miny)<_mb_eps_dom and abs(_yb-miny)<_mb_eps_dom) or
            (abs(_ya-maxy)<_mb_eps_dom and abs(_yb-maxy)<_mb_eps_dom))
    _mb_interior_bnd_edges = [(ek, el[0]) for ek, el in _mb_edge_elems.items()
                              if len(el)==1 and not _mb_on_dom_bnd(ek[0], ek[1])]
    
    # --- Boundary loop count
    _mb_bnd_edges_all = [(ek, el[0]) for ek, el in _mb_edge_elems.items() if len(el)==1]
    _mb_bnd_adj = _ddict(list)
    for (_na, _nb), _ in _mb_bnd_edges_all:
        _mb_bnd_adj[_na].append(_nb)
        _mb_bnd_adj[_nb].append(_na)
    _mb_bnd_remaining = set(_mb_bnd_adj.keys())
    _mb_bnd_loops = []
    while _mb_bnd_remaining:
        _start = min(_mb_bnd_remaining)
        _loop_nodes = set(); _bfsq = [_start]
        while _bfsq:
            _cur = _bfsq.pop()
            if _cur in _loop_nodes:
                continue
            _loop_nodes.add(_cur)
            for _nb in _mb_bnd_adj[_cur]:
                if _nb not in _loop_nodes:
                    _bfsq.append(_nb)
        _mb_bnd_loops.append(_loop_nodes)
        _mb_bnd_remaining -= _loop_nodes
    _mb_n_bnd_loops = len(_mb_bnd_loops)
    _mb_n_interior_holes = max(0, _mb_n_bnd_loops - 1)
    _mb_loops_ok = (_mb_n_bnd_loops == 1)
    
    # --- Advisory: long edges
    _mb_unique_edges = list(_mb_edge_elems.keys())
    _mb_edge_lens = np.array([np.linalg.norm(_mb_p1[a-1] - _mb_p1[b-1])
                               for a, b in _mb_unique_edges])
    _mb_mean_elen = float(np.mean(_mb_edge_lens))
    _mb_long_thresh = 4 * _mb_mean_elen
    _mb_long_edges = [(int(a),int(b), float(le))
                       for (a,b), le in zip(_mb_unique_edges, _mb_edge_lens)
                       if le > _mb_long_thresh]
    
    # --- Advisory: grain boundary edge alignment
    _mb_elem_grain_lut = {}
    for _ei in range(_mb_ne):
        _mb_elem_grain_lut[_ei + 1] = int(_mb_elem_grain[_ei])
    
    _mb_triple_junc_nodes = set()
    for _nid, _nelems in _mb_node_elems_map.items():
        _grains_at_node = set(_mb_elem_grain_lut[_ei+1] for _ei in _nelems
                              if _mb_elem_grain_lut.get(_ei+1, -1) >= 0)
        if len(_grains_at_node) >= 3:
            _mb_triple_junc_nodes.add(_nid)
    
    _mb_gb_node_only = []
    for _nid, _nelems in _mb_node_elems_map.items():
        if _nid in _mb_triple_junc_nodes:
            continue
        _grains_at = set(_mb_elem_grain_lut[_ei+1] for _ei in _nelems
                         if _mb_elem_grain_lut.get(_ei+1, -1) >= 0)
        if len(_grains_at) < 2:
            continue
        for _ga_v in _grains_at:
            for _gb_v in _grains_at:
                if _ga_v >= _gb_v:
                    continue
                _elA = [_ei+1 for _ei in _nelems if _mb_elem_grain_lut.get(_ei+1,-1)==_ga_v]
                _elB = [_ei+1 for _ei in _nelems if _mb_elem_grain_lut.get(_ei+1,-1)==_gb_v]
                _has_shared_edge = False
                for _ea in _elA:
                    _nsA = set(int(_mb_t1[_ea-1,j]) for j in range(3))
                    for _eb in _elB:
                        _nsB = set(int(_mb_t1[_eb-1,j]) for j in range(3))
                        if len(_nsA & _nsB) >= 2:
                            _has_shared_edge = True
                            break
                    if _has_shared_edge:
                        break
                if not _has_shared_edge:
                    _mb_gb_node_only.append((_nid, _elA[0], _ga_v, _elB[0], _gb_v))
    
    # --- Grain boundary edge count and statistics
    _mb_n_gb_edges = 0
    _mb_gb_edge_lengths = []
    _mb_gb_node_set = set()
    for _ek, _els in _mb_edge_elems.items():
        if len(_els) == 2:
            _g0 = _mb_elem_grain_lut.get(_els[0], -1)
            _g1 = _mb_elem_grain_lut.get(_els[1], -1)
            if _g0 != _g1 and _g0 >= 0 and _g1 >= 0:
                _mb_n_gb_edges += 1
                _n0, _n1 = _ek  # 1-based node indices
                _mb_gb_node_set.add(_n0)
                _mb_gb_node_set.add(_n1)
                _gb_len = np.linalg.norm(_mb_p[_n0 - 1] - _mb_p[_n1 - 1])
                _mb_gb_edge_lengths.append(_gb_len)
    _mb_gb_edge_lengths = np.array(_mb_gb_edge_lengths) if _mb_gb_edge_lengths else np.array([0.0])
    _mb_n_gb_nodes = len(_mb_gb_node_set)

    # --- Domain boundary edge statistics (edges on the four domain edges)
    _mb_dom_edge_lengths = []
    _mb_dom_node_set = set()
    for _ek, _els in _mb_edge_elems.items():
        if len(_els) == 1 and _mb_on_dom_bnd(_ek[0], _ek[1]):
            _n0, _n1 = _ek  # 1-based node indices
            _mb_dom_node_set.add(_n0)
            _mb_dom_node_set.add(_n1)
            _dom_len = np.linalg.norm(_mb_p[_n0 - 1] - _mb_p[_n1 - 1])
            _mb_dom_edge_lengths.append(_dom_len)
    _mb_dom_edge_lengths = np.array(_mb_dom_edge_lengths) if _mb_dom_edge_lengths else np.array([0.0])
    _mb_n_dom_nodes = len(_mb_dom_node_set)
    _mb_n_dom_edges = len(_mb_dom_edge_lengths)

    # Store grain boundary and domain boundary statistics on ms for mesh_statistics.md
    ms._gb_n_nodes = _mb_n_gb_nodes
    ms._gb_n_edges = _mb_n_gb_edges
    ms._gb_edge_lengths = _mb_gb_edge_lengths.copy()
    ms._dom_n_nodes = _mb_n_dom_nodes
    ms._dom_n_edges = _mb_n_dom_edges
    ms._dom_edge_lengths = _mb_dom_edge_lengths.copy()

    # Assemble check lists
    _mb_validity_checks = [
        ("Duplicate node IDs",                not _mb_dup_node_ids),
        ("Duplicate node coordinates",        len(_mb_dup_coords) == 0),
        ("Duplicate element IDs",             not _mb_dup_elem_ids),
        ("Duplicate element connectivity",    len(_mb_dup_conn) == 0),
        ("References to non-existent nodes",  len(_mb_bad_refs) == 0),
        ("Degenerate elements",               len(_mb_degen) == 0),
        ("Zero-area elements",                _mb_n_zero == 0),
        ("Consistent CCW winding",            _mb_n_cw == 0),
        ("All elements in a grain elset",     len(_mb_missing_from_sets) == 0),
        ("Each element in exactly one grain", len(_mb_multi_grain_elems) == 0),
        ("All nodes referenced",              len(_mb_unreferenced) == 0),
        ("No crossing edges",                 len(_mb_crossing_pairs) == 0),
        ("No T-junction edges (>2 elems)",    len(_mb_tjunctions) == 0),
        ("No bowtie (non-manifold) nodes",    len(_mb_bowtie_nodes) == 0),
        ("Single connected component",        len(_mb_disconnected_elems) == 0),
        ("Complete domain coverage (area sum)", _mb_area_ok),
        ("No interior boundary edges (gap walls)", len(_mb_interior_bnd_edges) == 0),
        ("Single boundary loop (no interior holes)", _mb_loops_ok),
    ]
    _mb_advisory_checks = [
        (f"No edges longer than 4×mean={_mb_long_thresh:.3f}", len(_mb_long_edges) == 0),
        ("Grain boundaries are edge-aligned", len(_mb_gb_node_only) == 0),
    ]
    
    _mb_n_val_pass = sum(1 for _, ok in _mb_validity_checks if ok)
    _mb_n_val_fail = len(_mb_validity_checks) - _mb_n_val_pass
    _mb_mesh_valid = (_mb_n_val_fail == 0)
    _mb_n_adv_warn = sum(1 for _, ok in _mb_advisory_checks if not ok)
    
    # Print to console
    print(f"\n  Mesh integrity: {'VALID' if _mb_mesh_valid else 'INVALID'}"
          f"  ({_mb_n_val_pass}/{len(_mb_validity_checks)} validity, "
          f"{_mb_n_adv_warn} advisory warnings)")
    for _label, _ok in _mb_validity_checks:
        print(f"    [{'OK' if _ok else 'FAIL'}]  {_label}")
    for _label, _ok in _mb_advisory_checks:
        print(f"    [{'OK' if _ok else 'WARN'}]  {_label}")
    
    # Write mesh_final.txt
    with open(_mb_path, 'w') as _mf:
        _mf.write(f"** TESA Toolbox — Final Mesh (fd-based grain assignment)\n")
        _mf.write(f"** File    : {os.path.basename(ebsd_path)}\n")
        _mf.write(f"** Nodes   : {_mb_nn}\n")
        _mf.write(f"** Elements (CPS3): {_mb_ne}\n")
        _mf.write(f"** Grains  : {_mb_n_grains}\n")
        _mf.write(f"** Method  : Post-iteration global Delaunay + most-negative fd grain assignment\n")
        _mf.write(f"** Units   : pixel (1 pixel = 1 EBSD scan step)\n")
        _mf.write(f"**\n")
        _mf.write(f"** ── Mesh Integrity Report ──\n**\n")
    
        # Validity checks
        _mf.write(f"**   ── Validity Checks (required for FEA) ──\n")
        for _label, _ok in _mb_validity_checks:
            _mf.write(f"**     [{'OK' if _ok else 'FAIL'}]   {_label}\n")
            if not _ok:
                # Detail lines for failures
                if 'Duplicate node coord' in _label:
                    for _ct, _nids in _mb_dup_coords[:5]:
                        _mf.write(f"**              coords={_ct} nodes={_nids}\n")
                elif 'crossing' in _label.lower():
                    for _e1, _e2 in _mb_crossing_pairs[:5]:
                        _mf.write(f"**              edge {_e1} crosses edge {_e2}\n")
                elif 'T-junction' in _label:
                    for _ek, _els in _mb_tjunctions[:5]:
                        _mf.write(f"**              edge {(int(_ek[0]),int(_ek[1]))} "
                                  f"shared by {len(_els)} elems: {_els[:5]}\n")
                elif 'bowtie' in _label.lower():
                    _mf.write(f"**              bowtie node IDs: "
                              f"{[int(n) for n in _mb_bowtie_nodes[:10]]}\n")
                elif 'connected component' in _label.lower():
                    _mf.write(f"**              {len(_mb_disconnected_elems)+1} components; "
                              f"disconnected: {_mb_disconnected_elems[:10]}\n")
                elif 'area sum' in _label.lower():
                    _mf.write(f"**              domain={_mb_domain_area:.6f}  "
                              f"area_sum={_mb_area_sum:.6f}  "
                              f"gap={_mb_area_gap:.6e}\n")
                elif 'interior boundary' in _label.lower():
                    _mf.write(f"**              {len(_mb_interior_bnd_edges)} "
                              f"interior boundary edge(s):\n")
                    for _ek, _el in _mb_interior_bnd_edges[:5]:
                        _mf.write(f"**                edge {(int(_ek[0]),int(_ek[1]))} "
                                  f"→ elem {_el}\n")
                elif 'boundary loop' in _label.lower():
                    _mf.write(f"**              {_mb_n_bnd_loops} boundary loops "
                              f"({_mb_n_interior_holes} interior hole(s))\n")
                elif 'grain elset' in _label.lower():
                    _mf.write(f"**              {len(_mb_missing_from_sets)} element(s) "
                              f"not in any grain\n")
    
        _mf.write(f"**     STATUS: {'MESH VALID' if _mb_mesh_valid else 'MESH INVALID'}\n")
        _mf.write(f"**\n")
    
        # Advisory checks
        _mf.write(f"**   ── Advisory Checks (accuracy and fidelity) ──\n")
        for _label, _ok in _mb_advisory_checks:
            _mf.write(f"**     [{'OK' if _ok else 'WARN'}]   {_label}\n")
            if not _ok:
                if 'long' in _label.lower():
                    for _a, _b, _le in _mb_long_edges[:5]:
                        _mf.write(f"**              edge ({_a}, {_b}): "
                                  f"{_le:.4f} ({_le/_mb_mean_elen:.1f}×mean)\n")
                elif 'edge-aligned' in _label.lower():
                    for _nid, _ea, _ga, _eb, _gb in _mb_gb_node_only[:5]:
                        _mf.write(f"**              elems {_ea}(g{_ga}) & {_eb}(g{_gb}) "
                                  f"meet only at node {_nid}\n")
        if _mb_n_adv_warn == 0:
            _mf.write(f"**     STATUS: ALL ADVISORY OK\n")
        else:
            _mf.write(f"**     STATUS: {_mb_n_adv_warn} advisory warning(s) "
                      f"— mesh is still valid\n")
        _mf.write(f"**\n")
    
        # Summary
        _mf.write(f"**   ── Summary ──\n")
        _mf.write(f"**     Unique edges        : {len(_mb_unique_edges)}\n")
        _mf.write(f"**     Mean edge length    : {_mb_mean_elen:.4f}\n")
        _mf.write(f"**     Grain boundary edges: {_mb_n_gb_edges}\n")
        _mf.write(f"**     Triple junc. nodes  : {len(_mb_triple_junc_nodes)}\n")
        _mf.write(f"**     Winding             : {_mb_ne - _mb_n_cw - _mb_n_zero} CCW, "
                  f"{_mb_n_cw} CW, {_mb_n_zero} zero-area\n")
        _mf.write(f"**     Area sum            : {_mb_area_sum:.6f}  "
                  f"(domain: {_mb_domain_area:.6f},  gap: {_mb_area_gap:.2e})\n")
        _mf.write(f"**     Boundary loops      : {_mb_n_bnd_loops}  "
                  f"(interior holes: {_mb_n_interior_holes})\n")
        _mf.write(f"**     Interior bnd edges  : {len(_mb_interior_bnd_edges)}\n")
        _mf.write(f"**\n")
    
        # Node section
        _mf.write("*Node\n")
        for _ni, (_x, _y) in enumerate(_mb_p1, start=1):
            _mf.write(f"  {_ni:8d},  {float(_x):18.10f},  {float(_y):18.10f},   0.0000000000\n")
    
        # Element section
        _mf.write("*Element, type=CPS3\n")
        for _ei, (_n0, _n1, _n2) in enumerate(_mb_t1, start=1):
            _mf.write(f"  {_ei:8d},  {int(_n0):8d},  {int(_n1):8d},  {int(_n2):8d}\n")
    
        # Grain elset sections
        for _gi in sorted(_mb_grain_elsets.keys()):
            _elems = _mb_grain_elsets[_gi]
            _mf.write(f"*Elset, elset=Grain_{_gi+1}\n")
            for _k, _idx in enumerate(_elems):
                _sep = "\n" if (_k + 1) % 16 == 0 else ", "
                _mf.write(f"{int(_idx + 1)}{_sep}")
            _mf.write("\n")
    
    print(f"  Mesh exported to: {os.path.basename(_mb_path)}")
    print(f"    Nodes: {_mb_nn}, Elements: {_mb_ne}, Grains: {len(_mb_grain_elsets)}")
    print(f"    Status: {'VALID' if _mb_mesh_valid else 'INVALID'}")
    
    # ── Figure: Final mesh colored by grain ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    _mb_cmap = plt.cm.get_cmap(_grain_cmap_name, 20)
    _mb_verts = _mb_p[_mb_t]
    _mb_fcolors = [_mb_cmap(_grain_color_idx.get(_mb_elem_grain[i], _mb_elem_grain[i] % _N_CYCLE)) for i in range(_mb_ne)]
    from matplotlib.collections import PolyCollection as _mb_PC
    pc = _mb_PC(_mb_verts, facecolors=_mb_fcolors, edgecolors='none', linewidths=0, alpha=_grain_alpha)
    ax.add_collection(pc)
    # Mesh wireframe overlay
    ax.triplot(_mb_p[:, 0], _mb_p[:, 1], _mb_t, color='k', linewidth=_overlay_lw, alpha=_overlay_alpha)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for _sp in ax.spines.values(): _sp.set_visible(False)
    _mb_q = simpqual(_mb_p, _mb_t)
    ax.set_title(f'{_title_info}\nFinal Mesh — {_mb_ne} elements, {_mb_nn} nodes\n'
                 f'Quality: min={_mb_q.min():.4f}, mean={_mb_q.mean():.4f}  |  '
                 f'Area: {_mb_area_sum:.1f}/{_mb_domain_area:.0f}', fontsize=11)
    save(fig, _next_fig_name('final_mesh.png'))
    
    # ── Final mesh overlaid on original grain boundaries ─────────────────────
    print("  Generating final mesh on original grain boundaries overlay...")
    fig, ax = plt.subplots(figsize=(9, 9))
    # Layer 1: shaded original grain polygons (same as 03a)
    from matplotlib.patches import Polygon as _MplPoly
    from matplotlib.collections import PatchCollection as _PC
    _ogb_cmap = plt.cm.get_cmap(_grain_cmap_name, 20)
    _ogb_patches = []
    _ogb_colors = []
    for _i, _g in enumerate(grains_original):
        _g_arr = np.asarray(_g)
        if _g_arr.size > 0:
            _ogb_patches.append(_MplPoly(_g_arr, closed=True))
            _ogb_colors.append(_ogb_cmap(_grain_color_idx.get(_i, _i % _N_CYCLE)))
    if _ogb_patches:
        _ogb_pc = _PC(_ogb_patches, facecolors=_ogb_colors,
                       edgecolors='k', linewidths=0.6, alpha=_grain_alpha)
        ax.add_collection(_ogb_pc)
    # Layer 2: final mesh wireframe
    ax.triplot(_mb_p[:, 0], _mb_p[:, 1], _mb_t, color='k', linewidth=_overlay_lw, alpha=_overlay_alpha)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for _sp in ax.spines.values(): _sp.set_visible(False)
    ax.set_title(f'{_title_info}\nFinal Mesh on Original Grain Boundaries\n'
                 f'{_mb_t.shape[0]} elements, {_mb_p.shape[0]} nodes',
                 fontsize=11)
    save(fig, _next_fig_name('final_mesh_on_original_GB.png'))

    # ── Final mesh overlaid on original phase map ────────────────────────────
    print("  Generating final mesh on phase map overlay...")
    fig, ax = plt.subplots(figsize=(9, 9))
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
    # Layer 2: mesh wireframe
    ax.triplot(_mb_p[:, 0], _mb_p[:, 1], _mb_t, color='k', linewidth=_overlay_lw, alpha=_overlay_alpha)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for _sp in ax.spines.values(): _sp.set_visible(False)
    # Phase legend
    from matplotlib.patches import Patch as _PatchPM
    _pm_legend = []
    for _ph in sorted(_pm_unique_phases):
        _rgba = _mcolors_global.to_rgba(_phase_base_colors[(_ph - 1) % len(_phase_base_colors)])
        _ph_name = ms.PhaseName[_ph-1] if ms.PhaseName and _ph-1 < len(ms.PhaseName) and ms.PhaseName[_ph-1] else f"Phase {_ph}"
        _pm_legend.append(_PatchPM(facecolor=_rgba, edgecolor='k', label=f'Phase {_ph}: {_ph_name}'))
    ax.legend(handles=_pm_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.set_title(f'{_title_info}\nFinal Mesh on Phase Map\n'
                 f'{_mb_t.shape[0]} elements, {_mb_p.shape[0]} nodes',
                 fontsize=11)
    save(fig, _next_fig_name('final_mesh_on_original_phase_map.png'))

    # ── Phase-colored mesh with Euler-angle shading ──────────────────────────
    print("  Generating phase-colored figure...")
    from tesa.inpoly import inpoly as _mb_inpoly2
    from scipy.stats import mode as _mb_mode_fn
    import matplotlib.colors as _mb_mcolors
    
    _mb_grain_phases = np.zeros(_mb_n_grains, dtype=int)
    _mb_grain_angles = np.zeros((_mb_n_grains, 3))
    _mb_grain_euler_hash = np.zeros(_mb_n_grains)
    _mb_ebsd_coords = ms.OriginalDataCoordinateList
    _mb_ebsd_phase = ms.OriginalDataPhase
    _mb_ebsd_euler = ms.OriginalDataEulerAngle

    # Assign phase and Euler angles to each grain using nearest EBSD data point
    # to grain centroid (fast — avoids inpoly on all data points)
    from scipy.spatial import cKDTree as _cKDTree_ph
    _ebsd_tree = _cKDTree_ph(_mb_ebsd_coords)
    for _gi in range(_mb_n_grains):
        if Gs[_gi] is None or Gs[_gi].shape[0] < 3:
            continue
        _gc = np.mean(Gs[_gi], axis=0)
        _, _nn_idx = _ebsd_tree.query(_gc)
        _mb_grain_phases[_gi] = int(_mb_ebsd_phase[_nn_idx])
        _mb_grain_angles[_gi, :] = _mb_ebsd_euler[_nn_idx, :]
        _e_mean = _mb_grain_angles[_gi, :]
        _mb_grain_euler_hash[_gi] = (_e_mean[0]*1.0 + _e_mean[1]*2.347 + _e_mean[2]*5.713) % (2*np.pi)
    
    _mb_unique_ph = np.unique(_mb_grain_phases[_mb_grain_phases > 0])
    _mb_n_phases = len(_mb_unique_ph)
    
    import colorsys as _cs_fm
    _mb_phase_base_rgba = {ph+1: _mcolors_global.to_rgba(c) for ph, c in enumerate(_phase_base_colors)}
    
    _mb_grain_colors = np.zeros((_mb_n_grains, 4))
    for _ph in _mb_unique_ph:
        _ph_idx = np.where(_mb_grain_phases == _ph)[0]
        _n_ip = len(_ph_idx)
        if _n_ip == 0:
            continue
        _hv = _mb_grain_euler_hash[_ph_idx]
        _ranks = np.argsort(np.argsort(_hv))
        _e_norm = _ranks / (_n_ip - 1) if _n_ip > 1 else np.array([0.5])
        _base = _mb_phase_base_rgba.get(int(_ph), (0.5, 0.5, 0.5, 1.0))
        _bh, _bs, _bv = _cs_fm.rgb_to_hsv(_base[0], _base[1], _base[2])
        for _j, _gi in enumerate(_ph_idx):
            _sh = _e_norm[_j]
            _h = (_bh + 0.04 * (_sh - 0.5)) % 1.0
            _s = max(0.15, min(1.0, _bs + 0.3 * (_sh - 0.5)))
            _v = max(0.4, min(1.0, _bv + 0.35 * (0.5 - _sh)))
            _rgb = _cs_fm.hsv_to_rgb(_h, _s, _v)
            _mb_grain_colors[_gi, :3] = _rgb
            _mb_grain_colors[_gi, 3] = 1.0
    
    _mb_fc_phase = np.array([_mb_grain_colors[_mb_elem_grain[i]] for i in range(_mb_ne)])
    
    fig, ax = plt.subplots(figsize=(9, 9))
    pc = _mb_PC(_mb_verts, facecolors=_mb_fc_phase, edgecolors='none', linewidths=0, alpha=_phase_alpha)
    ax.add_collection(pc)
    # Mesh wireframe overlay
    ax.triplot(_mb_p[:, 0], _mb_p[:, 1], _mb_t, color='k', linewidth=_overlay_lw, alpha=_overlay_alpha)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for _sp in ax.spines.values(): _sp.set_visible(False)

    from matplotlib.patches import Patch as _mb_Patch
    _mb_legend = []
    for _ph in _mb_unique_ph:
        _rgba = _mb_phase_base_rgba.get(int(_ph), (0.5, 0.5, 0.5, 1.0))
        _ph_name = ms.PhaseName[int(_ph)-1] if ms.PhaseName and int(_ph)-1 < len(ms.PhaseName) and ms.PhaseName[int(_ph)-1] else f"Phase {int(_ph)}"
        _mb_legend.append(_mb_Patch(facecolor=_rgba, edgecolor='k', label=f'Phase {int(_ph)}: {_ph_name}'))
    ax.legend(handles=_mb_legend, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.set_title(f'Phase Coloring [{ebsd_name}] (Euler shading)\n'
                 f'{_mb_ne} elements, {_mb_nn} nodes  |  '
                 f'{_mb_n_phases} phases, {_mb_n_grains} grains', fontsize=11)
    save(fig, os.path.join('diagnostics', _next_fig_name('final_mesh_phase.png')))
    
    # ── Worst element quality highlight ──────────────────────────────────────
    print("  Generating worst-element figure...")
    _mb_worst_idx = np.argsort(_mb_q)[:10]
    _mb_w0 = _mb_worst_idx[0]
    _mb_wc = _mb_p[_mb_t[_mb_w0]]
    _mb_wcx, _mb_wcy = _mb_wc[:, 0].mean(), _mb_wc[:, 1].mean()
    _mb_d01 = np.linalg.norm(_mb_wc[0] - _mb_wc[1])
    _mb_d12 = np.linalg.norm(_mb_wc[1] - _mb_wc[2])
    _mb_d02 = np.linalg.norm(_mb_wc[0] - _mb_wc[2])
    _mb_w_area = 0.5 * abs((_mb_wc[1,0]-_mb_wc[0,0])*(_mb_wc[2,1]-_mb_wc[0,1]) -
                            (_mb_wc[2,0]-_mb_wc[0,0])*(_mb_wc[1,1]-_mb_wc[0,1]))
    print(f"  Worst element #{_mb_w0}: q={_mb_q[_mb_w0]:.8f}")
    print(f"    centroid=({_mb_wcx:.3f},{_mb_wcy:.3f}), area={_mb_w_area:.8f}")
    print(f"    edges=[{_mb_d01:.6f}, {_mb_d12:.6f}, {_mb_d02:.6f}]")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left panel: full mesh colored by quality
    pc1 = _mb_PC(_mb_verts, array=_mb_q, cmap='RdYlGn',
                 edgecolors=(0, 0, 0, 0.5), linewidths=0.3)
    pc1.set_clim(0, 1)
    ax1.add_collection(pc1)
    ax1.set_xlim(minx, maxx); ax1.set_ylim(miny, maxy); ax1.set_aspect('equal')
    ax1.set_xticks([]); ax1.set_yticks([])
    for _sp in ax1.spines.values(): _sp.set_visible(False)
    plt.colorbar(pc1, ax=ax1, label='Element Quality', shrink=0.8)
    # Circle all top-10 worst elements
    for _rank, _wi in enumerate(_mb_worst_idx):
        _wci = _mb_p[_mb_t[_wi]]
        ax1.plot(_wci[:,0].mean(), _wci[:,1].mean(), 'ko', markersize=10,
                 markerfacecolor='none', markeredgewidth=2.0)
    # Label only the worst element
    ax1.annotate(f'WORST (q={_mb_q[_mb_w0]:.4f})', (_mb_wcx, _mb_wcy),
                 textcoords='offset points', xytext=(12, 12), fontsize=9,
                 fontweight='bold', color='black',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax1.set_title(f'{_title_info}\nElement Quality Map (top 10 worst circled)\n'
                  f'min={_mb_q.min():.6f}, mean={_mb_q.mean():.4f}, '
                  f'elements with q<0.1: {np.sum(_mb_q < 0.1)}', fontsize=11)

    # Right panel: zoomed view around worst element
    _mb_zoom = max(3.0, max(_mb_d01, _mb_d12, _mb_d02) * 5)
    ax2.set_xlim(_mb_wcx - _mb_zoom, _mb_wcx + _mb_zoom)
    ax2.set_ylim(_mb_wcy - _mb_zoom, _mb_wcy + _mb_zoom)
    for _i in range(_mb_ne):
        _tri = _mb_p[_mb_t[_i]]
        _tcx, _tcy = _tri[:, 0].mean(), _tri[:, 1].mean()
        if abs(_tcx - _mb_wcx) < _mb_zoom + 2 and abs(_tcy - _mb_wcy) < _mb_zoom + 2:
            _color = plt.cm.RdYlGn(_mb_q[_i])
            _poly = plt.Polygon(_tri, facecolor=_color, edgecolor='k', linewidth=0.5)
            ax2.add_patch(_poly)
            if abs(_tcx - _mb_wcx) < _mb_zoom and abs(_tcy - _mb_wcy) < _mb_zoom:
                ax2.text(_tcx, _tcy, f'{_mb_q[_i]:.2f}', fontsize=5,
                         ha='center', va='center')
    _poly_w = plt.Polygon(_mb_wc, facecolor='red', edgecolor='black',
                           linewidth=3, alpha=0.7, zorder=5)
    ax2.add_patch(_poly_w)
    for _k, _v in enumerate(_mb_wc):
        ax2.plot(_v[0], _v[1], 'ko', markersize=7, zorder=6)
        ax2.annotate(f'v{_k}({_v[0]:.3f},{_v[1]:.3f})', (_v[0], _v[1]),
                     textcoords='offset points', xytext=(5, 5 + _k*12),
                     fontsize=7, fontweight='bold', zorder=6)
    ax2.set_aspect('equal')
    ax2.set_xticks([]); ax2.set_yticks([])
    for _sp in ax2.spines.values(): _sp.set_visible(False)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Zoomed: Worst Element (q={_mb_q[_mb_w0]:.6f})\n'
                  f'area={_mb_w_area:.6f}, edges=[{_mb_d01:.4f},{_mb_d12:.4f},{_mb_d02:.4f}]',
                  fontsize=11)
    fig.tight_layout()
    save(fig, os.path.join('diagnostics', _next_fig_name('final_mesh_worst_element.png')))
    
    print(f"\n  Top-10 worst elements:")
    for _rank, _wi in enumerate(_mb_worst_idx):
        _wci = _mb_p[_mb_t[_wi]]
        _qi = _mb_q[_wi]
        _ai = 0.5*abs((_wci[1,0]-_wci[0,0])*(_wci[2,1]-_wci[0,1]) -
                       (_wci[2,0]-_wci[0,0])*(_wci[1,1]-_wci[0,1]))
        print(f"    #{_rank+1} elem {_wi}: q={_qi:.6f}, area={_ai:.6f}, "
              f"centroid=({_wci[:,0].mean():.2f},{_wci[:,1].mean():.2f})")
    
    # ── Final mesh summary ────────────────────────────────────────────────────
    _mb_q_all = simpqual(_mb_p, _mb_t)
    _mb_elem_areas = np.abs(triarea(_mb_p, _mb_t))
    _mb_covered_area = float(np.sum(_mb_elem_areas))
    _mb_domain_area_s = (maxx - minx) * (maxy - miny)
    _mb_edges_01 = np.linalg.norm(_mb_p[_mb_t[:, 0]] - _mb_p[_mb_t[:, 1]], axis=1)
    _mb_edges_12 = np.linalg.norm(_mb_p[_mb_t[:, 1]] - _mb_p[_mb_t[:, 2]], axis=1)
    _mb_edges_02 = np.linalg.norm(_mb_p[_mb_t[:, 0]] - _mb_p[_mb_t[:, 2]], axis=1)
    _mb_all_edges = np.concatenate([_mb_edges_01, _mb_edges_12, _mb_edges_02])
    _mb_on_bnd = (np.isclose(_mb_p[:, 0], minx) | np.isclose(_mb_p[:, 0], maxx) |
                  np.isclose(_mb_p[:, 1], miny) | np.isclose(_mb_p[:, 1], maxy))
    _mb_n_bnd_nodes_s = int(np.sum(_mb_on_bnd))
    _mb_n_int_nodes_s = _mb_p.shape[0] - _mb_n_bnd_nodes_s
    # midside nodes for 6-node mesh
    _mb_p6_12 = (_mb_p[_mb_t[:, 0]] + _mb_p[_mb_t[:, 1]]) / 2.0
    _mb_p6_23 = (_mb_p[_mb_t[:, 1]] + _mb_p[_mb_t[:, 2]]) / 2.0
    _mb_p6_13 = (_mb_p[_mb_t[:, 0]] + _mb_p[_mb_t[:, 2]]) / 2.0
    _mb_n_midside = len(np.unique(np.vstack([_mb_p6_12, _mb_p6_23, _mb_p6_13]), axis=0))
    
    print()
    print("=" * 62)
    print("  FINAL MESH SUMMARY (Type 1 — Conforming Non-uniform)")
    print("=" * 62)
    print(f"  Domain          : [{minx:.2f}, {maxx:.2f}] x [{miny:.2f}, {maxy:.2f}]")
    print(f"  Domain area     : {_mb_domain_area_s:.4g}")
    print(f"  Covered area    : {_mb_covered_area:.4g}  ({100*_mb_covered_area/_mb_domain_area_s:.2f}%)")
    print(f"  Grains          : {num_grains}")
    print()
    print(f"  --- 3-node (linear) mesh ---")
    print(f"  Elements        : {_mb_t.shape[0]}")
    print(f"  Corner nodes    : {_mb_p.shape[0]}  "
          f"(boundary: {_mb_n_bnd_nodes_s}, interior: {_mb_n_int_nodes_s})")
    print(f"  Avg elem area   : {np.mean(_mb_elem_areas):.4g}  "
          f"(min: {np.min(_mb_elem_areas):.4g}, max: {np.max(_mb_elem_areas):.4g})")
    print(f"  Target h0       : {h0:.4g}")
    print(f"  Edge length     : mean={np.mean(_mb_all_edges):.4g}  "
          f"min={np.min(_mb_all_edges):.4g}  max={np.max(_mb_all_edges):.4g}")
    print()
    print(f"  --- Grain boundaries (interior) ---")
    print(f"  GB nodes        : {_mb_n_gb_nodes}")
    print(f"  GB edges        : {_mb_n_gb_edges}")
    _gb_ratio = np.max(_mb_gb_edge_lengths) / np.min(_mb_gb_edge_lengths) if np.min(_mb_gb_edge_lengths) > 0 else np.inf
    print(f"  GB edge length  : mean={np.mean(_mb_gb_edge_lengths):.4g}  "
          f"min={np.min(_mb_gb_edge_lengths):.4g}  max={np.max(_mb_gb_edge_lengths):.4g}  "
          f"max/min={_gb_ratio:.1f}")
    print()
    print(f"  --- Domain boundary ---")
    print(f"  DB nodes        : {_mb_n_dom_nodes}")
    print(f"  DB edges        : {_mb_n_dom_edges}")
    _db_ratio = np.max(_mb_dom_edge_lengths) / np.min(_mb_dom_edge_lengths) if np.min(_mb_dom_edge_lengths) > 0 else np.inf
    print(f"  DB edge length  : mean={np.mean(_mb_dom_edge_lengths):.4g}  "
          f"min={np.min(_mb_dom_edge_lengths):.4g}  max={np.max(_mb_dom_edge_lengths):.4g}  "
          f"max/min={_db_ratio:.1f}")
    print()
    print(f"  --- 6-node (quadratic) mesh ---")
    print(f"  Elements        : {_mb_t.shape[0]}")
    print(f"  Total nodes     : {_mb_p.shape[0] + _mb_n_midside}  "
          f"(corner: {_mb_p.shape[0]}, midside: {_mb_n_midside})")
    print()
    print(f"  --- Element quality ---")
    print(f"  Min quality     : {np.min(_mb_q_all):.6g}")
    print(f"  Mean quality    : {np.mean(_mb_q_all):.6g}")
    print(f"  Std quality     : {np.std(_mb_q_all):.6g}")
    print(f"  Slivers q<0.05  : {int(np.sum(_mb_q_all < 0.05))}")
    print(f"  Slivers q<0.10  : {int(np.sum(_mb_q_all < 0.10))}")
    print(f"  Slivers q<0.20  : {int(np.sum(_mb_q_all < 0.20))}")
    print("=" * 62)
    print()

    # ── Close TeeWriter if active (must happen before writing to log.md) ──
    if _tee is not None:
        import sys as _sys
        _sys.stdout = _tee.terminal
        _tee.close()

    # ── Write Stage 2 results to log.md ─────────────────────────────────
    if log_path is not None and vl in ("medium", "high"):
        with open(log_path, "a") as lf:
            lf.write("## Stage 2 — Conforming Mesh Generation\n\n")

            # Mesh summary table
            lf.write(f"| {'Property':<22s} | {'Value':<36s} |\n")
            lf.write(f"|{'-'*24}|{'-'*38}|\n")
            lf.write(f"| {'Domain':<22s} | {f'[{minx:.2f}, {maxx:.2f}] x [{miny:.2f}, {maxy:.2f}]':<36s} |\n")
            lf.write(f"| {'Domain area':<22s} | {_mb_domain_area_s:<36.4g} |\n")
            lf.write(f"| {'Covered area':<22s} | {f'{_mb_covered_area:.4g} ({100*_mb_covered_area/_mb_domain_area_s:.2f}%)':<36s} |\n")
            lf.write(f"| {'Grains':<22s} | {num_grains:<36d} |\n")
            lf.write(f"| {'Elements':<22s} | {_mb_t.shape[0]:<36d} |\n")
            lf.write(f"| {'Corner nodes':<22s} | {f'{_mb_p.shape[0]} (bnd: {_mb_n_bnd_nodes_s}, int: {_mb_n_int_nodes_s})':<36s} |\n")
            lf.write(f"| {'Total nodes (6-node)':<22s} | {_mb_p.shape[0] + _mb_n_midside:<36d} |\n")
            lf.write(f"| {'Target h0':<22s} | {h0:<36.4g} |\n")
            lf.write(f"| {'Edge length (mean)':<22s} | {np.mean(_mb_all_edges):<36.4g} |\n")
            lf.write(f"| {'Edge length (min)':<22s} | {np.min(_mb_all_edges):<36.4g} |\n")
            lf.write(f"| {'Edge length (max)':<22s} | {np.max(_mb_all_edges):<36.4g} |\n")
            lf.write(f"| {'GB nodes':<22s} | {_mb_n_gb_nodes:<36d} |\n")
            lf.write(f"| {'GB edges':<22s} | {_mb_n_gb_edges:<36d} |\n")
            lf.write(f"| {'GB edge length (mean)':<22s} | {np.mean(_mb_gb_edge_lengths):<36.4g} |\n")
            lf.write(f"| {'GB edge length (min)':<22s} | {np.min(_mb_gb_edge_lengths):<36.4g} |\n")
            lf.write(f"| {'GB edge length (max)':<22s} | {np.max(_mb_gb_edge_lengths):<36.4g} |\n")
            lf.write(f"| {'GB max/min ratio':<22s} | {_gb_ratio:<36.1f} |\n")
            lf.write(f"| {'DB nodes':<22s} | {_mb_n_dom_nodes:<36d} |\n")
            lf.write(f"| {'DB edges':<22s} | {_mb_n_dom_edges:<36d} |\n")
            lf.write(f"| {'DB edge length (mean)':<22s} | {np.mean(_mb_dom_edge_lengths):<36.4g} |\n")
            lf.write(f"| {'DB edge length (min)':<22s} | {np.min(_mb_dom_edge_lengths):<36.4g} |\n")
            lf.write(f"| {'DB edge length (max)':<22s} | {np.max(_mb_dom_edge_lengths):<36.4g} |\n")
            lf.write(f"| {'DB max/min ratio':<22s} | {_db_ratio:<36.1f} |\n")
            lf.write(f"| {'Min quality':<22s} | {np.min(_mb_q_all):<36.6g} |\n")
            lf.write(f"| {'Mean quality':<22s} | {np.mean(_mb_q_all):<36.6g} |\n")
            lf.write(f"| {'Slivers (q < 0.1)':<22s} | {int(np.sum(_mb_q_all < 0.10)):<36d} |\n")
            lf.write(f"| {'Mesh integrity':<22s} | {'VALID' if _mb_n_val_fail == 0 else 'INVALID':<36s} |\n\n")

            # Figure list (all mesh figures in the output folder)
            lf.write("### Figures\n\n")
            figs = sorted([fn for fn in os.listdir(FIG_DIR)
                           if fn.endswith(('.png', '.gif'))])
            for fn in figs:
                size_kb = os.path.getsize(os.path.join(FIG_DIR, fn)) / 1024
                lf.write(f"- `{fn}` ({size_kb:.0f} KB)\n")
            lf.write("\n")

            # For "high" log: append full console output
            if vl == "high" and os.path.isfile(_console_log_path):
                lf.write("### Console Output\n\n")
                lf.write("```\n")
                with open(_console_log_path, "r") as cf:
                    lf.write(cf.read())
                lf.write("```\n\n")
                # Clean up temp file
                os.remove(_console_log_path)

            lf.write("---\n\n")

    # ══════════════════════════════════════════════════════════════════════
    # Store all mesh results on ms for subsequent stages (AEH analysis)
    # ══════════════════════════════════════════════════════════════════════

    # --- 3-node (linear) mesh ---
    ms.ThreeNodeCoordinateList = _mb_p.copy()
    ms.ThreeNodeElementIndexList = _mb_t.copy()

    # --- 6-node (quadratic) mesh: insert midside nodes ---
    # Compute midside node coordinates for each element edge
    _p6_12 = (_mb_p[_mb_t[:, 0]] + _mb_p[_mb_t[:, 1]]) / 2.0  # edge 1-2
    _p6_23 = (_mb_p[_mb_t[:, 1]] + _mb_p[_mb_t[:, 2]]) / 2.0  # edge 2-3
    _p6_13 = (_mb_p[_mb_t[:, 0]] + _mb_p[_mb_t[:, 2]]) / 2.0  # edge 1-3

    # Append unique midside nodes to the coordinate list
    _all_midside = np.vstack([_p6_12, _p6_23, _p6_13])
    _unique_midside = np.unique(_all_midside, axis=0)
    _p6 = np.vstack([_mb_p, _unique_midside])

    # Build 6-node element connectivity by looking up midside node indices
    # (ismember equivalent: find index of each midside node in the full coordinate list)
    _p6_round = np.round(_p6, 12)
    _coord_to_idx = {tuple(row): idx for idx, row in enumerate(_p6_round)}
    _t4 = np.array([_coord_to_idx[tuple(np.round(row, 12))] for row in _p6_12], dtype=int)
    _t5 = np.array([_coord_to_idx[tuple(np.round(row, 12))] for row in _p6_23], dtype=int)
    _t6 = np.array([_coord_to_idx[tuple(np.round(row, 12))] for row in _p6_13], dtype=int)
    _t6_full = np.column_stack([_mb_t[:, 0], _mb_t[:, 1], _mb_t[:, 2], _t4, _t5, _t6])

    ms.SixNodeCoordinateList = _p6.copy()
    # 1-based indices for AEH solver compatibility
    ms.SixNodeElementIndexList = _t6_full.copy() + 1
    ms.NumberElements = _mb_t.shape[0]
    ms.NumberNodes = _p6.shape[0]

    # --- Boundary node pairing (periodic boundary conditions) ---
    from .compute_boundary_pairs import compute_boundary_pairs
    _bnd_rel = compute_boundary_pairs(_p6)

    ms.BoundaryNodeRelationsList = _bnd_rel.copy()

    # --- Per-element properties ---
    # Element phase: from grain assignment
    _el_phase = np.zeros(_mb_t.shape[0], dtype=int)
    _el_angle = np.zeros((_mb_t.shape[0], 3))
    _el_grain = np.zeros(_mb_t.shape[0], dtype=int)
    for _gi in range(_mb_n_grains):
        if _gi in _mb_grain_elsets:
            _el_idx = _mb_grain_elsets[_gi]
            _el_grain[_el_idx] = _gi
            _el_phase[_el_idx] = _mb_grain_phases[_gi]
            _el_angle[_el_idx, :] = _mb_grain_angles[_gi, :]

    ms.ElementPhases = _el_phase.copy()
    ms.ElementGrains = _el_grain.copy()

    # --- Per-grain properties ---
    ms.GrainPhases = _mb_grain_phases.copy()
    ms.GrainAngles = _mb_grain_angles.copy()

    # --- GrainsElements: dict mapping grain index to element indices ---
    ms.GrainsElements = {k: v.copy() for k, v in _mb_grain_elsets.items()}

    # --- Overwrite DataCoordinateList/DataPhase/DataEulerAngle with per-element values ---
    # (Original EBSD data preserved in ms.OriginalData*)
    ms.DataCoordinateList = _mb_centroids.copy()
    ms.DataPhase = _el_phase.copy()
    ms.DataEulerAngle = _el_angle.copy()

    # --- Store meshed grain boundaries ---
    ms.GrainsMeshed = [np.asarray(g).copy() if g is not None else None for g in Gs]

    # --- Mesh type ---
    ms.CurrentMeshType = 1

    print(f"\n  Stored mesh results on ms: {ms.NumberElements} elements, "
          f"{ms.NumberNodes} nodes (6-node), "
          f"{len(ms.GrainsElements)} grains")

    # Update figure counter on ms for subsequent stages
    ms.fig_count = fig_num[0]

    return ms
