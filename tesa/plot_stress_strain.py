"""
Plot microscale stress and strain fields over the microstructure.

Visualizes stress and strain components from ms.Microfield as contour plots.
Supports three styles:
  - "element": flat shading (one color per element, averaged from quadrature points)
  - "smooth": Gouraud shading (interpolated to nodes for smooth gradients)
  - "rbf": grain-aware RBF interpolation (smooth within grains, discontinuous across
    grain boundaries — physically correct)

Plots are saved to subfolder(s) within the analysis plots directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay
from . import fem_definitions as FEMDef
from .dpoly import dpoly
from .tricentroid import tricentroid


# ── Field definitions ─────────────────────────────────────────────────────
# Each entry: (microfield_index, filename, label, unit, scale_factor)
# scale_factor converts from SI (Pa, dimensionless) to display units (MPa, microstrain)
STRESS_FIELDS = [
    (2,  "sigma_11",  r"$\sigma_{11}$",  "MPa", 1e-6),
    (3,  "sigma_22",  r"$\sigma_{22}$",  "MPa", 1e-6),
    (4,  "sigma_33",  r"$\sigma_{33}$",  "MPa", 1e-6),
    (5,  "sigma_23",  r"$\sigma_{23}$",  "MPa", 1e-6),
    (6,  "sigma_13",  r"$\sigma_{13}$",  "MPa", 1e-6),
    (7,  "sigma_12",  r"$\sigma_{12}$",  "MPa", 1e-6),
    (8,  "sigma_p1",  r"$\sigma_{p1}$ (max principal)", "MPa", 1e-6),
    (9,  "sigma_p2",  r"$\sigma_{p2}$ (mid principal)", "MPa", 1e-6),
    (10, "sigma_p3",  r"$\sigma_{p3}$ (min principal)", "MPa", 1e-6),
    (11, "tau_max",   r"$\tau_{max}$",   "MPa", 1e-6),
]

STRAIN_FIELDS = [
    (13, "epsilon_11", r"$\varepsilon_{11}$", r"$\mu\varepsilon$", 1e6),
    (14, "epsilon_22", r"$\varepsilon_{22}$", r"$\mu\varepsilon$", 1e6),
    (15, "epsilon_33", r"$\varepsilon_{33}$", r"$\mu\varepsilon$", 1e6),
    (16, "epsilon_23", r"$\varepsilon_{23}$", r"$\mu\varepsilon$", 1e6),
    (17, "epsilon_13", r"$\varepsilon_{13}$", r"$\mu\varepsilon$", 1e6),
    (18, "epsilon_12", r"$\varepsilon_{12}$", r"$\mu\varepsilon$", 1e6),
    (19, "epsilon_p1", r"$\varepsilon_{p1}$ (max principal)", r"$\mu\varepsilon$", 1e6),
    (20, "epsilon_p2", r"$\varepsilon_{p2}$ (mid principal)", r"$\mu\varepsilon$", 1e6),
    (21, "epsilon_p3", r"$\varepsilon_{p3}$ (min principal)", r"$\mu\varepsilon$", 1e6),
]

ALL_FIELDS = STRESS_FIELDS + STRAIN_FIELDS


def _qp_to_element_avg(mf_values, nElements):
    """
    Average quadrature point values to element-level values.

    Microfield arrays have nQP * nElements entries, interleaved:
    [QP0_elem0, QP1_elem0, QP2_elem0, QP3_elem0, QP0_elem1, ...]

    Parameters
    ----------
    mf_values : (nQP * nElements,) array
        Field values at all quadrature points.
    nElements : int
        Number of elements.

    Returns
    -------
    elem_values : (nElements,) array
        Element-averaged field values.
    """
    nQP = FEMDef.N_QUADRATURE_POINTS
    # Reshape to (nElements, nQP) and average across quadrature points
    reshaped = mf_values.reshape(-1, nQP) if mf_values.size == nElements * nQP else \
               mf_values[:nElements * nQP].reshape(-1, nQP)
    return np.mean(reshaped, axis=1)


def _compute_field_mean(field_data, ms):
    """
    Compute area-weighted mean of a field using full quadrature integration.

    mean = Σ(field_qp × weight_qp × det_J_qp) / Σ(weight_qp × det_J_qp)

    Parameters
    ----------
    field_data : (nQP * nElements,) array
        Field values at all quadrature points (element-major ordering).
    ms : Microstructure
        Must have quadraturePointJacobian stored (nElements, nQP).

    Returns
    -------
    mean : float
        Area-weighted mean of the field over the domain.
    """
    nQP = FEMDef.N_QUADRATURE_POINTS
    weights = FEMDef.QUADRATURE_WEIGHTS  # (nQP,) array

    detJ = getattr(ms, 'quadraturePointJacobian', None)
    if detJ is None:
        # Fallback: simple average if Jacobian not available
        return np.mean(field_data)

    nElements = detJ.shape[0]
    # Reshape field to (nElements, nQP) — element-major ordering
    field_qp = field_data[:nElements * nQP].reshape(nElements, nQP)

    # detJ is (nElements, nQP)
    # weights is (nQP,) — broadcast over elements
    # Integrate: Σ(field × weight × |detJ|)
    numerator = np.sum(field_qp * weights[np.newaxis, :] * np.abs(detJ))
    denominator = np.sum(weights[np.newaxis, :] * np.abs(detJ))

    if denominator < 1e-30:
        return np.mean(field_data)

    return numerator / denominator


def _qp_to_nodal(mf_values, nElements, nCornerNodes, t3):
    """
    Extrapolate quadrature point values to corner nodes by averaging
    contributions from adjacent elements.

    Parameters
    ----------
    mf_values : (nQP * nElements,) array
        Field values at all quadrature points.
    nElements : int
        Number of elements.
    nCornerNodes : int
        Number of corner nodes (3-node mesh).
    t3 : (nElements, 3) array
        3-node element connectivity (0-based).

    Returns
    -------
    nodal_values : (nCornerNodes,) array
        Nodal field values (averaged from adjacent elements).
    """
    # First get element averages
    elem_avg = _qp_to_element_avg(mf_values, nElements)

    # Accumulate element averages to nodes
    nodal_sum = np.zeros(nCornerNodes)
    nodal_count = np.zeros(nCornerNodes)
    for iElem in range(nElements):
        for iNode in range(3):
            nid = t3[iElem, iNode]
            nodal_sum[nid] += elem_avg[iElem]
            nodal_count[nid] += 1

    # Average (avoid division by zero for unreferenced nodes)
    nodal_count[nodal_count == 0] = 1
    return nodal_sum / nodal_count


def _build_rbf_cache(ms, p3, t3, qp_coords=None):
    """
    Build the grain-aware RBF cache for field interpolation.

    Computes grain assignments, LU factorizations of the RBF matrices,
    and evaluation matrices for each grain. The cache is computed once
    and reused for all subsequent field plots.

    Parameters
    ----------
    ms : Microstructure
        Must have GrainsMeshed (or Grains), Microfield, and optionally
        GrainHoles populated.
    p3 : ndarray, shape (n_nodes, 2)
        3-node (corner) mesh node coordinates.
    t3 : ndarray, shape (n_elements, 3)
        3-node element connectivity (0-based).
    qp_coords : tuple of (ndarray, ndarray) or None, optional
        (qp_x, qp_y) quadrature point coordinates. If None, uses
        ms.Microfield[0] and ms.Microfield[1].

    Returns
    -------
    grain_cache : list of dict
        Per-grain cache entries containing RBF matrices, LU factors,
        evaluation matrices, and grain-local indices.
    unassigned_cache : dict or None
        Information for nearest-neighbor gap filling of elements not
        assigned to any grain. None if all elements are assigned.
    """
    from scipy.linalg import lu_factor
    nQP = FEMDef.N_QUADRATURE_POINTS

    # Get quadrature point coordinates
    if qp_coords is not None:
        qp_x, qp_y = qp_coords
    else:
        qp_x = ms.Microfield[0]
        qp_y = ms.Microfield[1]

    Gm = getattr(ms, 'GrainsMeshed', None)
    if Gm is None:
        Gm = getattr(ms, 'Grains', [])
    Holes = getattr(ms, 'GrainHoles', None)

    tC = tricentroid(p3, t3)

    qp_remaining = np.ones(len(qp_x), dtype=bool)
    elem_remaining = np.ones(t3.shape[0], dtype=bool)
    mesh_type = getattr(ms, 'CurrentMeshType', 1)

    grain_cache = []

    for n in range(len(Gm)):
        g = Gm[n]
        if g is None:
            continue
        g = np.asarray(g)
        if g.size == 0:
            continue

        closed = np.vstack([g, g[0:1]])

        has_holes = False
        if Holes is not None:
            if isinstance(Holes, dict):
                has_holes = Holes.get(n) is not None and len(Holes[n]) > 0
            elif isinstance(Holes, list) and n < len(Holes):
                has_holes = Holes[n] is not None and len(Holes[n]) > 0
        if has_holes:
            from .ddiff_multi import ddiff_multi
            fd = lambda pts, _n=n, _h=Holes[n]: ddiff_multi(pts, Gm, _n, _h)
        else:
            fd = lambda pts, _c=closed: dpoly(pts, _c)

        qp_xy = np.column_stack([qp_x, qp_y])
        d_qp = fd(qp_xy[qp_remaining])
        in_grain_local = d_qp <= 0
        in_grain_global = np.where(qp_remaining)[0][in_grain_local]

        xg = qp_x[in_grain_global]
        yg = qp_y[in_grain_global]

        qp_remaining[in_grain_global] = False

        if len(xg) < 4:
            continue

        d_elem = fd(tC[elem_remaining])
        t_in_local = d_elem <= 0
        t_in_global = np.where(elem_remaining)[0][t_in_local]
        elem_remaining[t_in_global] = False

        t_grain = t3[t_in_global]
        unique_nodes = np.unique(t_grain.ravel())
        pg = p3[unique_nodes]

        if pg.shape[0] < 3:
            continue
        try:
            tg_local = Delaunay(pg).simplices
        except Exception:
            continue

        tgC = (pg[tg_local[:, 0]] + pg[tg_local[:, 1]] + pg[tg_local[:, 2]]) / 3.0
        d_tg = fd(tgC)
        tg_in = tg_local[d_tg <= 0]

        if len(tg_in) == 0:
            continue

        used_nodes = np.unique(tg_in.ravel())

        # Downsample if too many QPs
        step = 1
        if len(xg) > 33000:
            step = 3
            xg = xg[::step]
            yg = yg[::step]

        # Build and factorize the RBF matrix (reusable across all fields)
        dx_mat = xg[:, None] - xg[None, :]
        dy_mat = yg[:, None] - yg[None, :]
        r_mat = np.sqrt(dx_mat**2 + dy_mat**2)
        A = np.sqrt(r_mat + 1.0)

        try:
            lu_piv = lu_factor(A)
            use_lstsq = False
        except Exception:
            lu_piv = None
            use_lstsq = True

        # Build evaluation matrix B: B[node, qp] = sqrt(r_node_qp + 1)
        B = np.zeros((len(used_nodes), len(xg)))
        for k, node_idx in enumerate(used_nodes):
            px, py = pg[node_idx, 0], pg[node_idx, 1]
            r_node = np.sqrt((px - xg)**2 + (py - yg)**2)
            B[k, :] = np.sqrt(r_node + 1.0)

        grain_cache.append({
            'in_grain_global': in_grain_global,
            'step': step,
            'pg': pg,
            'tg_in': tg_in,
            'used_nodes': used_nodes,
            'lu_piv': lu_piv,
            'use_lstsq': use_lstsq,
            'A': A,
            'B': B,
            'mesh_type': mesh_type,
        })

    # Cache unassigned element info for gap filling
    unassigned_cache = None
    n_unassigned_elem = np.sum(elem_remaining)
    if n_unassigned_elem > 0:
        from scipy.spatial import cKDTree
        assigned_mask = ~qp_remaining
        assigned_xy = np.column_stack([qp_x[assigned_mask], qp_y[assigned_mask]])
        if len(assigned_xy) > 0:
            unassigned_elem_idx = np.where(elem_remaining)[0]
            unassigned_centroids = tC[unassigned_elem_idx]
            tree = cKDTree(assigned_xy)
            _, nn_idx = tree.query(unassigned_centroids)
            assigned_indices = np.where(assigned_mask)[0]
            unassigned_cache = {
                'unassigned_elem_idx': unassigned_elem_idx,
                'nn_global_idx': assigned_indices[nn_idx],
            }

    return grain_cache, unassigned_cache


def _plot_rbf_field(ax, ms, field_data, p3, t3, scale_factor, cmap='RdBu_r', qp_coords=None):
    """
    Grain-aware RBF interpolation of quadrature point field values.

    Uses cached RBF matrices (stored on ms._rbf_cache) to avoid rebuilding
    the grain assignments and matrix factorizations for each field plot.
    The cache is built on the first call and reused for subsequent calls.

    Parameters
    ----------
    ax : matplotlib Axes
    ms : Microstructure
    field_data : (nQP*nElements,) array
        Scaled field values at all quadrature points.
    p3 : (nCornerNodes, 2) array
        3-node mesh coordinates.
    t3 : (nElements, 3) array
        3-node element connectivity (0-based).
    scale_factor : float
        Already applied to field_data; needed for colorbar range.
    cmap : str
        Colormap name.
    qp_coords : tuple or None
        (qp_x, qp_y) arrays, or None to use ms.Microfield[0], [1].

    Returns
    -------
    sm : ScalarMappable
        For colorbar.
    vmin, vmax : float
        Color limits used.
    """
    from scipy.linalg import lu_solve

    # Build or retrieve the RBF cache
    if not hasattr(ms, '_rbf_cache') or ms._rbf_cache is None:
        ms._rbf_cache = _build_rbf_cache(ms, p3, t3, qp_coords=qp_coords)

    grain_cache, unassigned_cache = ms._rbf_cache

    # For each cached grain, solve for new field values and evaluate at nodes
    all_nodal_values = []
    grain_plot_data = []

    for gc in grain_cache:
        # Extract field values for this grain's QPs
        fg = field_data[gc['in_grain_global']]
        if gc['step'] > 1:
            fg = fg[::gc['step']]

        # Solve A @ lambda = fg using cached LU factorization
        if gc['use_lstsq']:
            lam, _, _, _ = np.linalg.lstsq(gc['A'], fg, rcond=None)
        else:
            lam = lu_solve(gc['lu_piv'], fg)

        # Evaluate at nodes using cached evaluation matrix: nodal = B @ lambda
        nodal_at_used = gc['B'] @ lam

        # Build full nodal field array
        nodal_field = np.full(gc['pg'].shape[0], np.nan)
        nodal_field[gc['used_nodes']] = nodal_at_used

        # Clip for non-conforming meshes
        if gc['mesh_type'] != 1:
            fg_min = np.min(fg)
            fg_max = np.max(fg)
            nodal_field[gc['used_nodes']] = np.clip(nodal_field[gc['used_nodes']], fg_min, fg_max)

        all_nodal_values.append(nodal_field[gc['used_nodes']])
        grain_plot_data.append((gc['pg'], gc['tg_in'], nodal_field))

    # Fill unassigned elements (gap filling)
    if unassigned_cache is not None and len(grain_plot_data) > 0:
        nn_field_vals = field_data[unassigned_cache['nn_global_idx']]
        for i, elem_idx in enumerate(unassigned_cache['unassigned_elem_idx']):
            t_elem = t3[elem_idx]
            unique_n = np.unique(t_elem)
            pg_fill = p3[unique_n]
            if pg_fill.shape[0] < 3:
                continue
            node_map = {g: l for l, g in enumerate(unique_n)}
            tg_fill = np.array([[node_map[t_elem[0]], node_map[t_elem[1]], node_map[t_elem[2]]]])
            nodal_fill = np.full(pg_fill.shape[0], nn_field_vals[i])
            all_nodal_values.append(np.array([nn_field_vals[i]]))
            grain_plot_data.append((pg_fill, tg_fill, nodal_fill))

    # Compute color limits: mean ± 2.5σ
    if all_nodal_values:
        all_data = np.concatenate(all_nodal_values)
        data_mean = np.mean(all_data)
        data_std = np.std(all_data)
        vmin = max(np.min(all_data), data_mean - 2.5 * data_std)
        vmax = min(np.max(all_data), data_mean + 2.5 * data_std)
    else:
        vmin, vmax = 0, 1

    # Plot each grain's triangulation
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.cm.get_cmap(cmap)

    for pg, tg_in, nodal_field in grain_plot_data:
        tri_grain = Triangulation(pg[:, 0], pg[:, 1], tg_in)
        ax.tripcolor(tri_grain, nodal_field, shading='gouraud',
                     cmap=cmap, norm=norm)

    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])

    return sm, vmin, vmax


def plot_stress_strain_field(ms, field_index, field_name, field_label, field_unit,
                              scale_factor=1.0, analysis_dir=None, settings=None,
                              subfolder="stress_strain"):
    """
    Plot a single stress or strain field over the microstructure.

    Parameters
    ----------
    ms : Microstructure
        Must have Microfield, ThreeNodeCoordinateList, ThreeNodeElementIndexList.
    field_index : int
        Index into ms.Microfield list.
    field_name : str
        Filename stem (e.g., "sigma_11").
    field_label : str
        LaTeX label for colorbar (e.g., "$\\sigma_{11}$").
    field_unit : str
        Unit string for colorbar label.
    scale_factor : float
        Multiply raw SI values by this for display (e.g., 1e-6 for Pa→MPa, 1e6 for strain→microstrain).
    analysis_dir : str or None
        Directory to save the figure.
    settings : dict or None
        Global settings (figure_dpi, field_plot_style).

    Returns
    -------
    fig : matplotlib Figure
    """
    if settings is None:
        settings = {}
    figure_dpi = settings.get("figure_dpi", 150)
    figure_fontsize = settings.get("figure_fontsize", 12)
    figure_title_fontsize = settings.get("figure_title_fontsize", 14)

    # Determine plot style based on mesh type
    # "default" resolves to: rbf for Type 1 (conforming), smooth for Type 2 (non-conforming)
    mesh_type = getattr(ms, 'CurrentMeshType', 1)
    user_style = settings.get("field_plot_style", "default")
    if mesh_type == 1:
        plot_style = "rbf"
    elif user_style in ("default", "rbf"):
        plot_style = "smooth"
    else:
        plot_style = user_style  # "element" or "smooth"

    # Extract field data from Microfield and apply scale factor for display
    mf = ms.Microfield
    field_data = mf[field_index] * scale_factor

    # Mesh data
    p3 = ms.ThreeNodeCoordinateList
    t3 = ms.ThreeNodeElementIndexList  # 0-based
    nElements = t3.shape[0]
    nCornerNodes = p3.shape[0]

    # Create triangulation
    tri = Triangulation(p3[:, 0], p3[:, 1], t3)

    # EBSD file name for title
    ebsd_name = os.path.basename(getattr(ms, 'ebsd_file', ''))

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Colorbar label
    cb_title = f'{field_label}\n({field_unit})' if field_unit else field_label

    # Get colormap from settings
    cmap_name = settings.get("field_colormap", "jet")

    # Compute robust color limits: mean ± 2.5σ (clips corner singularities)
    import matplotlib.colors as _mcolors
    data_mean = np.mean(field_data)
    data_std = np.std(field_data)
    vmin_clip = max(np.min(field_data), data_mean - 2.5 * data_std)
    vmax_clip = min(np.max(field_data), data_mean + 2.5 * data_std)
    robust_norm = _mcolors.Normalize(vmin=vmin_clip, vmax=vmax_clip)

    if plot_style == "rbf":
        # Grain-aware RBF interpolation (smooth within grains, discontinuous at boundaries)
        sm, vmin, vmax = _plot_rbf_field(ax, ms, field_data, p3, t3, scale_factor, cmap=cmap_name)
        cb = fig.colorbar(sm, ax=ax, shrink=0.85)
        cb.ax.set_title(cb_title, fontsize=figure_fontsize, pad=figure_fontsize * 0.6)
        cb.ax.tick_params(labelsize=figure_fontsize)
    elif plot_style == "smooth":
        # Interpolate to nodes for smooth Gouraud shading
        nodal_values = _qp_to_nodal(field_data, nElements, nCornerNodes, t3)
        tpc = ax.tripcolor(tri, nodal_values, shading='gouraud', cmap=cmap_name, norm=robust_norm)
        cb = fig.colorbar(tpc, ax=ax, shrink=0.85)
        cb.ax.set_title(cb_title, fontsize=figure_fontsize, pad=figure_fontsize * 0.6)
        cb.ax.tick_params(labelsize=figure_fontsize)
    else:
        # Element-level flat shading (default)
        elem_values = _qp_to_element_avg(field_data, nElements)
        tpc = ax.tripcolor(tri, facecolors=elem_values, cmap=cmap_name, norm=robust_norm)
        cb = fig.colorbar(tpc, ax=ax, shrink=0.85)
        cb.ax.set_title(cb_title, fontsize=figure_fontsize, pad=figure_fontsize * 0.6)
        cb.ax.tick_params(labelsize=figure_fontsize)

    # Overlay original (unsmoothed) grain boundaries
    if settings.get("show_grain_boundaries", True):
        grains = getattr(ms, 'Grains', None)
        if grains is not None:
            for g in grains:
                if g is not None:
                    g = np.asarray(g)
                    if g.size > 0:
                        closed = np.vstack([g, g[0:1]])
                        ax.plot(closed[:, 0], closed[:, 1], '-', color='gray', linewidth=0.3, alpha=0.3)

    # Formatting — clip to mesh domain
    ax.set_xlim(p3[:, 0].min(), p3[:, 0].max())
    ax.set_ylim(p3[:, 1].min(), p3[:, 1].max())
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title — line 1: file + field, line 2: min/max
    fmin = np.min(field_data)
    fmax = np.max(field_data)
    fmean = _compute_field_mean(field_data, ms)
    max_abs = np.max(np.abs(field_data))
    val_range = fmax - fmin
    title_line1 = f'{ebsd_name}: Microscale field {field_label}'
    title_line2 = f'Min = {fmin:.2f}, Mean = {fmean:.2f}, Max = {fmax:.2f} {field_unit}'
    if max_abs < 1e-20:
        title_line2 += '  ⚠ Values ≈ 0'
    elif val_range / (max_abs + 1e-30) < 1e-6:
        title_line2 += '  ⚠ Uniform field'
    elif max_abs < 1e-6:
        title_line2 += f'  ⚠ Very small values'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=figure_title_fontsize)

    # Save to {subfolder}
    if analysis_dir:
        ss_dir = os.path.join(analysis_dir, subfolder)
        os.makedirs(ss_dir, exist_ok=True)
        filepath = os.path.join(ss_dir, f"{field_name}.png")
        fig.savefig(filepath, dpi=figure_dpi, bbox_inches='tight')

    plt.close(fig)
    return fig


def plot_all_stress_strain(ms, analysis_dir=None, settings=None, verbose=True):
    """
    Plot all stress and strain fields from ms.Microfield.

    Saves to "plots/stress_strain" subfolder within analysis_dir.

    Parameters
    ----------
    ms : Microstructure
        Must have Microfield populated.
    analysis_dir : str or None
        Directory to save figures.
    settings : dict or None
        Global settings.
    verbose : bool
        Print progress to console.

    Returns
    -------
    fig_count : int
        Number of figures generated.
    """
    if not hasattr(ms, 'Microfield') or ms.Microfield is None:
        if verbose:
            print("  WARNING: No Microfield data — skipping stress/strain plots")
        return 0

    # Determine plot style based on mesh type
    mesh_type = getattr(ms, 'CurrentMeshType', 1)
    user_style = settings.get("field_plot_style", "default") if settings else "default"
    if mesh_type == 1:
        plot_style = "rbf"
    elif user_style in ("default", "rbf"):
        plot_style = "smooth"
    else:
        plot_style = user_style
    if verbose:
        print(f"  Generating stress/strain field plots ({plot_style} style, "
              f"{len(ALL_FIELDS)} figures)...")

    # Plot stress fields to plots/stresses/
    count_stress = 0
    for field_index, field_name, field_label, field_unit, scale in STRESS_FIELDS:
        if field_index >= len(ms.Microfield):
            continue
        plot_stress_strain_field(ms, field_index, field_name, field_label, field_unit,
                                 scale_factor=scale, analysis_dir=analysis_dir,
                                 settings=settings, subfolder="stresses")
        count_stress += 1

    # Plot strain fields to plots/strains/
    count_strain = 0
    for field_index, field_name, field_label, field_unit, scale in STRAIN_FIELDS:
        if field_index >= len(ms.Microfield):
            continue
        plot_stress_strain_field(ms, field_index, field_name, field_label, field_unit,
                                 scale_factor=scale, analysis_dir=analysis_dir,
                                 settings=settings, subfolder="strains")
        count_strain += 1

    if verbose:
        print(f"  Saved {count_stress} stress plots to plots/stresses/")
        print(f"  Saved {count_strain} strain plots to plots/strains/")

    return count_stress + count_strain
