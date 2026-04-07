"""
Plot microscale heat flux and temperature gradient fields over the microstructure.

Visualizes heat flux and temperature gradient components from ms.MicrofieldHeatConduction.
Supports three styles: "element", "smooth", "rbf" (grain-aware RBF interpolation).

Heat flux plots saved to plots/heat_flux/, temperature gradient to plots/temp_gradient/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


# ── Arrow plot settings ───────────────────────────────────────────────────
# These control the appearance of quiver arrows in vector plots.
# User settings (in run_tesa.py): arrow_length, arrow_stem_width, arrow_alpha
#   arrow_length = arrow_length * grid_spacing * field_value / max_field
#   shaft_width  = arrow_stem_width (direct, as fraction of plot width)
#   head dims    = headwidth/headlength/headaxislength × shaft_width
_ARROW_SETTINGS = {
    "headwidth": 3.5,           # Head width as multiple of shaft width
    "headlength": 4,            # Head length as multiple of shaft width
    "headaxislength": 3.5,      # Where head meets shaft as multiple of shaft width
    "linewidth": 0.2,           # Outline stroke width of arrows
}

# ── Field definitions ─────────────────────────────────────────────────────
# Each entry: (MicrofieldHeatConduction_index, filename, label, unit, scale_factor)

HEAT_FLUX_FIELDS = [
    (2, "q_1",             r"$q_1$",          "W/m²", 1.0),
    (3, "q_2",             r"$q_2$",          "W/m²", 1.0),
    (4, "q_3",             r"$q_3$",          "W/m²", 1.0),
    (5, "q_magnitude_3D",  r"$|q|$",          "W/m²", 1.0),
]

# 2D in-plane magnitude — computed from q1, q2 (not stored in MicrofieldHeatConduction)
# Uses special index -1 to signal computation at plot time
HEAT_FLUX_2D_MAG = (-1, "q_magnitude_2D", r"$|q_{12}|$", "W/m²", 1.0)

TEMP_GRADIENT_FIELDS = [
    (6, "grad_T_1",             r"$\partial T / \partial x_1$",  "K/m", 1.0),
    (7, "grad_T_2",             r"$\partial T / \partial x_2$",  "K/m", 1.0),
    (8, "grad_T_3",             r"$\partial T / \partial x_3$",  "K/m", 1.0),
    (9, "grad_T_magnitude_3D",  r"$|\nabla T|$",                 "K/m", 1.0),
]

# 2D in-plane temperature gradient magnitude
TEMP_GRADIENT_2D_MAG = (-1, "grad_T_magnitude_2D", r"$|\nabla T_{12}|$", "K/m", 1.0)


def _qp_to_element_avg(field_data, nElements):
    """
    Average quadrature point values to one value per element.

    Parameters
    ----------
    field_data : ndarray, shape (n_qp * n_elements,)
        Field values at all quadrature points, interleaved by element.
    nElements : int
        Number of elements.

    Returns
    -------
    elem_values : ndarray, shape (n_elements,)
        Element-averaged field values.
    """
    from . import fem_definitions as FEMDef
    nQP = FEMDef.N_QUADRATURE_POINTS
    elem_values = np.zeros(nElements)
    for iElem in range(nElements):
        vals = field_data[iElem * nQP:(iElem + 1) * nQP]
        elem_values[iElem] = np.mean(vals)
    return elem_values


def _qp_to_nodal(field_data, nElements, nCornerNodes, t3):
    """
    Extrapolate quadrature point values to corner nodes for smooth shading.

    Averages element-level values at each node from all adjacent elements.

    Parameters
    ----------
    field_data : ndarray, shape (n_qp * n_elements,)
        Field values at all quadrature points.
    nElements : int
        Number of elements.
    nCornerNodes : int
        Number of corner nodes (3-node mesh).
    t3 : ndarray, shape (n_elements, 3)
        3-node element connectivity (0-based).

    Returns
    -------
    nodal_values : ndarray, shape (n_corner_nodes,)
        Nodal field values averaged from adjacent elements.
    """
    elem_values = _qp_to_element_avg(field_data, nElements)
    nodal_sum = np.zeros(nCornerNodes)
    nodal_count = np.zeros(nCornerNodes)
    for iElem in range(nElements):
        for iNode in range(3):
            nid = t3[iElem, iNode]
            nodal_sum[nid] += elem_values[iElem]
            nodal_count[nid] += 1
    nodal_count[nodal_count == 0] = 1
    return nodal_sum / nodal_count


def plot_heat_flux_field(ms, field_index, field_name, field_label, field_unit,
                          scale_factor=1.0, analysis_dir=None, settings=None,
                          subfolder="heat_flux", field_data_override=None):
    """
    Plot a single heat flux or temperature gradient field.

    Parameters
    ----------
    ms : Microstructure
        Must have MicrofieldHeatConduction, ThreeNodeCoordinateList,
        ThreeNodeElementIndexList.
    field_index : int or None
        Index into ms.MicrofieldHeatConduction list. Ignored if field_data_override is set.
    field_name : str
        Filename stem.
    field_label : str
        LaTeX label for colorbar.
    field_unit : str
        Unit string for colorbar label.
    scale_factor : float
        Multiply raw values by this for display.
    analysis_dir : str or None
        Base analysis directory.
    settings : dict or None
        Global settings (figure_dpi, field_plot_style, field_colormap).
    subfolder : str
        Subfolder name within plots/.
    field_data_override : ndarray or None, optional
        If provided, use this data instead of MicrofieldHeatConduction[field_index].

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure (closed after saving).
    """
    if settings is None:
        settings = {}
    figure_dpi = settings.get("figure_dpi", 150)
    cmap_name = settings.get("field_colormap", "jet")
    field_fontsize = settings.get("field_fontsize", 12)

    # Determine plot style based on mesh type
    # "default" resolves to: rbf for Type 1 (conforming), smooth for Type 2 (non-conforming)
    mesh_type = getattr(ms, 'CurrentMeshType', 1)
    user_style = settings.get("field_plot_style", "default")
    if mesh_type == 1:
        plot_style = "rbf"
    elif user_style in ("default", "rbf"):
        plot_style = "smooth"
    else:
        plot_style = user_style

    # Extract field data and apply scale factor
    if field_data_override is not None:
        field_data = field_data_override * scale_factor
    else:
        mf = ms.MicrofieldHeatConduction
        field_data = mf[field_index] * scale_factor

    # Mesh data
    p3 = ms.ThreeNodeCoordinateList
    t3 = ms.ThreeNodeElementIndexList
    nElements = t3.shape[0]
    nCornerNodes = p3.shape[0]

    # Create triangulation
    tri = Triangulation(p3[:, 0], p3[:, 1], t3)

    # EBSD file name for title
    ebsd_name = os.path.basename(getattr(ms, 'ebsd_file', ''))

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Colorbar label
    cb_label = f'{field_label} ({field_unit})' if field_unit else field_label

    # Compute robust color limits: mean ± 2.5σ (clips corner singularities)
    import matplotlib.colors as _mcolors
    data_mean = np.mean(field_data)
    data_std = np.std(field_data)
    vmin_clip = max(np.min(field_data), data_mean - 2.5 * data_std)
    vmax_clip = min(np.max(field_data), data_mean + 2.5 * data_std)
    robust_norm = _mcolors.Normalize(vmin=vmin_clip, vmax=vmax_clip)

    if plot_style == "rbf":
        # Grain-aware RBF interpolation
        # Ensure QP coordinates are available (use heat conduction coords if Microfield not set)
        if ms.Microfield is None or ms.Microfield[0] is None:
            ms.Microfield = [None] * max(len(ms.Microfield) if ms.Microfield else 0, 22)
            ms.Microfield[0] = mf[0]  # qx from MicrofieldHeatConduction
            ms.Microfield[1] = mf[1]  # qy from MicrofieldHeatConduction
        from .plot_stress_strain import _plot_rbf_field
        sm, vmin, vmax = _plot_rbf_field(ax, ms, field_data, p3, t3, scale_factor, cmap=cmap_name)
        cb = fig.colorbar(sm, ax=ax, shrink=0.85)
        cb.set_label(cb_label, fontsize=field_fontsize)
        cb.ax.tick_params(labelsize=field_fontsize)
    elif plot_style == "smooth":
        # Smooth Gouraud shading
        nodal_values = _qp_to_nodal(field_data, nElements, nCornerNodes, t3)
        tpc = ax.tripcolor(tri, nodal_values, shading='gouraud', cmap=cmap_name, norm=robust_norm)
        cb = fig.colorbar(tpc, ax=ax, shrink=0.85)
        cb.set_label(cb_label, fontsize=field_fontsize)
        cb.ax.tick_params(labelsize=field_fontsize)
    else:
        # Element-level flat shading
        elem_values = _qp_to_element_avg(field_data, nElements)
        tpc = ax.tripcolor(tri, facecolors=elem_values, cmap=cmap_name, norm=robust_norm)
        cb = fig.colorbar(tpc, ax=ax, shrink=0.85)
        cb.set_label(cb_label, fontsize=field_fontsize)
        cb.ax.tick_params(labelsize=field_fontsize)

    # Overlay original grain boundaries
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

    # Title — line 1: file + field, line 2: min/mean/max
    from .plot_stress_strain import _compute_field_mean
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
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=12)

    # Save figure
    if analysis_dir is not None:
        plot_dir = os.path.join(analysis_dir, subfolder)
        os.makedirs(plot_dir, exist_ok=True)
        fig_path = os.path.join(plot_dir, f"{field_name}.png")
        fig.savefig(fig_path, dpi=figure_dpi, bbox_inches='tight')

    # Microscale field plots are not displayed on screen (too many figures)
    # They are saved to files only
    plt.close(fig)

    return fig


def plot_heat_flux_vectors(ms, analysis_dir=None, settings=None,
                           field_type="heat_flux", subfolder="heat_flux"):
    """
    Plot heat flux (or temperature gradient) magnitude with vector arrows on a grid.

    Creates a filled contour of the in-plane magnitude field, then overlays quiver arrows
    on a regular grid showing the direction and relative magnitude of the vector field.
    Uses plot_field_vectors for arrow interpolation.

    Parameters
    ----------
    ms : Microstructure
        Must have MicrofieldHeatConduction populated.
    analysis_dir : str or None
        Base analysis directory for saving.
    settings : dict or None
        Global settings.
    field_type : str
        "heat_flux" or "temp_gradient".
    subfolder : str
        Subfolder name within plots/.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure (closed after saving).
    """
    if settings is None:
        settings = {}
    figure_dpi = settings.get("figure_dpi", 150)
    cmap_name = settings.get("field_colormap", "jet")
    arrow_length = settings.get("arrow_length", 1.5)
    arrow_alpha = settings.get("arrow_alpha", 1.0)
    arrow_stem_width = settings.get("arrow_stem_width", 0.0012)
    field_fontsize = settings.get("field_fontsize", 12)

    # Determine plot style based on mesh type
    # "default" resolves to: rbf for Type 1 (conforming), smooth for Type 2 (non-conforming)
    mesh_type = getattr(ms, 'CurrentMeshType', 1)
    user_style = settings.get("field_plot_style", "default")
    if mesh_type == 1:
        plot_style = "rbf"
    elif user_style in ("default", "rbf"):
        plot_style = "smooth"
    else:
        plot_style = user_style

    mf = ms.MicrofieldHeatConduction

    # Select vector components based on field type
    if field_type == "heat_flux":
        v1 = mf[2]
        v2 = mf[3]
        field_label = r"$|q_{12}|$"
        field_unit = "W/m²"
        filename = "q_vector_2D"
        title_field = "Heat Flux Vectors (in-plane)"
    else:
        v1 = mf[6]
        v2 = mf[7]
        field_label = r"$|\nabla T_{12}|$"
        field_unit = "K/m"
        filename = "grad_T_vector_2D"
        title_field = "Temperature Gradient Vectors (in-plane)"

    # In-plane magnitude for background and title
    mag_2d = np.sqrt(v1**2 + v2**2)

    # Mesh data for background contour
    p3 = ms.ThreeNodeCoordinateList
    t3 = ms.ThreeNodeElementIndexList
    nElements = t3.shape[0]
    nCornerNodes = p3.shape[0]
    tri = Triangulation(p3[:, 0], p3[:, 1], t3)

    # Compute arrow grid using grain-aware RBF interpolation
    # comp < 5 for heat flux, comp >= 5 for temperature gradient
    from .plot_field_vectors import plot_field_vectors
    comp = 1 if field_type == "heat_flux" else 5
    xx_shifted, yy_shifted, vv1, vv2, maxNorm = plot_field_vectors(
        ms, comp, arrow_length=arrow_length)

    # ── Create figure ────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    cb_label = f'{field_label} ({field_unit})'

    # Ensure QP coordinates are available for RBF background
    if ms.Microfield is None or ms.Microfield[0] is None:
        ms.Microfield = [None] * max(len(ms.Microfield) if ms.Microfield else 0, 22)
        ms.Microfield[0] = mf[0]
        ms.Microfield[1] = mf[1]

    # Compute robust color limits for background: mean ± 2.5σ
    import matplotlib.colors as _mcolors
    bg_mean = np.mean(mag_2d)
    bg_std = np.std(mag_2d)
    bg_vmin = max(np.min(mag_2d), bg_mean - 2.5 * bg_std)
    bg_vmax = min(np.max(mag_2d), bg_mean + 2.5 * bg_std)
    bg_norm = _mcolors.Normalize(vmin=bg_vmin, vmax=bg_vmax)

    # Plot magnitude background
    if plot_style == "rbf":
        from .plot_stress_strain import _plot_rbf_field
        sm, vmin, vmax = _plot_rbf_field(ax, ms, mag_2d, p3, t3, 1.0, cmap=cmap_name)
        cb = fig.colorbar(sm, ax=ax, shrink=0.85)
        cb.set_label(cb_label, fontsize=field_fontsize)
        cb.ax.tick_params(labelsize=field_fontsize)
    elif plot_style == "smooth":
        nodal_values = _qp_to_nodal(mag_2d, nElements, nCornerNodes, t3)
        tpc = ax.tripcolor(tri, nodal_values, shading='gouraud', cmap=cmap_name, norm=bg_norm)
        cb = fig.colorbar(tpc, ax=ax, shrink=0.85)
        cb.set_label(cb_label, fontsize=field_fontsize)
        cb.ax.tick_params(labelsize=field_fontsize)
    else:
        elem_values = _qp_to_element_avg(mag_2d, nElements)
        tpc = ax.tripcolor(tri, facecolors=elem_values, cmap=cmap_name, norm=bg_norm)
        cb = fig.colorbar(tpc, ax=ax, shrink=0.85)
        cb.set_label(cb_label, fontsize=field_fontsize)
        cb.ax.tick_params(labelsize=field_fontsize)

    # Overlay arrows using quiver — explicit scaling, no auto-scale
    # scale=1 with scale_units='xy' means vv1,vv2 are in data coordinates
    # Arrow length controlled by arrow_length (via plot_field_vectors)
    # Shaft width controlled directly by arrow_stem_width
    aw = _ARROW_SETTINGS
    ax.quiver(xx_shifted.ravel(), yy_shifted.ravel(),
              vv1.ravel(), vv2.ravel(),
              color='k', linewidth=aw["linewidth"], alpha=arrow_alpha,
              scale=1, scale_units='xy', angles='xy',
              width=arrow_stem_width,
              headwidth=aw["headwidth"],
              headlength=aw["headlength"],
              headaxislength=aw["headaxislength"])

    # Overlay grain boundaries
    grains = getattr(ms, 'Grains', None)
    if grains is not None:
        for g in grains:
            if g is not None:
                g = np.asarray(g)
                if g.size > 0:
                    closed = np.vstack([g, g[0:1]])
                    ax.plot(closed[:, 0], closed[:, 1], '-',
                            color='gray', linewidth=0.3, alpha=0.3)

    # Formatting — clip to mesh domain
    ax.set_xlim(p3[:, 0].min(), p3[:, 0].max())
    ax.set_ylim(p3[:, 1].min(), p3[:, 1].max())
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title
    from .plot_stress_strain import _compute_field_mean
    ebsd_name = os.path.basename(getattr(ms, 'ebsd_file', ''))
    fmin_vec = np.min(mag_2d)
    fmax_vec = np.max(mag_2d)
    fmean_vec = _compute_field_mean(mag_2d, ms)
    title_line1 = f'{ebsd_name}: {title_field}'
    title_line2 = f'Min = {fmin_vec:.2f}, Mean = {fmean_vec:.2f}, Max = {fmax_vec:.2f} {field_unit}'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=12)

    # Save (microscale field plots are not displayed on screen — too many figures)
    if analysis_dir is not None:
        plot_dir = os.path.join(analysis_dir, subfolder)
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(os.path.join(plot_dir, f"{filename}.png"),
                    dpi=figure_dpi, bbox_inches='tight')

    plt.close(fig)

    return fig


def plot_all_heat_flux(ms, analysis_dir=None, settings=None, verbose=True):
    """
    Plot all heat flux and temperature gradient fields.

    Parameters
    ----------
    ms : Microstructure
        Must have MicrofieldHeatConduction populated.
    analysis_dir : str or None
        Base analysis directory for saving.
    settings : dict or None
        Global settings.
    verbose : bool
        Print progress to console.

    Returns
    -------
    None
    """
    if settings is None:
        settings = {}

    # Determine plot style based on mesh type
    # "default" resolves to: rbf for Type 1 (conforming), smooth for Type 2 (non-conforming)
    mesh_type = getattr(ms, 'CurrentMeshType', 1)
    user_style = settings.get("field_plot_style", "default")
    if mesh_type == 1:
        plot_style = "rbf"
    elif user_style in ("default", "rbf"):
        plot_style = "smooth"
    else:
        plot_style = user_style

    mf = ms.MicrofieldHeatConduction
    if mf is None or mf[0] is None:
        if verbose:
            print("  WARNING: No heat flux data to plot")
        return

    total = len(HEAT_FLUX_FIELDS) + len(TEMP_GRADIENT_FIELDS)
    if verbose:
        print(f"  Generating heat flux/temp gradient plots ({plot_style} style, {total} figures)...")

    # Plot heat flux component fields (q1, q2, q3, 3D magnitude)
    for field_index, field_name, field_label, field_unit, scale_factor in HEAT_FLUX_FIELDS:
        plot_heat_flux_field(ms, field_index, field_name, field_label, field_unit,
                              scale_factor=scale_factor, analysis_dir=analysis_dir,
                              settings=settings, subfolder="heat_flux")

    # Plot 2D in-plane magnitude: sqrt(q1^2 + q2^2)
    _, fn_2d, lbl_2d, unit_2d, sf_2d = HEAT_FLUX_2D_MAG
    mag_2d = np.sqrt(mf[2]**2 + mf[3]**2)
    plot_heat_flux_field(ms, None, fn_2d, lbl_2d, unit_2d,
                          scale_factor=sf_2d, analysis_dir=analysis_dir,
                          settings=settings, subfolder="heat_flux",
                          field_data_override=mag_2d)

    # Plot heat flux 2D vector overlay (in-plane magnitude + arrows)
    plot_heat_flux_vectors(ms, analysis_dir=analysis_dir, settings=settings,
                            field_type="heat_flux", subfolder="heat_flux")

    if verbose:
        print(f"  Saved {len(HEAT_FLUX_FIELDS) + 2} heat flux plots to plots/heat_flux/")

    # Plot temperature gradient component fields (dT1, dT2, dT3, 3D magnitude)
    for field_index, field_name, field_label, field_unit, scale_factor in TEMP_GRADIENT_FIELDS:
        plot_heat_flux_field(ms, field_index, field_name, field_label, field_unit,
                              scale_factor=scale_factor, analysis_dir=analysis_dir,
                              settings=settings, subfolder="temp_gradient")

    # Plot 2D in-plane temperature gradient magnitude: sqrt(dT1^2 + dT2^2)
    _, fn_2d_t, lbl_2d_t, unit_2d_t, sf_2d_t = TEMP_GRADIENT_2D_MAG
    grad_2d = np.sqrt(mf[6]**2 + mf[7]**2)
    plot_heat_flux_field(ms, None, fn_2d_t, lbl_2d_t, unit_2d_t,
                          scale_factor=sf_2d_t, analysis_dir=analysis_dir,
                          settings=settings, subfolder="temp_gradient",
                          field_data_override=grad_2d)

    # Plot temperature gradient 2D vector overlay (in-plane magnitude + arrows)
    plot_heat_flux_vectors(ms, analysis_dir=analysis_dir, settings=settings,
                            field_type="temp_gradient", subfolder="temp_gradient")

    if verbose:
        print(f"  Saved {len(TEMP_GRADIENT_FIELDS) + 2} temp gradient plots to plots/temp_gradient/")
