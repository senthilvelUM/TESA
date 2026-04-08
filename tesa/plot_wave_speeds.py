"""
Plot seismic wave speed fields as Lambert azimuthal equal-area projections
and 3D sphere surface plots.

Generates plots for each homogenization method (AEH, Voigt, Reuss, Hill, GeoMean)
and each wave speed field (VP, VS1, VS2, VSH, VSV, AVS, DTS, DTP).

Two plot types per field:
  - Lambert: 2D equal-area disk projection of the hemisphere
  - Sphere: 3D colored surface of the full sphere
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .lambert_azimuthal_projection import lambert_azimuthal_projection
from .mirrorsphere import mirrorsphere


# ── Field definitions ─────────────────────────────────────────────────────
# Internal wave speeds are in m/s; plots display km/s via scale factor.
# (VS_dict_key, filename, latex_label, display_unit, anisotropy_key, scale_factor)
# scale_factor converts internal units to display units (e.g., 1e-3 for m/s → km/s)
WAVE_SPEED_FIELDS = [
    ("VP",  "VP",  r"$V_P$",          "km/s", "AVP",   1e-3),
    ("VS1", "VS1", r"$V_{S1}$",       "km/s", "AVS1",  1e-3),
    ("VS2", "VS2", r"$V_{S2}$",       "km/s", "AVS2",  1e-3),
    ("VSH", "VSH", r"$V_{SH}$",       "km/s", "AVSH",  1e-3),
    ("VSV", "VSV", r"$V_{SV}$",       "km/s", "AVSV",  1e-3),
    ("AVS", "AVS", r"$AV_S$",         "%",    "MaxAVS", 1.0),
    ("DTS", "DTS", r"$\Delta t_S$",   "s/km", None,    1e3),
    ("DTP", "DTP", r"$\Delta t_P$",   "s/km", None,    1e3),
]


def plot_wave_speed_lambert(XC, YC, ZC, field, field_label, field_unit,
                             aniso_pct, method_name, ebsd_name, settings,
                             save_path):
    """
    Plot a single wave speed field as a Lambert azimuthal equal-area 2D disk.

    Parameters
    ----------
    XC, YC, ZC : (n/2+1, n+1) arrays
        Hemisphere coordinates from get_wave_speeds().
    field : (n/2+1, n+1) array
        Wave speed values at each direction.
    field_label : str
        LaTeX label for the field.
    field_unit : str
        Unit string for colorbar.
    aniso_pct : float or None
        Anisotropy percentage to display in title.
    method_name : str
        Homogenization method name (e.g., "AEH").
    ebsd_name : str
        EBSD filename for title.
    settings : dict
        Global settings (figure_dpi, field_colormap).
    save_path : str
        Full path to save the PNG file.

    Returns
    -------
    None
    """
    figure_dpi = settings.get("figure_dpi", 150)
    cmap_name = settings.get("field_colormap", "turbo")
    figure_fontsize = settings.get("figure_fontsize", 12)
    figure_title_fontsize = settings.get("figure_title_fontsize", 14)

    # Project lower hemisphere to 2D Lambert equal-area disk
    # z ranges from -1 (south pole → origin) to 0 (equator → boundary at r=sqrt(2))
    X, Y = lambert_azimuthal_projection(XC, YC, ZC)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # Plot using pcolormesh on the structured Lambert-projected grid
    # Smooth interpolated shading on the structured Lambert-projected grid
    pcm = ax.pcolormesh(X, Y, field, shading='gouraud', cmap=cmap_name)

    # Clip plot to the projection boundary circle (radius sqrt(2))
    # This masks any data that extends beyond the disk edge
    from matplotlib.patches import Circle as _Circle
    r = np.sqrt(2)
    clip_circle = _Circle((0, 0), r, transform=ax.transData)
    pcm.set_clip_path(clip_circle)

    # Draw boundary circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=1.5)

    # Add axis direction labels with tick marks
    tick_len = 0.08
    ax.plot([r, r + tick_len], [0, 0], 'k-', linewidth=1.5)
    ax.plot([0, 0], [r, r + tick_len], 'k-', linewidth=1.5)
    offset = r + 0.15
    ax.text(offset + tick_len, 0, 'x', ha='left', va='center', fontsize=figure_fontsize, fontweight='bold')
    ax.text(0, offset + tick_len, 'y', ha='center', va='bottom', fontsize=figure_fontsize, fontweight='bold')

    # Colorbar with unit label
    cb = fig.colorbar(pcm, ax=ax, shrink=0.75, pad=0.05)
    cb_title = f'{field_label}\n({field_unit})' if field_unit else field_label
    cb.ax.set_title(cb_title, fontsize=figure_fontsize, pad=figure_fontsize * 0.6)
    cb.ax.tick_params(labelsize=figure_fontsize)

    # Formatting
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    margin = 0.3
    top_margin = 0.55
    ax.set_xlim(-r - margin, r + margin)
    ax.set_ylim(-r - margin, r + top_margin)

    # Title with min/max and anisotropy
    fmin = np.nanmin(field)
    fmean = np.nanmean(field)
    fmax = np.nanmax(field)
    aniso_str = f" ({aniso_pct:.2f}%)" if aniso_pct is not None else ""
    title_line1 = f'{ebsd_name}: {method_name} {field_label}{aniso_str}'
    title_line2 = f'Min = {fmin:.4f}, Mean = {fmean:.4f}, Max = {fmax:.4f} {field_unit}'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=figure_title_fontsize, pad=figure_title_fontsize * 2.5)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=figure_dpi, bbox_inches='tight')

    plt.close(fig)


def plot_wave_speed_sphere(XC, YC, ZC, field, field_label, field_unit,
                            aniso_pct, method_name, ebsd_name, settings,
                            save_path):
    """
    Plot a single wave speed field as a 3D colored sphere surface.

    Parameters
    ----------
    XC, YC, ZC : ndarray, shape (n/2+1, n+1)
        Hemisphere coordinates from get_wave_speeds().
    field : ndarray, shape (n/2+1, n+1)
        Wave speed values at each direction.
    field_label : str
        LaTeX label for the field.
    field_unit : str
        Unit string for colorbar.
    aniso_pct : float or None
        Anisotropy percentage to display in title.
    method_name : str
        Homogenization method name (e.g., "AEH").
    ebsd_name : str
        EBSD filename for title.
    settings : dict
        Global settings (figure_dpi, field_colormap, show_figures, figure_pause,
        wave_speed_sphere_elev, wave_speed_sphere_azim).
    save_path : str
        Full path to save the PNG file.

    Returns
    -------
    None
    """
    figure_dpi = settings.get("figure_dpi", 150)
    cmap_name = settings.get("field_colormap", "turbo")
    figure_fontsize = settings.get("figure_fontsize", 12)
    figure_title_fontsize = settings.get("figure_title_fontsize", 14)
    # Wave speed plots are never displayed on screen (too many figures);
    # they are always saved to files only.

    # Mirror hemisphere to full sphere
    xn, yn, zn, dn = mirrorsphere(XC, YC, ZC, field)

    # Create 3D figure — use a wider subplot region so the sphere fills more space
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([-0.15, -0.05, 1.0, 1.0], projection='3d')

    # Normalize field values for colormapping
    cmap = plt.cm.get_cmap(cmap_name)
    vmin = np.nanmin(dn)
    vmax = np.nanmax(dn)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Disable automatic z-ordering so axis lines render on top of the sphere surface
    # (matplotlib's painter algorithm otherwise hides lines behind surface polygons)
    ax.computed_zorder = False

    # Plot the colored sphere surface using all grid points for smooth rendering
    ax.plot_surface(xn, yn, zn,
                    facecolors=cmap(norm(dn)),
                    rstride=1, cstride=1,
                    shade=False, antialiased=False,
                    zorder=1)

    # Set viewing angle (user-configurable, default 30 deg elevation and azimuth)
    # User azimuth is measured from the +x axis in the x-y plane.
    # Matplotlib measures azimuth from +y (CW), so convert: mpl_azim = 90 - user_azim.
    elev = settings.get("wave_speed_sphere_elev", 30)
    azim = settings.get("wave_speed_sphere_azim", 30)
    ax.view_init(elev=elev, azim=90 - azim)

    # Hide default axes, ticks, and panes — replace with custom axis lines
    ax.set_axis_off()

    # Draw custom axis lines extending beyond the sphere surface
    # Start just outside the sphere (r=1.02) so lines aren't occluded by the surface
    r0 = 1.02  # start just outside sphere surface
    L = 1.3    # axis line endpoint (sphere radius = 1)
    ax.plot([r0, L], [0, 0], [0, 0], 'k-', linewidth=1.0, zorder=10)
    ax.plot([0, 0], [r0, L], [0, 0], 'k-', linewidth=1.0, zorder=10)
    ax.plot([0, 0], [0, 0], [r0, L], 'k-', linewidth=1.0, zorder=10)

    # Place axis labels at the tips
    label_offset = L + 0.15
    ax.text(label_offset, 0, 0, 'x', fontsize=figure_fontsize, fontweight='bold', ha='center', va='center', zorder=10)
    ax.text(0, label_offset, 0, 'y', fontsize=figure_fontsize, fontweight='bold', ha='center', va='center', zorder=10)
    ax.text(0, 0, label_offset, 'z', fontsize=figure_fontsize, fontweight='bold', ha='center', va='center', zorder=10)

    # Set axis limits tight to the sphere so it fills the frame
    ax.set_xlim([-0.85, 0.85])
    ax.set_ylim([-0.85, 0.85])
    ax.set_zlim([-0.85, 0.85])

    # Add colorbar via ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cb_title = f'{field_label}\n({field_unit})' if field_unit else field_label
    cb.ax.set_title(cb_title, fontsize=figure_fontsize, pad=figure_fontsize * 0.6)
    cb.ax.tick_params(labelsize=figure_fontsize)

    # Title
    fmin = np.nanmin(field)
    fmean = np.nanmean(field)
    fmax = np.nanmax(field)
    aniso_str = f" ({aniso_pct:.2f}%)" if aniso_pct is not None else ""
    title_line1 = f'{ebsd_name}: {method_name} {field_label}{aniso_str}'
    title_line2 = f'Min = {fmin:.4f}, Mean = {fmean:.4f}, Max = {fmax:.4f} {field_unit}'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=figure_title_fontsize)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=figure_dpi, bbox_inches='tight')

    plt.close(fig)


def plot_wave_speed_field(ms, method_name, ws_data, field_key, field_name,
                           field_label, field_unit, aniso_key, scale_factor,
                           wave_speed_plots_dir, settings):
    """
    Plot both Lambert and sphere views for a single field and method.

    Parameters
    ----------
    ms : Microstructure
    method_name : str
        e.g., "AEH", "Voigt"
    ws_data : dict
        Wave speed data for this method (from ms.WaveSpeedResults[method_name]).
    field_key : str
        Key into VS dict (e.g., "VP").
    field_name : str
        Filename stem (e.g., "VP").
    field_label : str
        LaTeX label.
    field_unit : str
        Display unit string (after scaling).
    aniso_key : str or None
        Key into ws_data for anisotropy percentage.
    scale_factor : float
        Multiplier to convert internal units to display units
        (e.g., 1e-3 for m/s → km/s).
    wave_speed_plots_dir : str
        Path to 'wave_speed_plots' directory.
    settings : dict
        Global settings.

    Returns
    -------
    count : int
        Number of figures generated (1 or 2 depending on plot_type setting).
    """
    # Extract wave speed data and convert to display units
    VS = ws_data["VS"]
    XC = VS["XC"]
    YC = VS["YC"]
    ZC = VS["ZC"]
    field = VS[field_key] * scale_factor

    # Get anisotropy percentage if available
    aniso_pct = ws_data.get(aniso_key) if aniso_key else None

    # EBSD filename for title
    ebsd_name = os.path.basename(getattr(ms, 'ebsd_file', ''))

    # Save directory: wave_speeds/{method}/
    save_dir = os.path.join(wave_speed_plots_dir, method_name)

    # Determine which plot types to generate
    plot_type = settings.get("wave_speed_plot_type", "both")
    count = 0

    # Lambert equal-area projection (2D disk)
    if plot_type in ("lambert", "both"):
        plot_wave_speed_lambert(
            XC, YC, ZC, field, field_label, field_unit,
            aniso_pct, method_name, ebsd_name, settings,
            os.path.join(save_dir, f"{field_name}_lambert.png"))
        count += 1

    # 3D sphere surface
    if plot_type in ("sphere", "both"):
        plot_wave_speed_sphere(
            XC, YC, ZC, field, field_label, field_unit,
            aniso_pct, method_name, ebsd_name, settings,
            os.path.join(save_dir, f"{field_name}_sphere.png"))
        count += 1

    return count


def plot_all_wave_speeds(ms, wave_speed_plots_dir=None, settings=None, verbose=True):
    """
    Plot all wave speed fields for all homogenization methods.

    Generates Lambert (2D) and sphere (3D) plots for each of the 8 wave speed
    fields and each method. Total: 5 methods x 8 fields x 2 types = 80 files.

    Parameters
    ----------
    ms : Microstructure
        Must have WaveSpeedResults populated by run_analysis().
    wave_speed_plots_dir : str or None
        Path to 'wave_speed_plots' directory.
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

    # Check that wave speed data exists
    ws_results = getattr(ms, 'WaveSpeedResults', None)
    if ws_results is None or len(ws_results) == 0:
        if verbose:
            print("  WARNING: No wave speed data to plot")
        return

    if wave_speed_plots_dir is None:
        if verbose:
            print("  WARNING: No wave speeds directory specified — skipping wave speed plots")
        return

    # Determine which fields to plot
    ws_fields_setting = settings.get("wave_speed_plots", "all")
    if ws_fields_setting == "all":
        fields_to_plot = WAVE_SPEED_FIELDS
    else:
        # Filter to only requested fields
        requested = set(ws_fields_setting) if isinstance(ws_fields_setting, list) else {ws_fields_setting}
        fields_to_plot = [f for f in WAVE_SPEED_FIELDS if f[0] in requested]
        if not fields_to_plot:
            if verbose:
                print(f"  WARNING: No valid wave speed fields in {ws_fields_setting}")
            return

    # Determine plot type multiplier
    plot_type = settings.get("wave_speed_plot_type", "both")
    plots_per_field = 2 if plot_type == "both" else 1

    n_methods = len(ws_results)
    n_fields = len(fields_to_plot)
    total = n_methods * n_fields * plots_per_field
    if verbose:
        field_names = [f[0] for f in fields_to_plot]
        print(f"  Generating wave speed plots ({n_methods} methods × {n_fields} fields × "
              f"{plots_per_field} types = {total} figures)...")
        print(f"    Fields: {', '.join(field_names)} | Plot type: {plot_type}")

    # Loop over methods and fields
    count = 0
    for method_name, ws_data in ws_results.items():
        method_count = 0
        for field_key, field_name, field_label, field_unit, aniso_key, scale_factor in fields_to_plot:
            n = plot_wave_speed_field(
                ms, method_name, ws_data, field_key, field_name,
                field_label, field_unit, aniso_key, scale_factor,
                wave_speed_plots_dir, settings)
            method_count += n
            count += n

        if verbose:
            print(f"    {method_name}: {method_count} plots saved to wave_speed_plots/{method_name}/")

    if verbose:
        print(f"  Saved {count} wave speed plots total")


def plot_all_phase_wave_speeds(ms, wave_speed_plots_dir=None, settings=None, verbose=True):
    """
    Plot wave speed fields for all crystal phases (single-crystal).

    Generates Lambert (2D) and sphere (3D) plots for each phase using
    its raw stiffness matrix (crystal coordinates aligned with global axes).

    Parameters
    ----------
    ms : Microstructure
        Must have PhaseWaveSpeedResults populated by run_analysis().
    wave_speed_plots_dir : str or None
        Path to 'wave_speed_plots' directory.
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

    # Check that per-phase wave speed data exists
    phase_results = getattr(ms, 'PhaseWaveSpeedResults', None)
    if not phase_results:
        if verbose:
            print("  No per-phase wave speed data to plot")
        return

    if wave_speed_plots_dir is None:
        if verbose:
            print("  WARNING: No wave speeds directory specified — skipping per-phase wave speed plots")
        return

    # Determine which fields to plot
    ws_fields_setting = settings.get("wave_speed_plots", "all")
    if ws_fields_setting == "all":
        fields_to_plot = WAVE_SPEED_FIELDS
    else:
        requested = set(ws_fields_setting) if isinstance(ws_fields_setting, list) else {ws_fields_setting}
        fields_to_plot = [f for f in WAVE_SPEED_FIELDS if f[0] in requested]
        if not fields_to_plot:
            if verbose:
                print(f"  WARNING: No valid wave speed fields in {ws_fields_setting}")
            return

    # Determine plot type multiplier
    plot_type = settings.get("wave_speed_plot_type", "both")
    plots_per_field = 2 if plot_type == "both" else 1

    n_phases = len(phase_results)
    n_fields = len(fields_to_plot)
    total = n_phases * n_fields * plots_per_field
    if verbose:
        field_names = [f[0] for f in fields_to_plot]
        print(f"  Generating per-phase wave speed plots ({n_phases} phases × {n_fields} fields × "
              f"{plots_per_field} types = {total} figures)...")

    # Loop over phases and fields
    count = 0
    for phase_key, ws_data in phase_results.items():
        phase_count = 0
        for field_key, field_name, field_label, field_unit, aniso_key, scale_factor in fields_to_plot:
            n = plot_wave_speed_field(
                ms, phase_key, ws_data, field_key, field_name,
                field_label, field_unit, aniso_key, scale_factor,
                wave_speed_plots_dir, settings)
            phase_count += n
            count += n

        if verbose:
            print(f"    {phase_key}: {phase_count} plots saved to wave_speed_plots/{phase_key}/")

    if verbose:
        print(f"  Saved {count} per-phase wave speed plots total")
