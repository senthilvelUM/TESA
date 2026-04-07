"""
Plot thermal characteristic displacement function (psi).

Visualizes the displacement component u_i of the characteristic function
psi over the microstructure as a smooth continuous contour plot,
where psi is the response to a unit temperature change ΔT = 1.

Plots are saved to a "psi" subfolder within the analysis directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def plot_psi(ms, i, analysis_dir=None, settings=None):
    """
    Plot one psi characteristic function component.

    Parameters
    ----------
    ms : Microstructure
        Must have psiCharacteristicFunctions, SixNodeCoordinateList,
        ThreeNodeCoordinateList, ThreeNodeElementIndexList populated.
    i : int
        Displacement component (1=u1, 2=u2, 3=u3).
    analysis_dir : str or None
        Directory to save the figure (saved to "psi" subfolder).
    settings : dict or None
        Global settings (for figure_dpi).

    Returns
    -------
    fig : matplotlib Figure
        The generated figure.
    """
    if settings is None:
        settings = {}
    figure_dpi = settings.get("figure_dpi", 150)
    field_fontsize = settings.get("field_fontsize", 12)

    # Extract nodal values of u_i from psi (all 6-node mesh nodes)
    psi = ms.psiCharacteristicFunctions  # (3*nNodes,)
    u_i_all = psi[i - 1::3]              # every 3rd DOF → (nNodes,)

    # Use 6-node mesh subdivided into 4 sub-elements per element
    p6 = ms.SixNodeCoordinateList
    t6 = ms.SixNodeElementIndexList - 1  # convert 1-based to 0-based
    p3 = ms.ThreeNodeCoordinateList  # for domain limits

    # Split each 6-node triangle into 4 sub-triangles
    t_sub = np.vstack([
        t6[:, [0, 3, 5]],
        t6[:, [3, 1, 4]],
        t6[:, [5, 4, 2]],
        t6[:, [3, 4, 5]],
    ])

    # Create triangulation using all 6-node coordinates and sub-element connectivity
    tri = Triangulation(p6[:, 0], p6[:, 1], t_sub)

    # EBSD file name for title
    ebsd_name = os.path.basename(getattr(ms, 'ebsd_file', ''))

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Get colormap from settings
    cmap_name = settings.get("field_colormap", "jet") if settings else "jet"

    # Continuous (smooth) filled contour plot using tripcolor on sub-elements
    tpc = ax.tripcolor(tri, u_i_all, shading='gouraud', cmap=cmap_name)
    cb = fig.colorbar(tpc, ax=ax, shrink=0.85)
    cb.set_label(f'$\\psi_{{{i}}}$', fontsize=field_fontsize)
    cb.ax.tick_params(labelsize=field_fontsize)

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
    fmin = np.min(u_i_all)
    fmax = np.max(u_i_all)
    title_line1 = f'{ebsd_name}: $\\psi_{{{i}}}$ — $\\Delta T$ displacement $u_{{{i}}}$'
    title_line2 = f'Min = {fmin:.2f}, Max = {fmax:.2f}'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=12)

    # Save to "psi" subfolder (no figure numbering)
    if analysis_dir:
        psi_dir = os.path.join(analysis_dir, "psi")
        os.makedirs(psi_dir, exist_ok=True)
        filename = f"psi_u{i}.png"
        filepath = os.path.join(psi_dir, filename)
        fig.savefig(filepath, dpi=figure_dpi, bbox_inches='tight')

    plt.close(fig)
    return fig


def plot_all_psi(ms, analysis_dir=None, settings=None, verbose=True):
    """
    Plot all 3 psi characteristic function components (3 displacements).

    Saves to "psi" subfolder within analysis_dir. No figure numbering.
    Does not display on screen.

    Parameters
    ----------
    ms : Microstructure
        Must have psiCharacteristicFunctions populated.
    analysis_dir : str or None
        Directory to save figures.
    settings : dict or None
        Global settings.
    verbose : bool
        Print progress to console.

    Returns
    -------
    fig_count : int
        Number of figures generated (3).
    """
    if verbose:
        print("  Generating psi characteristic function plots (3 figures)...")

    count = 0
    for i_disp in [1, 2, 3]:
        plot_psi(ms, i_disp, analysis_dir=analysis_dir, settings=settings)
        count += 1

    if verbose:
        psi_dir = os.path.join(analysis_dir, "psi") if analysis_dir else "N/A"
        print(f"  Saved {count} psi plots to psi/")

    return count
