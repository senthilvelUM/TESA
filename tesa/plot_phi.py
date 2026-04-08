"""
Plot thermal conductivity characteristic temperature functions (phi).

Visualizes the characteristic temperature field phi_j over the microstructure
as a smooth continuous contour plot, where phi_j corresponds to the response
to a unit temperature gradient in direction j.

For 2D problems there are 2 phi functions (j=1,2).

Plots are saved to a "phi" subfolder within the analysis directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def plot_phi(ms, j, analysis_dir=None, settings=None):
    """
    Plot one phi characteristic temperature function.

    Parameters
    ----------
    ms : Microstructure
        Must have thermalConductivityCharacteristicFunctions,
        ThreeNodeCoordinateList, ThreeNodeElementIndexList populated.
    j : int
        Temperature gradient direction (1=∂T/∂x1, 2=∂T/∂x2).
    analysis_dir : str or None
        Directory to save the figure (saved to "phi" subfolder).
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
    figure_fontsize = settings.get("figure_fontsize", 12)
    figure_title_fontsize = settings.get("figure_title_fontsize", 14)

    # Extract nodal values of phi_j (all 6-node mesh nodes)
    # phi shape: (nNodes, 2) for 2D — column 0 for x1, column 1 for x2
    phi = ms.thermalConductivityCharacteristicFunctions  # (nNodes, 2)
    phi_j = phi[:, j - 1]  # (nNodes,)

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
    tpc = ax.tripcolor(tri, phi_j, shading='gouraud', cmap=cmap_name)
    cb = fig.colorbar(tpc, ax=ax, shrink=0.85)
    cb.ax.set_title(f'$\\phi_{{{j}}}$', fontsize=figure_fontsize, pad=figure_fontsize * 0.6)
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
    fmin = np.min(phi_j)
    fmax = np.max(phi_j)
    title_line1 = f'{ebsd_name}: $\\phi_{{{j}}}$ — Temperature field for $\\partial T / \\partial x_{{{j}}} = 1$'
    title_line2 = f'Min = {fmin:.2f}, Max = {fmax:.2f}'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=figure_title_fontsize)

    # Save to "phi" subfolder (no figure numbering)
    if analysis_dir:
        phi_dir = os.path.join(analysis_dir, "phi")
        os.makedirs(phi_dir, exist_ok=True)
        filename = f"phi_{j}.png"
        filepath = os.path.join(phi_dir, filename)
        fig.savefig(filepath, dpi=figure_dpi, bbox_inches='tight')

    plt.close(fig)
    return fig


def plot_all_phi(ms, analysis_dir=None, settings=None, verbose=True):
    """
    Plot all phi characteristic temperature functions (2 for 2D).

    Saves to "phi" subfolder within analysis_dir. No figure numbering.
    Does not display on screen.

    Parameters
    ----------
    ms : Microstructure
        Must have thermalConductivityCharacteristicFunctions populated.
    analysis_dir : str or None
        Directory to save figures.
    settings : dict or None
        Global settings.
    verbose : bool
        Print progress to console.

    Returns
    -------
    fig_count : int
        Number of figures generated (2 for 2D).
    """
    if verbose:
        print("  Generating phi characteristic function plots...")

    # Determine number of phi functions from array shape
    phi = ms.thermalConductivityCharacteristicFunctions
    n_phi = phi.shape[1]  # 2 for 2D, 3 for 3D

    count = 0
    for j in range(1, n_phi + 1):
        plot_phi(ms, j, analysis_dir=analysis_dir, settings=settings)
        count += 1

    if verbose:
        phi_dir = os.path.join(analysis_dir, "phi") if analysis_dir else "N/A"
        print(f"  Saved {count} phi plots to phi/")

    return count
