"""
Plot elastic characteristic displacement functions (chi).

Visualizes the displacement component u_i of the characteristic function
chi^{kl} over the microstructure as a smooth continuous contour plot,
where (k,l) specifies the strain component and i specifies the
displacement direction.

Plots are saved to a "chi" subfolder within the analysis directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

from .contract import contract


def plot_chi(ms, k, l, i, analysis_dir=None, settings=None):
    """
    Plot one chi characteristic function component.

    Parameters
    ----------
    ms : Microstructure
        Must have chiCharacteristicFunctions, SixNodeCoordinateList,
        ThreeNodeCoordinateList, ThreeNodeElementIndexList populated.
    k, l : int
        Strain component indices (1-based). E.g., k=1, l=1 for epsilon_11.
    i : int
        Displacement component (1=u1, 2=u2, 3=u3).
    analysis_dir : str or None
        Directory to save the figure (saved to "chi" subfolder).
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

    # Map (k,l) to chi column using Voigt contraction
    voigt_idx = contract(k, l)  # 1-based (1-6)
    col = voigt_idx - 1          # 0-based column into chi array

    # Extract nodal values of u_i from chi (all 6-node mesh nodes)
    chi_col = ms.chiCharacteristicFunctions[:, col]  # (3*nNodes,)
    u_i_all = chi_col[i - 1::3]                       # every 3rd DOF → (nNodes,)

    # Use 6-node mesh subdivided into 4 sub-elements per element
    # This captures the quadratic FE variation using all 6 nodal values
    p6 = ms.SixNodeCoordinateList
    t6 = ms.SixNodeElementIndexList - 1  # convert 1-based to 0-based
    p3 = ms.ThreeNodeCoordinateList  # for domain limits

    # Split each 6-node triangle into 4 sub-triangles:
    #   t6 columns: [n1, n2, n3, n4, n5, n6]
    #   sub1: (n1, n4, n6)   sub2: (n4, n2, n5)
    #   sub3: (n6, n5, n3)   sub4: (n4, n5, n6)
    t_sub = np.vstack([
        t6[:, [0, 3, 5]],
        t6[:, [3, 1, 4]],
        t6[:, [5, 4, 2]],
        t6[:, [3, 4, 5]],
    ])

    # Create triangulation using all 6-node coordinates and sub-element connectivity
    tri = Triangulation(p6[:, 0], p6[:, 1], t_sub)

    # Strain component label
    strain_labels = {1: '11', 2: '22', 3: '33', 4: '23', 5: '13', 6: '12'}
    strain_label = strain_labels[voigt_idx]

    # EBSD file name for title
    ebsd_name = os.path.basename(getattr(ms, 'ebsd_file', ''))

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Get colormap from settings
    cmap_name = settings.get("field_colormap", "jet") if settings else "jet"

    # Continuous (smooth) filled contour plot using tripcolor on sub-elements
    tpc = ax.tripcolor(tri, u_i_all, shading='gouraud', cmap=cmap_name)
    cb = fig.colorbar(tpc, ax=ax, shrink=0.85)
    cb.set_label(f'$\\chi^{{{strain_label}}}_{{{i}}}$', fontsize=field_fontsize)
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
    title_line1 = f'{ebsd_name}: $\\chi^{{{strain_label}}}_{{{i}}}$ — $\\varepsilon_{{{strain_label}}}$ displacement $u_{{{i}}}$'
    title_line2 = f'Min = {fmin:.2f}, Max = {fmax:.2f}'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=12)

    # Save to "chi" subfolder (no figure numbering)
    if analysis_dir:
        chi_dir = os.path.join(analysis_dir, "chi")
        os.makedirs(chi_dir, exist_ok=True)
        filename = f"chi_{strain_label}_u{i}.png"
        filepath = os.path.join(chi_dir, filename)
        fig.savefig(filepath, dpi=figure_dpi, bbox_inches='tight')

    plt.close(fig)
    return fig


def plot_all_chi(ms, analysis_dir=None, settings=None, verbose=True):
    """
    Plot all 18 chi characteristic function components (6 strains × 3 displacements).

    Saves to "chi" subfolder within analysis_dir. No figure numbering.
    Does not display on screen.

    Parameters
    ----------
    ms : Microstructure
        Must have chiCharacteristicFunctions populated.
    analysis_dir : str or None
        Directory to save figures.
    settings : dict or None
        Global settings.
    verbose : bool
        Print progress to console.

    Returns
    -------
    fig_count : int
        Number of figures generated (18).
    """
    if verbose:
        print("  Generating chi characteristic function plots (18 figures)...")

    # Strain component pairs in order: (1,1), (2,2), (3,3), (2,3), (1,3), (1,2)
    strain_pairs = [(1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2)]

    count = 0
    for k, l in strain_pairs:
        for i_disp in [1, 2, 3]:
            plot_chi(ms, k, l, i_disp, analysis_dir=analysis_dir, settings=settings)
            count += 1

    if verbose:
        chi_dir = os.path.join(analysis_dir, "chi") if analysis_dir else "N/A"
        print(f"  Saved {count} chi plots to chi/")

    return count
