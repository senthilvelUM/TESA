"""
Generate a finite element mesh for the EBSD microstructure.

Supports three mesh types:
  Type 1: Conforming non-uniform mesh (non-uniform element size, conforms to grain boundaries)
  Type 2: Non-conforming hexagonal grid (uniform element size, structured hex grid)
  Type 3: Non-conforming rectangular grid (uniform element size, structured rectangular grid)
"""

import os
import numpy as np

from .mesh_conforming import mesh_conforming
from .mesh_nonconforming_hexgrid import mesh_nonconforming_hexgrid
from .mesh_nonconforming_rectgrid import mesh_nonconforming_rectgrid
from .simpvol import simpvol
from .simpqual import simpqual


def create_mesh(ms, job, run_dir=None, log_path=None, settings=None):
    """
    Generate a finite element mesh and return the updated Microstructure.

    Parameters
    ----------
    ms : Microstructure
        Must have EBSD data loaded (via load_ebsd).
    job : dict
        Job dictionary with mesh_type, advanced_mesh_params [K, R, g], target_elements, etc.
    run_dir : str or None
        Results directory for saving figures and mesh files.
    log_path : str or None
        Path to log.md file.
    settings : dict or None
        Global settings (verbose, show_figures, etc.).

    Returns
    -------
    ms : Microstructure
        Updated with mesh fields.
    """
    # Use defaults if settings not provided
    if settings is None:
        settings = {}
    if "verbose_console" not in settings:
        print("  WARNING: 'verbose_console' not in settings, using default 'medium'")
    vc = settings.get("verbose_console", "medium")
    console_on = vc in ("medium", "high")

    # Extract mesh parameters from job dictionary
    mesh_type   = job["mesh_type"]
    mesh_params = list(job.get("advanced_mesh_params", [0.01, 0.5, 0.3]))
    target_elements = job["target_elements"]

    # For Type 1: auto-derive l and bsep from target_elements.
    # The initial l/bsep are used for grain boundary smoothing; the actual mesh
    # size function is scaled to hit target_elements in mesh_conforming.py.
    if mesh_type == 1:
        coords = ms.OriginalDataCoordinateList
        domain_area = (coords[:, 0].max() - coords[:, 0].min()) * \
                      (coords[:, 1].max() - coords[:, 1].min())
        h_avg = np.sqrt(4 * domain_area / (target_elements * np.sqrt(3)))
        # Initial l and bsep proportional to expected element size near boundaries
        l_init = max(h_avg / 3.0, 0.1)
        bsep = l_init
        job["grain_boundary_resolution"] = bsep
        if console_on:
            print(f"  Auto mesh density: target_elements={target_elements}, h_avg={h_avg:.4f}")
            print(f"    Initial l = {l_init:.4f}, bsep = {bsep:.4f}")
        # Store [K, R, g, l] internally — user provides [K, R, g], l is auto-derived
        ms.MeshParameters = np.array([mesh_params[0], mesh_params[1], mesh_params[2], l_init])
    else:
        ms.MeshParameters = np.array(mesh_params)

    mesh_type_names = {1: "conforming non-uniform", 2: "non-conforming hexgrid", 3: "non-conforming rectangular"}
    if console_on:
        print(f"\n── Stage 2: Generate mesh (Type {mesh_type} — {mesh_type_names.get(mesh_type, 'unknown')}) ──")

    # Dispatch to the appropriate mesh generator based on mesh_type
    if mesh_type == 1:
        ms = mesh_conforming(ms, job, run_dir, log_path, settings)
    elif mesh_type == 2:
        ms = mesh_nonconforming_hexgrid(ms, job, run_dir, log_path, settings)
    elif mesh_type == 3:
        ms = mesh_nonconforming_rectgrid(ms, job, run_dir, log_path, settings)
    else:
        raise ValueError(f"Unknown mesh_type: {mesh_type}. Must be 1, 2, or 3.")

    # Write mesh statistics file to Mesh/ results folder
    if run_dir is not None:
        _write_mesh_statistics(ms, run_dir, mesh_type_names.get(mesh_type, 'unknown'), mesh_type)

    return ms


def _write_mesh_statistics(ms, run_dir, mesh_type_name, mesh_type):
    """
    Compute and write mesh statistics to mesh_statistics.txt in the Mesh/ results folder.

    Computes element sizes (h = sqrt(area)), element areas, edge lengths, and element
    quality (radius ratio q = 2r/R) from the 3-node triangular mesh.

    Parameters
    ----------
    ms             : Microstructure, must have ThreeNodeCoordinateList and ThreeNodeElementIndexList
    run_dir        : str, map results directory (Mesh/ subfolder is created here)
    mesh_type_name : str, human-readable mesh type name
    mesh_type      : int, mesh type (1, 2, or 3)
    """
    # Node coordinates and element connectivity (convert from 1-based to 0-based)
    p = ms.ThreeNodeCoordinateList
    t = ms.ThreeNodeElementIndexList        # already 0-based (SixNodeElementIndexList is 1-based)

    # Element areas via simpvol (signed; take absolute value)
    areas = np.abs(simpvol(p, t))           # (nelems,)

    # Element sizes: h = sqrt(area) — standard FEM characteristic element size
    sizes = np.sqrt(areas)

    # Edge lengths for all three edges of each element
    L12 = np.linalg.norm(p[t[:, 1]] - p[t[:, 0]], axis=1)
    L13 = np.linalg.norm(p[t[:, 2]] - p[t[:, 0]], axis=1)
    L23 = np.linalg.norm(p[t[:, 2]] - p[t[:, 1]], axis=1)
    all_edges = np.concatenate([L12, L13, L23])

    # Element quality: radius ratio q = 2r/R (0=degenerate, 1=equilateral)
    q = simpqual(p, t)

    # Counts
    n_elements   = t.shape[0]
    n_nodes_3    = p.shape[0]
    n_nodes_6    = ms.SixNodeCoordinateList.shape[0] if ms.SixNodeCoordinateList is not None else 0
    n_grains     = len(ms.GrainAngles) if ms.GrainAngles is not None else 0
    n_phases     = int(ms.NumberPhases) if ms.NumberPhases is not None else 0
    total_area   = np.sum(areas)

    # EBSD step size for physical unit reporting (None if not available)
    step = getattr(ms, 'EBSDStepSize', None)
    has_phys = step is not None and step > 0

    # Write file
    mesh_dir = os.path.join(run_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    diag_dir = os.path.join(mesh_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    fpath = os.path.join(diag_dir, "mesh_statistics.md")
    with open(fpath, 'w') as f:
        f.write(f"# Mesh Statistics\n\n")

        # ── Overview ──────────────────────────────────────────────────────────
        f.write(f"## Overview\n\n")
        f.write(f"| Property | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Mesh type | {mesh_type_name} (Type {mesh_type}) |\n")
        if has_phys:
            f.write(f"| EBSD step size | {step} phys. units/pixel |\n")
        f.write(f"\n")

        # ── Counts ────────────────────────────────────────────────────────────
        f.write(f"## Counts\n\n")
        f.write(f"| Property | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Total elements (3-node triangles) | {n_elements} |\n")
        f.write(f"| Total nodes (3-node mesh) | {n_nodes_3} |\n")
        f.write(f"| Total nodes (6-node mesh, FEA) | {n_nodes_6} |\n")
        f.write(f"| Number of grains | {n_grains} |\n")
        f.write(f"| Number of phases | {n_phases} |\n")
        f.write(f"\n")

        # ── Total mesh area ───────────────────────────────────────────────────
        f.write(f"## Total Mesh Area\n\n")
        f.write(f"| Units | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| pixels² | {total_area:.6e} |\n")
        if has_phys:
            f.write(f"| phys. units² | {total_area * step**2:.6e} |\n")
        f.write(f"\n")

        # ── Element size ──────────────────────────────────────────────────────
        f.write(f"## Element Size  h = sqrt(area)\n\n")
        f.write(f"| Statistic | pixels | {'phys. units' if has_phys else ''} |\n")
        f.write(f"|---|---|{'---|' if has_phys else ''}\n")
        f.write(f"| Mean | {np.mean(sizes):.6e} | {f'{np.mean(sizes)*step:.6e}' if has_phys else ''} |\n")
        f.write(f"| Min  | {np.min(sizes):.6e} | {f'{np.min(sizes)*step:.6e}' if has_phys else ''} |\n")
        f.write(f"| Max  | {np.max(sizes):.6e} | {f'{np.max(sizes)*step:.6e}' if has_phys else ''} |\n")
        f.write(f"| Std  | {np.std(sizes):.6e} | {f'{np.std(sizes)*step:.6e}' if has_phys else ''} |\n")
        f.write(f"\n")

        # ── Element area ──────────────────────────────────────────────────────
        f.write(f"## Element Area  A\n\n")
        f.write(f"| Statistic | pixels² | {'phys. units²' if has_phys else ''} |\n")
        f.write(f"|---|---|{'---|' if has_phys else ''}\n")
        f.write(f"| Mean | {np.mean(areas):.6e} | {f'{np.mean(areas)*step**2:.6e}' if has_phys else ''} |\n")
        f.write(f"| Min  | {np.min(areas):.6e} | {f'{np.min(areas)*step**2:.6e}' if has_phys else ''} |\n")
        f.write(f"| Max  | {np.max(areas):.6e} | {f'{np.max(areas)*step**2:.6e}' if has_phys else ''} |\n")
        f.write(f"\n")

        # ── Edge lengths ──────────────────────────────────────────────────────
        f.write(f"## Edge Lengths\n\n")
        f.write(f"| Statistic | pixels | {'phys. units' if has_phys else ''} |\n")
        f.write(f"|---|---|{'---|' if has_phys else ''}\n")
        f.write(f"| Mean | {np.mean(all_edges):.6e} | {f'{np.mean(all_edges)*step:.6e}' if has_phys else ''} |\n")
        f.write(f"| Min  | {np.min(all_edges):.6e} | {f'{np.min(all_edges)*step:.6e}' if has_phys else ''} |\n")
        f.write(f"| Max  | {np.max(all_edges):.6e} | {f'{np.max(all_edges)*step:.6e}' if has_phys else ''} |\n")
        f.write(f"\n")

        # ── Element quality ───────────────────────────────────────────────────
        f.write(f"## Element Quality  q = 2r/R\n\n")
        f.write(f"q = 1 for equilateral triangle, q → 0 for degenerate triangle.\n\n")
        f.write(f"| Statistic | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Mean | {np.mean(q):.6e} |\n")
        f.write(f"| Min  | {np.min(q):.6e} |\n")
        f.write(f"| Max  | {np.max(q):.6e} |\n")

        # ── Grain boundary statistics (Type 1 only) ──────────────────────────
        # ── Grain boundary statistics (Type 1 only) ──────────────────────────
        _gb_edges = getattr(ms, '_gb_edge_lengths', None)
        if _gb_edges is not None and len(_gb_edges) > 0 and _gb_edges[0] > 0:
            _gb_nn = getattr(ms, '_gb_n_nodes', 0)
            _gb_ne = getattr(ms, '_gb_n_edges', 0)
            f.write(f"\n## Grain Boundary Statistics (Interior)\n\n")
            f.write(f"| Property | Value |\n")
            f.write(f"|---|---|\n")
            f.write(f"| GB nodes | {_gb_nn} |\n")
            f.write(f"| GB edges | {_gb_ne} |\n")
            f.write(f"\n")
            f.write(f"| Statistic | pixels | {'phys. units' if has_phys else ''} |\n")
            f.write(f"|---|---|{'---|' if has_phys else ''}\n")
            f.write(f"| Mean | {np.mean(_gb_edges):.6e} | {f'{np.mean(_gb_edges)*step:.6e}' if has_phys else ''} |\n")
            f.write(f"| Min  | {np.min(_gb_edges):.6e} | {f'{np.min(_gb_edges)*step:.6e}' if has_phys else ''} |\n")
            f.write(f"| Max  | {np.max(_gb_edges):.6e} | {f'{np.max(_gb_edges)*step:.6e}' if has_phys else ''} |\n")
            _gb_ratio = np.max(_gb_edges) / np.min(_gb_edges) if np.min(_gb_edges) > 0 else float('inf')
            f.write(f"| Max/Min | {_gb_ratio:.1f} | |\n")

        # ── Domain boundary statistics (Type 1 only) ─────────────────────────
        _dom_edges = getattr(ms, '_dom_edge_lengths', None)
        if _dom_edges is not None and len(_dom_edges) > 0 and _dom_edges[0] > 0:
            _dom_nn = getattr(ms, '_dom_n_nodes', 0)
            _dom_ne = getattr(ms, '_dom_n_edges', 0)
            f.write(f"\n## Domain Boundary Statistics\n\n")
            f.write(f"| Property | Value |\n")
            f.write(f"|---|---|\n")
            f.write(f"| DB nodes | {_dom_nn} |\n")
            f.write(f"| DB edges | {_dom_ne} |\n")
            f.write(f"\n")
            f.write(f"| Statistic | pixels | {'phys. units' if has_phys else ''} |\n")
            f.write(f"|---|---|{'---|' if has_phys else ''}\n")
            f.write(f"| Mean | {np.mean(_dom_edges):.6e} | {f'{np.mean(_dom_edges)*step:.6e}' if has_phys else ''} |\n")
            f.write(f"| Min  | {np.min(_dom_edges):.6e} | {f'{np.min(_dom_edges)*step:.6e}' if has_phys else ''} |\n")
            f.write(f"| Max  | {np.max(_dom_edges):.6e} | {f'{np.max(_dom_edges)*step:.6e}' if has_phys else ''} |\n")
            _db_ratio = np.max(_dom_edges) / np.min(_dom_edges) if np.min(_dom_edges) > 0 else float('inf')
            f.write(f"| Max/Min | {_db_ratio:.1f} | |\n")
