"""
Shared pipeline function for running a single TESA job.

Used by run_tesa.py for standard analyses and can be imported by
future parametric study scripts.
"""

import os
import time
import numpy as np

from .setup_job import setup_job
from .load_ebsd import load_ebsd
from .create_mesh import create_mesh
from .save_load_mesh import save_mesh, load_mesh
from .run_analysis import run_analysis
from .post_process import post_process
from .Microstructure import Microstructure
from .read_properties import read_properties
from .compute_ebsd_correction_matrix import compute_ebsd_correction_matrix


def _refresh_phase_properties(ms, job, log_path, settings):
    """
    Re-read phase property files and update ms after loading a saved mesh.

    This ensures property changes are picked up without regenerating the mesh.
    Also recomputes EBSD correction matrices, homogenization method, and
    homogenized density from the fresh property values.

    Parameters
    ----------
    ms : Microstructure
        Loaded Microstructure object with stale phase properties.
    job : dict
        Job dictionary with phase_properties, ref_frame_angle, and
        element_homogenization keys.
    log_path : str or None
        Path to log.md for writing property details.
    settings : dict
        Global settings (verbose_console, verbose_log).

    Returns
    -------
    None
        The function modifies `ms` in place.
    """
    vc = settings.get("verbose_console", "medium")
    console_on = vc in ("medium", "high")

    phase_properties = job.get("phase_properties", {})
    if phase_properties:
        if console_on:
            print("  Re-reading phase property files...")
        # Create temporary ms to hold fresh properties
        _tmp = Microstructure()
        _tmp.NumberPhases = len(phase_properties)
        _tmp.PhaseName = [''] * _tmp.NumberPhases
        _tmp.PhasePropertyFilename = [''] * _tmp.NumberPhases
        _tmp.PhaseStiffnessMatrix = [None] * _tmp.NumberPhases
        _tmp.PhaseThermalExpansionMatrix = [None] * _tmp.NumberPhases
        _tmp.PhaseThermalConductivityMatrix = [None] * _tmp.NumberPhases
        _tmp.PhaseDensity = np.zeros(_tmp.NumberPhases)
        _tmp = read_properties(_tmp, phase_properties, log_path=log_path, settings=settings)

        # Overwrite properties on the loaded ms
        for i in range(min(len(phase_properties), ms.NumberPhases)):
            ms.PhaseName[i] = _tmp.PhaseName[i]
            ms.PhasePropertyFilename[i] = _tmp.PhasePropertyFilename[i]
            ms.PhaseStiffnessMatrix[i] = _tmp.PhaseStiffnessMatrix[i]
            ms.PhaseThermalExpansionMatrix[i] = _tmp.PhaseThermalExpansionMatrix[i]
            ms.PhaseThermalConductivityMatrix[i] = _tmp.PhaseThermalConductivityMatrix[i]
            ms.PhaseDensity[i] = _tmp.PhaseDensity[i]

    # Recompute EBSD correction matrices (ref_frame_angle may have changed)
    ref_frame_angle = job.get("ref_frame_angle", 90)
    direction_cosines, bond_matrix, bond_matrix_inv_T, theta_rad = \
        compute_ebsd_correction_matrix(ref_frame_angle)
    ms.EBSDCorrectionMatrix = bond_matrix
    ms.EBSDCorrectionMatrixInvT = bond_matrix_inv_T
    ms.EBSDDirectionCosines = direction_cosines
    ms.EBSDCorrectionAngle = theta_rad

    # Recompute homogenization method (may have changed)
    ms.ElementLevelHomogenizationMethodValue = job.get("element_homogenization", 4)

    # Recompute homogenized density from fresh properties
    ms.HomogenizedDensity = float(np.sum(ms.PhaseDensity * ms.PhaseVolumeFraction))

    if console_on:
        print(f"  Phase properties refreshed, homogenized density: {ms.HomogenizedDensity:.1f} kg/m³")


def run_job(job, job_num, total_jobs, settings):
    """
    Run a single TESA job through the full 4-stage pipeline.

    Parameters
    ----------
    job : dict
        Job dictionary with EBSD file, phase properties, mesh settings,
        and analysis settings. Optional keys:
        - "job_name": custom result folder name (default: EBSD file stem)
        - "mesh_source": path to folder with saved mesh to reuse
    job_num : int
        Current job number (1-based, for display).
    total_jobs : int
        Total number of jobs (for display).
    settings : dict
        Global settings (verbose, figure display, colormaps, etc.).

    Returns
    -------
    ms : Microstructure or None
        The completed Microstructure object, or None if the job failed.
    study_dir : str or None
        Path to the per-study results folder, or None if the job failed.
    """
    # ── Matplotlib backend ────────────────────────────────────────────
    # Use non-interactive Agg backend when figures are not displayed on screen
    if not settings.get("show_figures", True):
        import matplotlib
        matplotlib.use('Agg')

    # ── Job setup ─────────────────────────────────────────────────────
    # Create results folders, validate inputs, write log header and console banner
    map_dir, study_dir, log_path, success = setup_job(job, job_num, total_jobs, settings=settings)

    # Skip job if validation failed (error written to log.md and console)
    if not success:
        return None, study_dir

    # Start job timer
    job_t0 = time.time()

    # ── Stages 1 & 2: Load EBSD data + Generate mesh ─────────────────
    # Shared data (input_data, microstructure, mesh) goes to map_dir
    # mesh_source allows loading mesh from a different map folder
    mesh_source = job.get("mesh_source", None)
    mesh_dir = mesh_source if mesh_source else map_dir

    if job.get("reuse_mesh", False):
        ms_loaded = load_mesh(mesh_dir, job)
        if ms_loaded is not None:
            ms = ms_loaded
            # Re-read phase property files (may have changed since mesh was saved)
            _refresh_phase_properties(ms, job, log_path, settings)
        else:
            # No saved mesh — run full Stages 1 & 2 (output to map_dir)
            ms = load_ebsd(job, run_dir=map_dir, log_path=log_path, settings=settings)
            ms = create_mesh(ms, job, run_dir=map_dir, log_path=log_path, settings=settings)
            save_mesh(ms, map_dir, job)
    else:
        # reuse_mesh disabled — always run full pipeline (output to map_dir)
        ms = load_ebsd(job, run_dir=map_dir, log_path=log_path, settings=settings)
        ms = create_mesh(ms, job, run_dir=map_dir, log_path=log_path, settings=settings)

    # ── Stage 3: AEH analysis ─────────────────────────────────────────
    # Per-study results go to study_dir
    ms = run_analysis(ms, job, run_dir=study_dir, log_path=log_path, settings=settings)

    # ── Stage 4: Post-processing ─────────────────────────────────────
    # Save result files, write log tables, generate all plots
    ms = post_process(ms, job, run_dir=study_dir, log_path=log_path, settings=settings)

    # Print job completion time to console and log
    job_elapsed = time.time() - job_t0
    job_mins = int(job_elapsed // 60)
    job_secs = job_elapsed % 60
    ebsd_name = os.path.basename(job["ebsd_file"])
    study_name = job.get("study_name", "default")
    if job_mins > 0:
        time_str = f"{job_mins}m {job_secs:.1f}s"
    else:
        time_str = f"{job_secs:.1f}s"
    print(f"\n{'=' * 70}")
    print(f"  Job {job_num}/{total_jobs} complete — {ebsd_name}/{study_name} — Total time: {time_str}")
    print(f"  Results: {study_dir}")
    print(f"{'=' * 70}")

    # Write to log
    if log_path and os.path.isfile(log_path):
        with open(log_path, "a") as f:
            f.write(f"\n---\n\n**Total job time:** {time_str}\n")

    return ms, study_dir
