"""
Set up a TESA analysis job: create results folder, log header, and console banner.
"""

import os
import shutil
from datetime import datetime


def setup_job(job, job_num, total_jobs, settings=None):
    """
    Create the results folder and initialize log.md for a job.

    Parameters
    ----------
    job : dict
        Job dictionary with keys: ebsd_file, euler_col, xy_col, phase_col,
        ref_frame_angle, phase_properties, mesh_type, advanced_mesh_params [K, R, g],
        target_elements, element_homogenization, run_thermoelastic, run_heat_conduction.
    job_num : int
        Current job number (1-indexed).
    total_jobs : int
        Total number of jobs.
    settings : dict or None
        Global settings (show_figures, verbose, etc.). If None, defaults are used.

    Returns
    -------
    map_dir : str
        Path to the shared map folder (input_data, microstructure, mesh).
    study_dir : str
        Path to the per-study results folder (AEH Analysis, Properties, Fields).
    log_path : str
        Path to the log.md file (inside study_dir).
    success : bool
        True if all input files exist and the job can proceed.
    """
    # Use defaults if settings not provided
    if settings is None:
        settings = {}
    if "verbose_console" not in settings:
        print("  WARNING: 'verbose_console' not in settings, using default 'medium'")
    vc = settings.get("verbose_console", "medium")
    console_on = vc in ("medium", "high")
    ebsd_file        = job["ebsd_file"]
    euler_col        = job["euler_col"]
    xy_col           = job["xy_col"]
    phase_col        = job["phase_col"]
    ref_frame_angle  = job["ref_frame_angle"]
    phase_properties = job["phase_properties"]
    mesh_type        = job["mesh_type"]
    mesh_params      = job.get("advanced_mesh_params", [0.01, 0.5, 0.3])
    mesh_convergence = job.get("mesh_convergence", [40, 0.2, 0.90, 100])
    homog_method     = job["element_homogenization"]
    run_te           = job.get("run_thermoelastic", True)
    run_tc           = job.get("run_heat_conduction", True)
    macro_mechanical_field_type = job.get("macro_mechanical_field_type", "none")
    macro_thermal_field_type = job.get("macro_thermal_field_type", "none")
    macro_mech       = job.get("macro_mechanical_field", [0, 0, 0, 0, 0, 0])
    macro_temp       = job.get("macro_temperature_field", 0)
    macro_therm      = job.get("macro_thermal_field", [0, 0, 0])
    if "random_seed" not in settings:
        print("  WARNING: 'random_seed' not in settings, using default 42")
    random_seed         = settings.get("random_seed", 42)
    remove_small_grains = job.get("remove_small_grains", False)
    min_grain_pixels    = job.get("min_grain_pixels", 10)

    # Validate field_plot_style — fail fast if invalid
    field_style = settings.get("field_plot_style", "default")
    valid_styles = ("default", "element", "smooth", "rbf")
    if field_style not in valid_styles:
        print(f"\n  ERROR: Invalid field_plot_style '{field_style}'.")
        print(f"  Valid options: {', '.join(valid_styles)}")
        return None, None, False

    # Validate field_colormap — fail fast if invalid
    field_cmap = settings.get("field_colormap", "jet")
    import matplotlib.pyplot as plt
    try:
        plt.cm.get_cmap(field_cmap)
    except ValueError:
        print(f"\n  ERROR: Invalid field_colormap '{field_cmap}'.")
        print(f"  Valid options include: jet, turbo, rainbow, viridis, RdBu_r, etc.")
        print(f"  See: https://matplotlib.org/stable/gallery/color/colormap_reference.html")
        return None, None, False

    # Validate grain_colormap — fail fast if invalid
    grain_cmap = settings.get("grain_colormap", "tab20")
    try:
        plt.cm.get_cmap(grain_cmap)
    except ValueError:
        print(f"\n  ERROR: Invalid grain_colormap '{grain_cmap}'.")
        print(f"  Valid options include: tab20, Set3, Paired, etc.")
        print(f"  See: https://matplotlib.org/stable/gallery/color/colormap_reference.html")
        return None, None, False

    # ── Validate all inputs — fail fast ─────────────────────────────────
    errors = []

    # Global settings validation
    valid_verbose = ("none", "medium", "high")
    if vc not in valid_verbose:
        errors.append(f"verbose_console '{vc}' invalid — must be one of: {', '.join(valid_verbose)}")
    vl = settings.get("verbose_log", "medium")
    if vl not in valid_verbose:
        errors.append(f"verbose_log '{vl}' invalid — must be one of: {', '.join(valid_verbose)}")
    if not isinstance(settings.get("show_figures", False), bool):
        errors.append(f"show_figures must be True or False")
    fp = settings.get("figure_pause", 1.0)
    if not isinstance(fp, (int, float)) or fp < 0:
        errors.append(f"figure_pause must be a number >= 0 (got {fp})")
    fd = settings.get("figure_dpi", 150)
    if not isinstance(fd, (int, float)) or fd <= 0:
        errors.append(f"figure_dpi must be a positive number (got {fd})")
    if not isinstance(settings.get("phase_colors", []), list):
        errors.append(f"phase_colors must be a list of color strings")
    rs = settings.get("random_seed", 42)
    if rs is not None and not isinstance(rs, int):
        errors.append(f"random_seed must be an integer or None (got {type(rs).__name__}: {rs})")

    # Job settings validation — EBSD data
    if not isinstance(euler_col, int) or euler_col < 1:
        errors.append(f"euler_col must be an integer >= 1 (got {euler_col})")
    if not isinstance(xy_col, int) or xy_col < 1:
        errors.append(f"xy_col must be an integer >= 1 (got {xy_col})")
    if not isinstance(phase_col, int) or phase_col < 1:
        errors.append(f"phase_col must be an integer >= 1 (got {phase_col})")
    if not isinstance(ref_frame_angle, (int, float)):
        errors.append(f"ref_frame_angle must be a number (got {ref_frame_angle})")

    # Job settings validation — phase properties
    if not isinstance(phase_properties, dict) or len(phase_properties) == 0:
        errors.append(f"phase_properties must be a non-empty dict (e.g., {{1: 'file.txt'}})")

    # Job settings validation — mesh
    if mesh_type not in (1, 2, 3):
        errors.append(f"mesh_type must be 1, 2, or 3 (got {mesh_type})")
    if not isinstance(mesh_params, (list, tuple)) or len(mesh_params) != 3:
        errors.append(f"advanced_mesh_params must be a list of 3 numbers [K, R, g]")
    else:
        for i, v in enumerate(mesh_params):
            if not isinstance(v, (int, float)):
                errors.append(f"advanced_mesh_params[{i}] must be a number (got {v})")
    te_uniform = job.get("target_elements", 500)
    if not isinstance(te_uniform, int) or te_uniform <= 0:
        errors.append(f"target_elements must be a positive integer (got {te_uniform})")
    # Job settings validation — small grain absorption
    if not isinstance(remove_small_grains, bool):
        errors.append(f"remove_small_grains must be True or False (got {remove_small_grains})")
    if not isinstance(min_grain_pixels, int) or isinstance(min_grain_pixels, bool) or min_grain_pixels < 1:
        errors.append(f"min_grain_pixels must be an integer >= 1 (got {min_grain_pixels})")
    if not isinstance(mesh_convergence, (list, tuple)) or len(mesh_convergence) != 4:
        errors.append(f"mesh_convergence must be a list of 4 values [min_iter, q_worst_avg, q_mean, max_iter]")
    else:
        if not isinstance(mesh_convergence[0], int) or mesh_convergence[0] < 1:
            errors.append(f"mesh_convergence[0] (min_iter) must be a positive integer")
        if not isinstance(mesh_convergence[1], (int, float)) or not (0 <= mesh_convergence[1] <= 1):
            errors.append(f"mesh_convergence[1] (q_worst_avg_target) must be between 0 and 1")
        if not isinstance(mesh_convergence[2], (int, float)) or not (0 <= mesh_convergence[2] <= 1):
            errors.append(f"mesh_convergence[2] (q_mean_target) must be between 0 and 1")
        if not isinstance(mesh_convergence[3], int) or mesh_convergence[3] < mesh_convergence[0]:
            errors.append(f"mesh_convergence[3] (max_iter) must be an integer >= min_iter")
    # mesh_floor_ratio validation (optional, Type 1 only, default 0.25)
    _msf = job.get("mesh_floor_ratio", 0.25)
    if not isinstance(_msf, (int, float)) or _msf < 0 or _msf > 1:
        errors.append(f"mesh_floor_ratio must be a number between 0 and 1 (got {_msf})")
    # junction_refine_ratio validation (optional, Type 1 only, default 0.7)
    _jrr = job.get("junction_refine_ratio", 0.7)
    if not isinstance(_jrr, (int, float)) or _jrr < 0:
        errors.append(f"junction_refine_ratio must be a number >= 0 (got {_jrr})")
    if homog_method not in (1, 2, 3, 4, 5):
        errors.append(f"element_homogenization must be 1-5 (got {homog_method})")

    # Job settings validation — analysis
    if not isinstance(run_te, bool):
        errors.append(f"run_thermoelastic must be True or False (got {run_te})")
    if not isinstance(run_tc, bool):
        errors.append(f"run_heat_conduction must be True or False (got {run_tc})")
    if macro_mechanical_field_type not in ("none", "stress", "strain"):
        errors.append(f"macro_mechanical_field_type must be 'none', 'stress', or 'strain' (got '{macro_mechanical_field_type}')")
    if macro_thermal_field_type not in ("none", "temperature_gradient", "heat_flux"):
        errors.append(f"macro_thermal_field_type must be 'none', 'temperature_gradient', or 'heat_flux' (got '{macro_thermal_field_type}')")
    if macro_mechanical_field_type != "none" and not run_te:
        print(f"  WARNING: macro_mechanical_field_type='{macro_mechanical_field_type}' ignored (run_thermoelastic=False)")
    if macro_thermal_field_type != "none" and not run_tc:
        print(f"  WARNING: macro_thermal_field_type='{macro_thermal_field_type}' ignored (run_heat_conduction=False)")
    if not isinstance(macro_mech, (list, tuple)) or len(macro_mech) != 6:
        errors.append(f"macro_mechanical_field must be a list of 6 numbers")
    if not isinstance(macro_temp, (int, float)):
        errors.append(f"macro_temperature_field must be a number (got {macro_temp})")
    # macro_thermal_field_type already validated above (line ~170)
    if not isinstance(macro_therm, (list, tuple)) or len(macro_therm) != 3:
        errors.append(f"macro_thermal_field must be a list of 3 numbers")

    # Report all errors at once
    if errors:
        print(f"\n  ERROR: Invalid input settings ({len(errors)} error(s)):")
        for e in errors:
            print(f"    - {e}")
        return None, None, False

    # Create results folders:
    #   map_dir   = results/{ebsd_stem}/           — shared data (input_data, microstructure, mesh)
    #   study_dir = results/{ebsd_stem}/{study}/    — per-study results (analysis, properties, fields)
    ebsd_stem = os.path.splitext(os.path.basename(ebsd_file))[0]
    study_name = job.get("study_name", "default")
    map_dir = os.path.join("results", ebsd_stem)
    study_dir = os.path.join(map_dir, f"study_{study_name}")

    # map_dir is never deleted — shared data persists across studies
    os.makedirs(map_dir, exist_ok=True)

    # study_dir is cleaned on each new run
    if os.path.exists(study_dir):
        shutil.rmtree(study_dir)
    os.makedirs(study_dir, exist_ok=True)

    log_path = os.path.join(study_dir, "log.md")

    # Console banner (always printed)
    print(f"\n{'=' * 70}")
    print(f"  TESA Analysis — Job {job_num}/{total_jobs}")
    print(f"  EBSD file : {ebsd_file}")
    print(f"  Results   : {study_dir}")
    print(f"{'=' * 70}")
    if not console_on:
        print("  Running (verbose=none, see log.md for details)...")

    # Validate input files exist — log errors and skip job
    missing = []
    if not os.path.isfile(ebsd_file):
        missing.append(f"EBSD file not found: {ebsd_file}")
    for ph_num, ph_file in sorted(phase_properties.items()):
        if not os.path.isfile(ph_file):
            missing.append(f"Phase {ph_num} property file not found: {ph_file}")
    if missing:
        with open(log_path, "w") as f:
            f.write("# TESA Toolbox — Thermal and Elastic Scale-bridging Analysis\n\n")
            f.write(f"**EBSD file:** `{os.path.basename(ebsd_file)}`\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## ERROR — Missing Input Files\n\n")
            for msg in missing:
                f.write(f"- {msg}\n")
                print(f"  ERROR: {msg}")  # errors always printed
            f.write("\nJob aborted.\n")
        print(f"  Log saved to: {log_path}")  # errors always printed
        return map_dir, study_dir, log_path, False
    if console_on:
        print(f"  Columns   : Euler={euler_col},{euler_col+1},{euler_col+2}  "
              f"XY={xy_col},{xy_col+1}  Phase={phase_col}")
        for ph_num, ph_file in sorted(phase_properties.items()):
            print(f"  Phase {ph_num}   : {ph_file}")
        mesh_type_names = {1: "conforming non-uniform", 2: "non-conforming hexgrid", 3: "non-conforming rectangular"}
        print(f"  Mesh type : {mesh_type} ({mesh_type_names.get(mesh_type, 'unknown')})")
        print(f"  Mesh [K,R,g] : {mesh_params}")
        print(f"  Target elements : {job.get('target_elements', 'N/A')}")
        if remove_small_grains:
            print(f"  Small grain removal : enabled (min_grain_pixels={min_grain_pixels})")
        if mesh_type == 1:
            print(f"  Size floor  : {job.get('mesh_floor_ratio', 0.25)}  "
                  f"(h_min ≥ {job.get('mesh_floor_ratio', 0.25)}×h_max)")
            _jrr_val = job.get('junction_refine_ratio', 0.7)
            if _jrr_val > 0:
                print(f"  Junction refine : radius = {_jrr_val} × h_max")
        print(f"  Convergence : {mesh_convergence}  [min_iter, q_worst_avg, q_mean, max_iter]")
        homog_names = {1: "Nearest", 2: "Voigt", 3: "Reuss", 4: "Hill", 5: "Geometric Mean"}
        print(f"  Homog.    : {homog_method} ({homog_names.get(homog_method, 'unknown')})")
        analyses = []
        if run_te: analyses.append("Thermo-elastic")
        if run_tc: analyses.append("Thermal conductivity")
        print(f"  Analysis  : {', '.join(analyses) if analyses else 'None'}")
        if macro_mechanical_field_type != "none":
            print(f"  Fields    : stress/strain ({macro_mechanical_field_type})")
            print(f"    Mechanical : {macro_mech}")
            print(f"    ΔT         : {macro_temp}")
        if macro_thermal_field_type != "none":
            print(f"  Fields    : heat flux ({macro_thermal_field_type})")
            print(f"    Thermal    : {macro_therm}")
        print(f"  Random seed : {random_seed}")

        # Global settings
        print(f"  ── Global Settings ──")
        print(f"  Verbose     : console={vc}, log={settings.get('verbose_log', 'medium')}")
        print(f"  Figures     : show={settings.get('show_figures', False)}, "
              f"pause={settings.get('figure_pause', 1.0)}s, "
              f"dpi={settings.get('figure_dpi', 150)}")
        print(f"  Field style : {settings.get('field_plot_style', 'rbf')}")
        print(f"  Field cmap  : {settings.get('field_colormap', 'jet')}")
        print(f"  Grain cmap  : {settings.get('grain_colormap', 'tab20')}")

    # Create log.md
    with open(log_path, "w") as f:
        f.write("# TESA Toolbox — Thermal and Elastic Scale-bridging Analysis\n\n")
        f.write(f"**EBSD file:** `{os.path.basename(ebsd_file)}`\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Input Settings\n\n")
        ec = f"{euler_col}, {euler_col+1}, {euler_col+2}"
        xc = f"{xy_col}, {xy_col+1}"
        pc = f"{phase_col}"
        ra = f"{ref_frame_angle}"
        f.write(f"| {'Setting':<20s} | {'Value':^9s} | {'Description':<19s} |\n")
        f.write(f"|{'-'*22}|:{'-'*9}:|{'-'*21}|\n")
        f.write(f"| {'Euler angle columns':<20s} | {ec:^9s} | {'phi1, Phi, phi2':<19s} |\n")
        f.write(f"| {'Coordinate columns':<20s} | {xc:^9s} | {'X, Y':<19s} |\n")
        f.write(f"| {'Phase column':<20s} | {pc:^9s} | {'':<19s} |\n")
        f.write(f"| {'Ref. frame angle':<20s} | {ra:^9s} | {'degrees':<19s} |\n")
        mesh_type_names = {1: "conforming non-uniform", 2: "non-conforming hexgrid", 3: "non-conforming rectangular"}
        mt = f"{mesh_type}"
        mp = f"{mesh_params}"
        te = f"{job.get('target_elements', 'N/A')}"
        f.write(f"| {'Mesh type':<20s} | {mt:^9s} | {mesh_type_names.get(mesh_type, ''):<19s} |\n")
        f.write(f"| {'Mesh [K,R,g]':<20s} | {mp:^9s} | {'curv, MA, gradient':<19s} |\n")
        f.write(f"| {'Target elements':<20s} | {te:^9s} | {'target element count':<19s} |\n")
        if mesh_type == 1:
            msf_val = str(job.get('mesh_floor_ratio', 0.25))
            f.write(f"| {'Mesh size floor':<20s} | {msf_val:^9s} | {'h_min/h_max ratio':<19s} |\n")
            jrr_val = str(job.get('junction_refine_ratio', 0.7))
            f.write(f"| {'Junction refine R':<20s} | {jrr_val:^9s} | {'× h_max radius':<19s} |\n")
        rsg = str(remove_small_grains)
        mgp = str(min_grain_pixels)
        f.write(f"| {'Remove small grains':<20s} | {rsg:^9s} | {'merge small grains':<19s} |\n")
        f.write(f"| {'Min grain pixels':<20s} | {mgp:^9s} | {'data points cutoff':<19s} |\n")
        mc = str(mesh_convergence)
        f.write(f"| {'Mesh convergence':<20s} | {mc:^9s} | {'min,qwa,qmean,max':<19s} |\n")
        homog_names = {1: "Nearest", 2: "Voigt", 3: "Reuss", 4: "Hill", 5: "Geometric Mean"}
        hm = f"{homog_method}"
        analyses = []
        if run_te: analyses.append("Thermo-elastic")
        if run_tc: analyses.append("Thermal cond.")
        al = ", ".join(analyses) if analyses else "None"
        f.write(f"| {'Homogenization':<20s} | {hm:^9s} | {homog_names.get(homog_method, ''):<19s} |\n")
        f.write(f"| {'Analyses':<20s} | {al:^9s} | {'':<19s} |\n")
        f.write(f"| {'Stress/strain fields':<20s} | {macro_mechanical_field_type:^9s} | {'':<19s} |\n")
        f.write(f"| {'Heat flux fields':<20s} | {macro_thermal_field_type:^9s} | {'':<19s} |\n")
        rs = f"{random_seed}"
        f.write(f"| {'Random seed':<20s} | {rs:^9s} | {'reproducibility':<19s} |\n\n")

        # Global settings table
        f.write("### Global Settings\n\n")
        f.write(f"| {'Setting':<20s} | {'Value':<40s} |\n")
        f.write(f"|{'-'*22}|{'-'*42}|\n")
        f.write(f"| {'Verbose (console)':<20s} | {vc:<40s} |\n")
        vl = settings.get('verbose_log', 'medium')
        f.write(f"| {'Verbose (log)':<20s} | {vl:<40s} |\n")
        sf = str(settings.get('show_figures', False))
        f.write(f"| {'Show figures':<20s} | {sf:<40s} |\n")
        fp = str(settings.get('figure_pause', 1.0))
        f.write(f"| {'Figure pause (s)':<20s} | {fp:<40s} |\n")
        fd = str(settings.get('figure_dpi', 150))
        f.write(f"| {'Figure DPI':<20s} | {fd:<40s} |\n")
        fps = settings.get('field_plot_style', 'rbf')
        f.write(f"| {'Field plot style':<20s} | {fps:<40s} |\n")
        fcm = settings.get('field_colormap', 'jet')
        f.write(f"| {'Field colormap':<20s} | {fcm:<40s} |\n")
        gcm = settings.get('grain_colormap', 'tab20')
        f.write(f"| {'Grain colormap':<20s} | {gcm:<40s} |\n")
        pc = str(settings.get('phase_colors', []))
        if len(pc) > 40:
            pc = pc[:37] + '...'
        f.write(f"| {'Phase colors':<20s} | {pc:<40s} |\n\n")

        f.write("**Phase property files:**\n\n")
        for ph_num, ph_file in sorted(phase_properties.items()):
            f.write(f"- Phase {ph_num}: `{os.path.basename(ph_file)}`\n")
        f.write("\n---\n\n")

    return map_dir, study_dir, log_path, True
