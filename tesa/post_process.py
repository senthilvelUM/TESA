"""
Stage 4/5: Post-processor — save results and visualize fields.

Reads computed results from ms (populated by run_analysis) and:
1. Saves effective property matrices to 'homogenized_properties' (organized by method)
2. Saves characteristic function plots to 'AEH_characteristic_functions' (chi, psi, phi)
3. Generates field plots to 'microscale_fields' (stresses, strains, heat flux, temp gradient)
4. Writes analysis summary tables to log.md

Called from run_tesa.py after run_analysis().
"""

import os
import time
import numpy as np


def post_process(ms, job, run_dir=None, log_path=None, settings=None):
    """
    Stage 4: Post-process AEH analysis results.

    Parameters
    ----------
    ms : Microstructure
        Must have effective properties populated by run_analysis().
    job : dict
        Job configuration dictionary.
    run_dir : str
        Results directory.
    log_path : str
        Path to log.md file.
    settings : dict
        Global settings.

    Returns
    -------
    ms : Microstructure
        Unchanged (post-processing only reads from ms).
    """
    if settings is None:
        settings = {}

    # Verbosity settings
    vc = settings.get("verbose_console", "medium")
    vl = settings.get("verbose_log", "medium")
    console_on = vc in ("medium", "high")
    log_on = vl in ("medium", "high")

    # Analysis flags from job dictionary
    run_te = job.get("run_thermoelastic", False)
    run_tc = job.get("run_heat_conduction", False)
    has_stress_strain = job.get("macro_mechanical_field_type", "none") != "none"
    has_heat_flux = job.get("macro_thermal_field_type", "none") != "none"

    if not run_te and not run_tc:
        return ms

    t0 = time.time()

    if console_on:
        print("\n── Stage 4/5: Post-processing ──")

    # Create output directories
    analysis_dir = os.path.join(run_dir, "AEH_characteristic_functions") if run_dir else None
    properties_dir = os.path.join(run_dir, "homogenized_properties") if run_dir else None
    field_plots_dir = os.path.join(run_dir, "microscale_fields") if run_dir else None
    wave_speed_plots_dir = os.path.join(run_dir, "wave_speed_plots") if run_dir else None
    if analysis_dir:
        os.makedirs(analysis_dir, exist_ok=True)
    if properties_dir:
        os.makedirs(properties_dir, exist_ok=True)
    if field_plots_dir:
        os.makedirs(field_plots_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # Thermoelastic post-processing
    # ══════════════════════════════════════════════════════════════════════
    if run_te:
        _postprocess_thermoelastic(ms, job, analysis_dir, properties_dir, field_plots_dir,
                                    wave_speed_plots_dir, log_path, settings, console_on, log_on, has_stress_strain)

    # ══════════════════════════════════════════════════════════════════════
    # Thermal conductivity post-processing
    # ══════════════════════════════════════════════════════════════════════
    if run_tc:
        _postprocess_thermal_conductivity(ms, job, analysis_dir, properties_dir, field_plots_dir,
                                           log_path, settings, console_on, log_on, has_heat_flux)

    # Stage 4 timing
    elapsed = time.time() - t0
    if console_on:
        print(f"  Stage 4 complete ({elapsed:.1f}s)")

    # Log completion
    if log_on and log_path:
        with open(log_path, "a") as f:
            f.write(f"\n**Stage 4 elapsed time:** {elapsed:.1f}s\n\n")

    return ms


# ═══════════════════════════════════════════════════════════════════════════
# Thermoelastic post-processing
# ═══════════════════════════════════════════════════════════════════════════

def _postprocess_thermoelastic(ms, job, analysis_dir, properties_dir, field_plots_dir,
                                wave_speed_plots_dir, log_path, settings, console_on, log_on, has_stress_strain):
    """
    Save files, write log, and generate plots for thermoelastic analysis.

    Writes effective stiffness, compliance, thermal expansion, and stress-
    temperature modulus files for all homogenization methods. Generates
    wave speed, characteristic function, and stress/strain field plots.

    Parameters
    ----------
    ms : Microstructure
        Must have thermoelastic effective properties populated.
    job : dict
        Job configuration dictionary.
    analysis_dir : str or None
        Directory for characteristic function plots.
    properties_dir : str or None
        Directory for homogenized property text files.
    field_plots_dir : str or None
        Directory for microscale field plots.
    wave_speed_plots_dir : str or None
        Directory for wave speed plots.
    log_path : str or None
        Path to log.md file.
    settings : dict
        Global settings.
    console_on : bool
        Whether to print progress to console.
    log_on : bool
        Whether to write to log.md.
    has_stress_strain : bool
        Whether microscale stress/strain fields were computed.

    Returns
    -------
    None
        Results are written to disk; ms is not modified.
    """
    from .get_aeh import save_stiffness_file, save_engineering_properties_file

    # Extract all effective properties from ms
    D_aeh = ms.DEffectiveAEH
    alpha_aeh = ms.alphaEffectiveAEH
    beta_aeh = ms.betaEffectiveAEH
    props_aeh = ms.EngineeringProperties

    D_voigt = ms.DEffectiveVoigt
    D_reuss = ms.DEffectiveReuss
    D_hill = ms.DEffectiveHill
    D_geo = ms.DEffectiveGeoMean

    alpha_voigt = ms.alphaEffectiveVoigt
    alpha_reuss = ms.alphaEffectiveReuss
    alpha_hill = ms.alphaEffectiveHill
    alpha_geo = ms.alphaEffectiveGeoMean

    beta_voigt = ms.betaEffectiveVoigt
    beta_reuss = ms.betaEffectiveReuss
    beta_hill = ms.betaEffectiveHill
    beta_geo = ms.betaEffectiveGeoMean

    wave_results = ms.WaveSpeedResults
    rho = ms.HomogenizedDensity

    # ── Write analysis tables to log.md ──────────────────────────────
    if log_on and log_path:
        _log_thermoelastic(log_path, D_aeh, alpha_aeh, beta_aeh, props_aeh,
                           D_voigt, D_reuss, D_hill, D_geo,
                           alpha_voigt, alpha_reuss, alpha_hill, alpha_geo,
                           beta_voigt, beta_reuss, beta_hill, beta_geo,
                           wave_results, rho)

    # ── Save results to text files (organized by method in subfolders) ──
    if properties_dir:
        # All methods: stiffness, thermal expansion, beta
        methods_data = [
            ("AEH",     D_aeh,   alpha_aeh,   beta_aeh),
            ("Voigt",   D_voigt, alpha_voigt,  beta_voigt),
            ("Reuss",   D_reuss, alpha_reuss,  beta_reuss),
            ("Hill",    D_hill,  alpha_hill,   beta_hill),
            ("GeoMean", D_geo,   alpha_geo,    beta_geo),
        ]
        for name, D, alpha, beta in methods_data:
            # Create method subfolder
            method_dir = os.path.join(properties_dir, name)
            os.makedirs(method_dir, exist_ok=True)

            # Stiffness matrix
            save_stiffness_file(D, os.path.join(method_dir, f"{name}_effective_stiffness.txt"))

            # Compliance matrix (inverse of stiffness)
            S = np.linalg.inv(D)
            np.savetxt(os.path.join(method_dir, f"{name}_effective_compliance.txt"),
                       S, fmt='%15.6e',
                       header=f'{name} Effective Compliance Matrix S = inv(C) (1/Pa)')

            # Thermal expansion
            np.savetxt(os.path.join(method_dir, f"{name}_thermal_expansion.txt"),
                       np.asarray(alpha).ravel(), fmt='%15.6e',
                       header=f'{name} Effective Thermal Expansion (1/K)')

            # Stress-temperature modulus
            np.savetxt(os.path.join(method_dir, f"{name}_stress_temperature_modulus.txt"),
                       np.asarray(beta).ravel(), fmt='%15.6e',
                       header=f'{name} Effective Stress-Temperature Modulus (Pa/K)')

        # AEH-only: engineering properties
        aeh_dir = os.path.join(properties_dir, "AEH")
        save_engineering_properties_file(props_aeh, os.path.join(aeh_dir, "AEH_engineering_properties.txt"))

        if console_on:
            print(f"  Saved stiffness, thermal expansion, beta, and engineering properties to: homogenized_properties/")

    # ── Plot wave speed fields → wave_speed_plots/{method}/ ──
    if wave_speed_plots_dir and hasattr(ms, 'WaveSpeedResults') and ms.WaveSpeedResults:
        from .plot_wave_speeds import plot_all_wave_speeds
        # Merge job-level wave speed settings into settings for the plotter
        ws_settings = dict(settings)
        ws_settings["wave_speed_plots"] = job.get("wave_speed_plots", "all")
        ws_settings["wave_speed_plot_type"] = job.get("wave_speed_plot_type", "both")
        ws_settings["wave_speed_sphere_elev"] = job.get("wave_speed_sphere_elev", 30)
        ws_settings["wave_speed_sphere_azim"] = job.get("wave_speed_sphere_azim", 30)
        plot_all_wave_speeds(ms, wave_speed_plots_dir=wave_speed_plots_dir, settings=ws_settings,
                             verbose=console_on)

    # ── Plot per-phase (single crystal) wave speed fields ──
    if wave_speed_plots_dir and hasattr(ms, 'PhaseWaveSpeedResults') and ms.PhaseWaveSpeedResults:
        from .plot_wave_speeds import plot_all_phase_wave_speeds
        ws_settings = dict(settings)
        ws_settings["wave_speed_plots"] = job.get("wave_speed_plots", "all")
        ws_settings["wave_speed_plot_type"] = job.get("wave_speed_plot_type", "both")
        ws_settings["wave_speed_sphere_elev"] = job.get("wave_speed_sphere_elev", 30)
        ws_settings["wave_speed_sphere_azim"] = job.get("wave_speed_sphere_azim", 30)
        plot_all_phase_wave_speeds(ms, wave_speed_plots_dir=wave_speed_plots_dir,
                                   settings=ws_settings, verbose=console_on)

    # ── Plot chi characteristic functions → AEH Analysis/chi/ ─────────
    if analysis_dir and hasattr(ms, 'chiCharacteristicFunctions') and ms.chiCharacteristicFunctions is not None:
        from .plot_chi import plot_all_chi
        plot_all_chi(ms, analysis_dir=analysis_dir, settings=settings,
                     verbose=console_on)

    # ── Plot psi characteristic functions → AEH Analysis/psi/ ─────────
    if analysis_dir and hasattr(ms, 'psiCharacteristicFunctions') and ms.psiCharacteristicFunctions is not None:
        from .plot_psi import plot_all_psi
        plot_all_psi(ms, analysis_dir=analysis_dir, settings=settings,
                     verbose=console_on)

    # ── Plot stress/strain fields → microscale_fields/stresses/, strains/ ──
    if has_stress_strain and field_plots_dir:
        if hasattr(ms, 'Microfield') and ms.Microfield is not None and ms.Microfield[0] is not None:
            from .plot_stress_strain import plot_all_stress_strain
            plot_all_stress_strain(ms, analysis_dir=field_plots_dir, settings=settings,
                                   verbose=console_on)


# ═══════════════════════════════════════════════════════════════════════════
# Thermal conductivity post-processing
# ═══════════════════════════════════════════════════════════════════════════

def _postprocess_thermal_conductivity(ms, job, analysis_dir, properties_dir, field_plots_dir,
                                       log_path, settings, console_on, log_on, has_heat_flux):
    """
    Save files, write log, and generate plots for thermal conductivity analysis.

    Writes effective thermal conductivity matrices and in-plane anisotropy
    files for all homogenization methods. Generates phi characteristic
    function and heat flux field plots.

    Parameters
    ----------
    ms : Microstructure
        Must have thermal conductivity effective properties populated.
    job : dict
        Job configuration dictionary.
    analysis_dir : str or None
        Directory for characteristic function plots.
    properties_dir : str or None
        Directory for homogenized property text files.
    field_plots_dir : str or None
        Directory for microscale field plots.
    log_path : str or None
        Path to log.md file.
    settings : dict
        Global settings.
    console_on : bool
        Whether to print progress to console.
    log_on : bool
        Whether to write to log.md.
    has_heat_flux : bool
        Whether microscale heat flux fields were computed.

    Returns
    -------
    None
        Results are written to disk; ms is not modified.
    """

    # Extract all effective properties from ms
    kappa_aeh = ms.kappaEffectiveAEH
    kappa_voigt = ms.kappaEffectiveVoigt
    kappa_reuss = ms.kappaEffectiveReuss
    kappa_hill = ms.kappaEffectiveHill
    kappa_geo = ms.kappaEffectiveGeoMean

    # ── Write analysis tables to log.md ──────────────────────────────
    if log_on and log_path:
        _log_thermal_conductivity(log_path, kappa_aeh, kappa_voigt, kappa_reuss,
                                  kappa_hill, kappa_geo)

    # ── Save results to text files (organized by method in subfolders) ──
    if properties_dir:
        for name, kappa in [("AEH", kappa_aeh), ("Voigt", kappa_voigt),
                            ("Reuss", kappa_reuss), ("Hill", kappa_hill),
                            ("GeoMean", kappa_geo)]:
            # Create method subfolder (may already exist from thermo-elastic)
            method_dir = os.path.join(properties_dir, name)
            os.makedirs(method_dir, exist_ok=True)
            np.savetxt(os.path.join(method_dir, f"{name}_thermal_conductivity.txt"),
                       kappa, fmt='%15.6e',
                       header=f'{name} Effective Thermal Conductivity (W/(m K))')
            # Write in-plane anisotropy file
            _write_thermal_conductivity_anisotropy(name, kappa, method_dir)

        if console_on:
            print(f"  Saved thermal conductivity matrices to: homogenized_properties/")

    # ── Plot phi characteristic functions → AEH Analysis/phi/ ─────────
    if analysis_dir and hasattr(ms, 'thermalConductivityCharacteristicFunctions') and \
            ms.thermalConductivityCharacteristicFunctions is not None:
        from .plot_phi import plot_all_phi
        plot_all_phi(ms, analysis_dir=analysis_dir, settings=settings,
                     verbose=console_on)

    # ── Plot heat flux fields → microscale_fields/heat_flux/, temp_gradient/ ──
    if has_heat_flux and field_plots_dir:
        if hasattr(ms, 'MicrofieldHeatConduction') and ms.MicrofieldHeatConduction is not None \
                and ms.MicrofieldHeatConduction[0] is not None:
            # Clear RBF cache — heat conduction may use different QP coordinates
            ms._rbf_cache = None
            from .plot_heat_flux import plot_all_heat_flux
            plot_all_heat_flux(ms, analysis_dir=field_plots_dir, settings=settings,
                              verbose=console_on)


# ═══════════════════════════════════════════════════════════════════════════
# Thermal conductivity anisotropy helper
# ═══════════════════════════════════════════════════════════════════════════

def _write_thermal_conductivity_anisotropy(name, kappa, method_dir):
    """
    Compute and write in-plane thermal conductivity anisotropy for one method.

    Extracts the 2x2 in-plane tensor K_2D = kappa[0:2, 0:2], computes its
    principal values (k_max, k_min) and principal directions (theta_max,
    theta_min) via eigendecomposition, and writes the results to a text file.

    Parameters
    ----------
    name : str
        Method name (e.g. "AEH", "Voigt").
    kappa : ndarray, shape (3, 3)
        Effective thermal conductivity tensor (W/(m K)).
    method_dir : str
        Folder path where the output file is written.

    Returns
    -------
    None
        Results are written to a text file in `method_dir`.
    """
    # Extract 2x2 in-plane tensor from upper-left block of 3x3 tensor
    K2D = kappa[0:2, 0:2]
    k11, k12, k22 = K2D[0, 0], K2D[0, 1], K2D[1, 1]

    # Principal values: eigenvalues of symmetric 2x2 tensor
    #   k_avg = (k11 + k22) / 2
    #   R     = sqrt(((k11 - k22) / 2)^2 + k12^2)
    #   k_max = k_avg + R,  k_min = k_avg - R
    evals, evecs = np.linalg.eigh(K2D)   # evals ascending: evals[0]=k_min, evals[1]=k_max
    k_min = evals[0]
    k_max = evals[1]

    # Principal directions: angle from x1-axis (horizontal), CCW toward x2 positive, range [0, 180]
    #   theta = arctan2(v2, v1)  where v1 = x1-component, v2 = x2-component
    #   Eigenvectors have sign ambiguity; flip to upper half-plane (v2 >= 0) to enforce range.
    v_max = evecs[:, 1]                                      # eigenvector for k_max
    v_min = evecs[:, 0]                                      # eigenvector for k_min
    v_max = v_max if v_max[1] >= 0 else -v_max               # ensure x2-component >= 0
    v_min = v_min if v_min[1] >= 0 else -v_min
    theta_max = np.degrees(np.arctan2(v_max[1], v_max[0]))   # angle from x1-axis
    theta_min = np.degrees(np.arctan2(v_min[1], v_min[0]))

    # In-plane thermal conductivity anisotropy ratio
    A = k_max / k_min if k_min != 0 else float('inf')

    # Write results to text file
    fpath = os.path.join(method_dir, f"{name}_thermal_conductivity_anisotropy.txt")
    with open(fpath, 'w') as f:
        f.write(f"# {name} In-Plane Thermal Conductivity Anisotropy\n")
        f.write(f"# Extracted from 3x3 effective tensor; in-plane = x1-x2 plane (k11, k12, k22)\n")
        f.write(f"#\n")
        f.write(f"# 2x2 in-plane tensor (W/(m K)):\n")
        f.write(f"  k11 = {k11:15.6e}\n")
        f.write(f"  k12 = {k12:15.6e}\n")
        f.write(f"  k22 = {k22:15.6e}\n")
        f.write(f"#\n")
        # Frobenius norm of symmetric 2x2 tensor: ||K|| = sqrt(k11^2 + 2*k12^2 + k22^2)
        k_norm = np.sqrt(k11**2 + 2*k12**2 + k22**2)
        f.write(f"# Frobenius norm of 2x2 tensor (W/(m K)):\n")
        f.write(f"  ||K|| = {k_norm:15.6e}\n")
        f.write(f"#\n")
        f.write(f"# Principal values (W/(m K)):\n")
        f.write(f"  k_max = {k_max:15.6e}\n")
        f.write(f"  k_min = {k_min:15.6e}\n")
        f.write(f"#\n")
        f.write(f"# Principal directions (degrees from x1-axis, CCW toward x2 positive, range [0, 180]):\n")
        f.write(f"  theta_max = {theta_max:15.6f}\n")
        f.write(f"  theta_min = {theta_min:15.6f}\n")
        f.write(f"#\n")
        f.write(f"# In-plane thermal conductivity anisotropy:\n")
        f.write(f"  A = k_max / k_min = {A:15.6f}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Log writing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _log_thermoelastic(log_path, D_aeh, alpha_aeh, beta_aeh, props_aeh,
                       D_voigt, D_reuss, D_hill, D_geo,
                       alpha_voigt, alpha_reuss, alpha_hill, alpha_geo,
                       beta_voigt, beta_reuss, beta_hill, beta_geo,
                       wave_results, rho):
    """
    Write all thermoelastic results to log.md.

    Appends stiffness matrices, engineering properties, thermal expansion,
    and wave speed anisotropy tables in Markdown format.

    Parameters
    ----------
    log_path : str
        Path to log.md file.
    D_aeh : ndarray, shape (6, 6)
        AEH effective stiffness matrix (Pa).
    alpha_aeh : ndarray, shape (6,) or (6, 1)
        AEH effective thermal expansion (1/K).
    beta_aeh : ndarray, shape (6,) or (6, 1)
        AEH effective stress-temperature modulus (Pa/K).
    props_aeh : dict
        AEH engineering properties (E1, E2, E3, G12, nu12, etc.).
    D_voigt : ndarray, shape (6, 6)
        Voigt effective stiffness matrix (Pa).
    D_reuss : ndarray, shape (6, 6)
        Reuss effective stiffness matrix (Pa).
    D_hill : ndarray, shape (6, 6)
        Hill effective stiffness matrix (Pa).
    D_geo : ndarray, shape (6, 6)
        Geometric mean effective stiffness matrix (Pa).
    alpha_voigt, alpha_reuss, alpha_hill, alpha_geo : ndarray
        Thermal expansion vectors for each method.
    beta_voigt, beta_reuss, beta_hill, beta_geo : ndarray
        Stress-temperature modulus vectors for each method.
    wave_results : dict
        Wave speed results keyed by method name.
    rho : float
        Homogenized density (kg/m^3).

    Returns
    -------
    None
        Results are appended to the log file.
    """
    with open(log_path, "a") as f:
        # AEH stiffness
        f.write("### AEH Effective Stiffness (GPa)\n\n")
        f.write("```\n")
        for row in range(6):
            f.write("  " + "  ".join(f"{D_aeh[row, col]/1e9:10.4f}" for col in range(6)) + "\n")
        f.write("```\n\n")

        # AEH engineering properties
        f.write("### AEH Engineering Properties\n\n")
        f.write("| Property | Value |\n")
        f.write("|----------|-------|\n")
        for key in ['E1', 'E2', 'E3']:
            f.write(f"| {key:<8s} | {props_aeh[key]/1e9:.4f} GPa |\n")
        for key in ['G23', 'G13', 'G12']:
            f.write(f"| {key:<8s} | {props_aeh[key]/1e9:.4f} GPa |\n")
        for key in ['nu12', 'nu13', 'nu21', 'nu23', 'nu31', 'nu32']:
            f.write(f"| {key:<8s} | {props_aeh[key]:.6f} |\n")
        f.write("\n")

        # AEH thermal expansion
        f.write("### AEH Effective Thermal Expansion (1/K)\n\n")
        alpha_flat = np.asarray(alpha_aeh).ravel()
        f.write("| Component | Value |\n")
        f.write("|-----------|-------|\n")
        for i in range(6):
            f.write(f"| α_{i+1}     | {alpha_flat[i]:.6e} |\n")
        f.write("\n")

        # VRH comparison table (diagonal terms)
        f.write("### Stiffness Comparison — Diagonal Terms (GPa)\n\n")
        f.write("| Component | AEH | Voigt | Reuss | Hill | GeoMean |\n")
        f.write("|-----------|-----|-------|-------|------|---------|\n")
        labels = ['C11', 'C22', 'C33', 'C44', 'C55', 'C66']
        for i in range(6):
            f.write(f"| {labels[i]:<9s} | {D_aeh[i,i]/1e9:.4f} | {D_voigt[i,i]/1e9:.4f} | "
                    f"{D_reuss[i,i]/1e9:.4f} | {D_hill[i,i]/1e9:.4f} | {D_geo[i,i]/1e9:.4f} |\n")
        f.write("\n")

        # Thermal expansion comparison
        f.write("### Thermal Expansion Comparison (1/K)\n\n")
        f.write("| Component | AEH | Voigt | Reuss | Hill | GeoMean |\n")
        f.write("|-----------|-----|-------|-------|------|---------|\n")
        alpha_v = np.asarray(alpha_voigt).ravel()
        alpha_r = np.asarray(alpha_reuss).ravel()
        alpha_h = np.asarray(alpha_hill).ravel()
        alpha_g = np.asarray(alpha_geo).ravel()
        for i in range(6):
            f.write(f"| α_{i+1}     | {alpha_flat[i]:.4e} | {alpha_v[i]:.4e} | "
                    f"{alpha_r[i]:.4e} | {alpha_h[i]:.4e} | {alpha_g[i]:.4e} |\n")
        f.write("\n")

        # Stress-temperature modulus comparison
        f.write("### Stress-Temperature Modulus Comparison (Pa/K)\n\n")
        f.write("| Component | AEH | Voigt | Reuss | Hill | GeoMean |\n")
        f.write("|-----------|-----|-------|-------|------|---------|\n")
        beta_a = np.asarray(beta_aeh).ravel()
        beta_v = np.asarray(beta_voigt).ravel()
        beta_r = np.asarray(beta_reuss).ravel()
        beta_h = np.asarray(beta_hill).ravel()
        beta_g = np.asarray(beta_geo).ravel()
        for i in range(6):
            f.write(f"| β_{i+1}     | {beta_a[i]:.4e} | {beta_v[i]:.4e} | "
                    f"{beta_r[i]:.4e} | {beta_h[i]:.4e} | {beta_g[i]:.4e} |\n")
        f.write("\n")

        # Wave speeds
        f.write(f"### Wave Speeds (ρ = {rho:.1f} kg/m³)\n\n")
        f.write(f"| {'Method':<10s} | {'AVP (%)':>10s} | {'AVS1 (%)':>10s} | {'AVS2 (%)':>10s} | {'MaxAVS (%)':>11s} |\n")
        f.write(f"|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*13}|\n")
        for name, ws in wave_results.items():
            f.write(f"| {name:<10s} | {ws['AVP']:>10.2f} | {ws['AVS1']:>10.2f} | "
                    f"{ws['AVS2']:>10.2f} | {ws['MaxAVS']:>11.2f} |\n")
        f.write("\n")


def _log_thermal_conductivity(log_path, kappa_aeh, kappa_voigt, kappa_reuss,
                              kappa_hill, kappa_geo):
    """
    Write all thermal conductivity results to log.md.

    Appends effective thermal conductivity matrices and a comparison
    table of diagonal terms in Markdown format.

    Parameters
    ----------
    log_path : str
        Path to log.md file.
    kappa_aeh : ndarray, shape (3, 3)
        AEH effective thermal conductivity (W/(m K)).
    kappa_voigt : ndarray, shape (3, 3)
        Voigt effective thermal conductivity (W/(m K)).
    kappa_reuss : ndarray, shape (3, 3)
        Reuss effective thermal conductivity (W/(m K)).
    kappa_hill : ndarray, shape (3, 3)
        Hill effective thermal conductivity (W/(m K)).
    kappa_geo : ndarray, shape (3, 3)
        Geometric mean effective thermal conductivity (W/(m K)).

    Returns
    -------
    None
        Results are appended to the log file.
    """
    with open(log_path, "a") as f:
        # AEH conductivity
        f.write("### AEH Effective Thermal Conductivity (W/(m K))\n\n")
        f.write("```\n")
        for row in range(3):
            f.write("  " + "  ".join(f"{kappa_aeh[row, col]:10.4f}" for col in range(3)) + "\n")
        f.write("```\n\n")

        # Comparison table (diagonal terms)
        f.write("### Thermal Conductivity Comparison — Diagonal Terms (W/(m K))\n\n")
        f.write("| Component | AEH | Voigt | Reuss | Hill | GeoMean |\n")
        f.write("|-----------|-----|-------|-------|------|---------|\n")
        labels = ['κ11', 'κ22', 'κ33']
        for i in range(3):
            f.write(f"| {labels[i]:<9s} | {kappa_aeh[i,i]:.4f} | {kappa_voigt[i,i]:.4f} | "
                    f"{kappa_reuss[i,i]:.4f} | {kappa_hill[i,i]:.4f} | {kappa_geo[i,i]:.4f} |\n")
        f.write("\n")
