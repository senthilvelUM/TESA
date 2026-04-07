"""
Stage 3: AEH Analysis — compute effective material properties.

Orchestrator that dispatches to the appropriate AEH-FE analyses based on
the job dictionary settings:
  - run_thermoelastic: True/False → thermo-elastic analysis (chi, psi → C, α, β)
  - run_heat_conduction: True/False → thermal conductivity analysis (phi → κ)
  - macro_mechanical_field_type: "none"/"stress"/"strain" → optional microscale stress/strain fields
  - macro_thermal_field_type: "none"/"temperature_gradient"/"heat_flux" → optional microscale heat flux fields

Called from run_tesa.py after load_ebsd() and create_mesh().
"""

import os
import time
import numpy as np


def run_analysis(ms, job, run_dir=None, log_path=None, settings=None):
    """
    Stage 3: Run AEH analysis to compute effective properties.

    Parameters
    ----------
    ms : Microstructure
        Must have mesh, phase properties, and correction matrices populated.
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
        Updated with effective properties and analysis results.
    """
    if settings is None:
        settings = {}

    # Verbosity settings
    vc = settings.get("verbose_console", "medium")
    vl = settings.get("verbose_log", "medium")
    console_on = vc in ("medium", "high")
    log_on = vl in ("medium", "high")

    # Analysis flags from job dictionary
    run_te = job.get("run_thermoelastic", True)
    run_tc = job.get("run_heat_conduction", True)

    # Determine if microscale fields should be computed
    # "none" means skip, "stress"/"strain" or "temperature_gradient"/"heat_flux" means compute
    macro_mechanical_field_type = job.get("macro_mechanical_field_type", "none")
    macro_thermal_field_type = job.get("macro_thermal_field_type", "none")
    compute_ss = macro_mechanical_field_type != "none"
    compute_hf = macro_thermal_field_type != "none"

    # Validation: field evaluation requires corresponding homogenization
    if compute_ss and not run_te:
        print("  WARNING: macro_mechanical_field_type requires run_thermoelastic=True — skipping field evaluation")
        compute_ss = False
    if compute_hf and not run_tc:
        print("  WARNING: macro_thermal_field_type requires run_heat_conduction=True — skipping field evaluation")
        compute_hf = False

    # Create analysis results subfolder
    analysis_dir = None
    if run_dir:
        analysis_dir = os.path.join(run_dir, "AEH_characteristic_functions")
        os.makedirs(analysis_dir, exist_ok=True)

    # Determine what will be run for the banner
    analyses = []
    if run_te:
        analyses.append("Thermo-elastic")
    if run_tc:
        analyses.append("Thermal conductivity")
    analysis_label = " + ".join(analyses) if analyses else "None"

    # Print Stage 3 header
    print(f"\n── Stage 3: AEH Analysis ({analysis_label}) ──")

    # Write Stage 3 header to log
    if log_on and log_path:
        with open(log_path, "a") as f:
            f.write(f"\n## Stage 3 — AEH Analysis ({analysis_label})\n\n")

    t0 = time.time()

    # ── Thermo-elastic analysis ────────────────────────────────────────
    if run_te:
        ms = _run_thermoelastic(ms, job, run_dir, log_path, settings, analysis_dir,
                                compute_fields=compute_ss)

    # ── Thermal conductivity analysis ──────────────────────────────────
    if run_tc:
        ms = _run_heat_conduction(ms, job, run_dir, log_path, settings, analysis_dir,
                                       compute_fields=compute_hf)

    # ── Neither analysis requested ─────────────────────────────────────
    if not run_te and not run_tc:
        print("  No analysis requested (run_thermoelastic=False, run_heat_conduction=False)")

    elapsed = time.time() - t0
    print(f"\n  Stage 3 complete ({elapsed:.1f}s)")

    # Log completion time
    if log_on and log_path:
        with open(log_path, "a") as f:
            f.write(f"\n**Stage 3 elapsed time:** {elapsed:.1f}s\n\n")

    return ms


# ═══════════════════════════════════════════════════════════════════════════
# Thermo-elastic analysis branch
# ═══════════════════════════════════════════════════════════════════════════

def _run_thermoelastic(ms, job, run_dir, log_path, settings, analysis_dir=None,
                       compute_fields=False):
    """
    Run thermo-elastic AEH analysis with VRH bounds and wave speeds.

    Performs AEH-FE thermo-elastic analysis to compute effective stiffness,
    thermal expansion, and stress-temperature moduli. Also computes
    Voigt-Reuss-Hill bounds, geometric mean estimates, and optionally
    wave speeds and microscale stress/strain fields.

    Parameters
    ----------
    ms : Microstructure
        Must have mesh, phase properties, and correction matrices populated.
    job : dict
        Job configuration dictionary with keys such as
        'wave_speed_fields', 'macro_mechanical_field',
        'macro_temperature_field', 'macro_mechanical_field_type'.
    run_dir : str or None
        Results directory for output files.
    log_path : str or None
        Path to log.md file for logging results.
    settings : dict
        Global settings dictionary (verbose_console, verbose_log, etc.).
    analysis_dir : str or None
        Subdirectory for characteristic function output.
    compute_fields : bool
        If True, compute microscale stress/strain fields.

    Returns
    -------
    ms : Microstructure
        Updated with effective properties, VRH bounds, geometric mean,
        wave speeds, and optionally microscale stress/strain fields.
    """
    from .aehfe_thermoelastic_analysis import aehfe_thermoelastic_analysis
    from .get_voigt_reuss_hill import get_voigt_reuss_hill
    from .get_voigt_reuss_hill_thermal_expansion import get_voigt_reuss_hill_thermal_expansion
    from .get_geometric_mean import get_geometric_mean
    from .get_geometric_mean_thermal_expansion import get_geometric_mean_thermal_expansion
    from .get_wave_speeds import get_wave_speeds
    from .get_aeh import engineering_properties

    vc = settings.get("verbose_console", "medium")
    vl = settings.get("verbose_log", "medium")
    console_on = vc in ("medium", "high")
    log_on = vl in ("medium", "high")

    # ── Step 1: AEH-FE analysis — solve chi, psi → effective C, α, β ──
    if console_on:
        print("  ---- AEH-FE thermo-elastic analysis ----")
    ms = aehfe_thermoelastic_analysis(ms, verbose=(vc == "high"))

    # Extract AEH results
    D_aeh = ms.DEffectiveAEH
    alpha_aeh = ms.alphaEffectiveAEH
    beta_aeh = ms.betaEffectiveAEH

    # Compute engineering properties from AEH stiffness
    props_aeh = engineering_properties(D_aeh)
    ms.EngineeringProperties = props_aeh

    # Print AEH results
    if console_on:
        _print_stiffness("AEH", D_aeh)
        _print_thermal_expansion("AEH", alpha_aeh)
        _print_beta("AEH", beta_aeh)
        _print_engineering_props("AEH", props_aeh)

    # Common data for bounds calculations
    euler_data = ms.OriginalDataEulerAngle
    phase_data = ms.OriginalDataPhase
    phase_C = ms.PhaseStiffnessMatrix
    phase_alpha = ms.PhaseThermalExpansionMatrix
    MStar = ms.EBSDCorrectionMatrix
    theta_rad = ms.EBSDCorrectionAngle

    # ── Step 2: Voigt-Reuss-Hill bounds ────────────────────────────────
    if console_on:
        print("  ---- Voigt-Reuss-Hill bounds ----")
    D_voigt, D_reuss, D_hill = get_voigt_reuss_hill(
        euler_data, phase_data, phase_C, MStar)

    # Store on ms
    ms.DEffectiveVoigt = D_voigt
    ms.DEffectiveReuss = D_reuss
    ms.DEffectiveHill = D_hill

    if console_on:
        _print_stiffness("Voigt", D_voigt)
        _print_stiffness("Reuss", D_reuss)
        _print_stiffness("Hill", D_hill)

    # ── Step 3: VRH thermal expansion bounds ───────────────────────────
    alpha_voigt, alpha_reuss, alpha_hill = \
        get_voigt_reuss_hill_thermal_expansion(
            euler_data, phase_data, phase_C, phase_alpha, MStar)

    ms.alphaEffectiveVoigt = alpha_voigt
    ms.alphaEffectiveReuss = alpha_reuss
    ms.alphaEffectiveHill = alpha_hill

    # Compute beta = C @ alpha for VRH methods
    beta_voigt = D_voigt @ alpha_voigt
    beta_reuss = D_reuss @ alpha_reuss
    beta_hill  = 0.5 * (beta_voigt + beta_reuss)
    ms.betaEffectiveVoigt = beta_voigt
    ms.betaEffectiveReuss = beta_reuss
    ms.betaEffectiveHill  = beta_hill

    if console_on:
        _print_thermal_expansion("Voigt", alpha_voigt)
        _print_beta("Voigt", beta_voigt)
        _print_thermal_expansion("Reuss", alpha_reuss)
        _print_beta("Reuss", beta_reuss)
        _print_thermal_expansion("Hill", alpha_hill)
        _print_beta("Hill", beta_hill)

    # ── Step 4: Geometric mean stiffness ───────────────────────────────
    if console_on:
        print("  ---- Geometric mean ----")
    D_geo = get_geometric_mean(euler_data, phase_data, phase_C, MStar)
    ms.DEffectiveGeoMean = D_geo

    if console_on:
        _print_stiffness("GeoMean", D_geo)

    # ── Step 5: Geometric mean thermal expansion ───────────────────────
    alpha_geo = get_geometric_mean_thermal_expansion(
        euler_data, phase_data, phase_C, phase_alpha, MStar, theta_rad, D_geo)
    ms.alphaEffectiveGeoMean = alpha_geo

    # Compute beta = C @ alpha for geometric mean
    beta_geo = D_geo @ alpha_geo
    ms.betaEffectiveGeoMean = beta_geo

    if console_on:
        _print_thermal_expansion("GeoMean", alpha_geo)
        _print_beta("GeoMean", beta_geo)

    # ── Step 6: Wave speeds (optional) ──────────────────────────────────
    ws_fields = job.get("wave_speed_fields", "none")
    compute_ws = ws_fields != "none"
    if compute_ws:
        if console_on:
            print("  ---- Wave speeds ----")
        rho = ms.HomogenizedDensity
        methods = {
            "AEH": D_aeh,
            "Voigt": D_voigt,
            "Reuss": D_reuss,
            "Hill": D_hill,
            "GeoMean": D_geo,
        }
        wave_results = {}
        for name, D in methods.items():
            VS_dict, AVP, AVS1, AVS2, AVSH, AVSV, MaxAVS, v_raw, ev = get_wave_speeds(D, rho)
            wave_results[name] = {
                "VS": VS_dict, "AVP": AVP, "AVS1": AVS1, "AVS2": AVS2,
                "AVSH": AVSH, "AVSV": AVSV, "MaxAVS": MaxAVS,
                "v_raw": v_raw, "ev": ev,
            }
            if console_on:
                _print_wave_speeds(name, AVP, AVS1, AVS2, MaxAVS, rho)

        # Store wave results on ms for later use
        ms.WaveSpeedResults = wave_results

        # ── Step 6b: Per-phase single-crystal wave speeds ──────────────
        if console_on:
            print("  ---- Single-crystal wave speeds (per phase) ----")
        phase_wave_results = {}
        for i in range(ms.NumberPhases):
            C_phase = ms.PhaseStiffnessMatrix[i]
            rho_phase = ms.PhaseDensity[i]
            phase_name = ms.PhaseName[i] if ms.PhaseName else f"Phase_{i+1}"
            # Skip phases with missing stiffness or invalid density
            if C_phase is None or rho_phase <= 0:
                if console_on:
                    print(f"    Skipping {phase_name}: missing stiffness or density")
                continue
            VS_dict, AVP, AVS1, AVS2, AVSH, AVSV, MaxAVS, v_raw, ev = get_wave_speeds(C_phase, rho_phase)
            folder_key = f"phase_{i+1}_{phase_name.replace(' ', '_')}"
            phase_wave_results[folder_key] = {
                "VS": VS_dict, "AVP": AVP, "AVS1": AVS1, "AVS2": AVS2,
                "AVSH": AVSH, "AVSV": AVSV, "MaxAVS": MaxAVS,
                "v_raw": v_raw, "ev": ev,
            }
            if console_on:
                _print_wave_speeds(f"{phase_name} (crystal)", AVP, AVS1, AVS2, MaxAVS, rho_phase)
        ms.PhaseWaveSpeedResults = phase_wave_results
    else:
        ms.WaveSpeedResults = {}
        ms.PhaseWaveSpeedResults = {}
        if console_on:
            print("  ---- Wave speeds (skipped) ----")

    # ── Log all thermo-elastic results ─────────────────────────────────
    # ── Optional microscale stress/strain fields ─────────────────────
    if compute_fields:
        macro_mech = np.array(job.get("macro_mechanical_field", [0, 0, 0, 0, 0, 0]), dtype=float)
        macro_temp = float(job.get("macro_temperature_field", 0))
        macro_type = job.get("macro_mechanical_field_type", "stress")

        # Check if any loading is specified
        if np.allclose(macro_mech, 0) and macro_temp == 0:
            print("  WARNING: macro_mechanical_field and macro_temperature_field are all zero "
                  "— skipping field evaluation")
        else:
            if console_on:
                print(f"  ---- Microscale stress/strain fields ({macro_type}) ----")

            # Construct macroscaleLoadInfo: [type, s1, s2, s3, s4, s5, s6, ΔT]
            load_type = 1 if macro_type == "stress" else 2
            ms.macroscaleLoadInfo = np.array([load_type, *macro_mech, macro_temp])

            # Compute localization tensors DHat and betaHat from characteristic functions
            if console_on:
                print("  Computing localization tensors (DHat, betaHat)...")
            from .compute_aeh_stress_strain_info import compute_aeh_stress_strain_info
            DHat, betaHat = compute_aeh_stress_strain_info(
                ms.SixNodeElementIndexList,
                ms.quadraturePointElasticStiffnesses,
                ms.quadraturePointStressTemperatureModuli,
                ms.strainDisplacementMatrix,
                ms.combinedCharacteristicFunctions)
            ms.DHat = DHat
            ms.betaHat = betaHat

            # Compute microscale stress and strain fields
            if console_on:
                print("  Computing microscale stress/strain fields...")
            from .perform_aehfe_thermoelastic_stress_strain_analysis import \
                perform_aehfe_thermoelastic_stress_strain_analysis
            ms = perform_aehfe_thermoelastic_stress_strain_analysis(ms)

            # Print field summary
            if console_on and hasattr(ms, 'Microfield') and ms.Microfield is not None:
                mf = ms.Microfield
                # Stress components
                # Stress components (displayed in MPa)
                labels_s = ['σ₁₁', 'σ₂₂', 'σ₃₃', 'σ₂₃', 'σ₁₃', 'σ₁₂']
                print("  Microscale stress summary (MPa):")
                for idx, lbl in zip(range(2, 8), labels_s):
                    vals = mf[idx] * 1e-6  # Pa → MPa
                    print(f"    {lbl}: min={np.min(vals):.4f}, max={np.max(vals):.4f}, mean={np.mean(vals):.4f}")
                # Strain components (displayed in microstrain)
                labels_e = ['ε₁₁', 'ε₂₂', 'ε₃₃', 'ε₂₃', 'ε₁₃', 'ε₁₂']
                print("  Microscale strain summary (microstrain):")
                for idx, lbl in zip(range(13, 19), labels_e):
                    vals = mf[idx] * 1e6  # dimensionless → microstrain
                    print(f"    {lbl}: min={np.min(vals):.4f}, max={np.max(vals):.4f}, mean={np.mean(vals):.4f}")

    return ms


# ═══════════════════════════════════════════════════════════════════════════
# Thermal conductivity analysis branch
# ═══════════════════════════════════════════════════════════════════════════

def _run_heat_conduction(ms, job, run_dir, log_path, settings, analysis_dir=None,
                              compute_fields=False):
    """
    Run thermal conductivity AEH analysis with VRH bounds.

    Performs AEH-FE thermal conductivity analysis to compute effective
    thermal conductivity tensor, along with Voigt-Reuss-Hill bounds
    and geometric mean estimates. Optionally computes microscale heat
    flux and temperature gradient fields.

    Parameters
    ----------
    ms : Microstructure
        Must have mesh, phase properties, and correction matrices populated.
    job : dict
        Job configuration dictionary with keys such as
        'macro_thermal_field', 'macro_thermal_field_type'.
    run_dir : str or None
        Results directory for output files.
    log_path : str or None
        Path to log.md file for logging results.
    settings : dict
        Global settings dictionary (verbose_console, verbose_log, etc.).
    analysis_dir : str or None
        Subdirectory for characteristic function output.
    compute_fields : bool
        If True, compute microscale heat flux/temperature gradient fields.

    Returns
    -------
    ms : Microstructure
        Updated with effective thermal conductivity, VRH bounds,
        geometric mean, and optionally microscale heat flux fields.
    """
    from .aehfe_thermal_conductivity_analysis import aehfe_thermal_conductivity_analysis
    from .compute_macroscale_voigt_reuss_hill_thermal_conductivity import \
        compute_macroscale_voigt_reuss_hill_thermal_conductivity
    from .compute_macroscale_geometric_mean_thermal_conductivity import \
        compute_macroscale_geometric_mean_thermal_conductivity

    vc = settings.get("verbose_console", "medium")
    vl = settings.get("verbose_log", "medium")
    console_on = vc in ("medium", "high")
    log_on = vl in ("medium", "high")

    # ── Step 1: AEH-FE analysis — solve phi → effective κ ──────────────
    if console_on:
        print("  ---- AEH-FE thermal conductivity analysis ----")
    ms = aehfe_thermal_conductivity_analysis(ms, verbose=(vc == "high"))

    # Extract AEH result
    kappa_aeh = ms.kappaEffectiveAEH

    if console_on:
        _print_conductivity("AEH", kappa_aeh)

    # Common data for bounds
    euler_data = ms.OriginalDataEulerAngle
    phase_data = ms.OriginalDataPhase
    phase_kappa = ms.PhaseThermalConductivityMatrix
    theta_rad = ms.EBSDCorrectionAngle

    # ── Step 2: VRH bounds for thermal conductivity ────────────────────
    if console_on:
        print("  ---- VRH bounds (thermal conductivity) ----")
    kappa_voigt, kappa_reuss, kappa_hill = \
        compute_macroscale_voigt_reuss_hill_thermal_conductivity(
            euler_data, phase_data, phase_kappa, theta_rad)

    ms.kappaEffectiveVoigt = kappa_voigt
    ms.kappaEffectiveReuss = kappa_reuss
    ms.kappaEffectiveHill = kappa_hill

    if console_on:
        _print_conductivity("Voigt", kappa_voigt)
        _print_conductivity("Reuss", kappa_reuss)
        _print_conductivity("Hill", kappa_hill)

    # ── Step 3: Geometric mean for thermal conductivity ────────────────
    if console_on:
        print("  ---- Geometric mean (thermal conductivity) ----")
    kappa_geo = compute_macroscale_geometric_mean_thermal_conductivity(
        euler_data, phase_data, phase_kappa, theta_rad)
    ms.kappaEffectiveGeoMean = kappa_geo

    if console_on:
        _print_conductivity("GeoMean", kappa_geo)

    # ── Optional microscale heat flux fields ─────────────────────────
    if compute_fields:
        macro_thermal = np.array(job.get("macro_thermal_field", [0, 0, 0]), dtype=float)
        macro_thermal_field_type = job.get("macro_thermal_field_type", "temperature_gradient")

        # Check if any loading is specified
        if np.allclose(macro_thermal, 0):
            print("  WARNING: macro_thermal_field is all zero "
                  "— skipping field evaluation")
        else:
            if console_on:
                print(f"  ---- Microscale heat flux fields ({macro_thermal_field_type}) ----")
            from .perform_aehfe_heat_flux_temp_grad_analysis import \
                perform_aehfe_heat_flux_temp_grad_analysis
            ms = perform_aehfe_heat_flux_temp_grad_analysis(
                ms, macro_field=macro_thermal, field_type=macro_thermal_field_type,
                verbose=(vc == "high"))
            if console_on:
                print(f"  Microscale heat flux fields computed for {macro_thermal_field_type} loading")

    return ms


# ═══════════════════════════════════════════════════════════════════════════
# Console printing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _print_stiffness(method_name, D):
    """
    Print a 6x6 stiffness matrix to console in GPa.

    Parameters
    ----------
    method_name : str
        Label for the homogenization method (e.g. 'AEH', 'Voigt').
    D : ndarray, shape (6, 6)
        Stiffness matrix in Pa.
    """
    print(f"  Effective stiffness C_{method_name} (GPa):")
    for row in range(6):
        print(f"    " + "  ".join(f"{D[row, col]/1e9:10.4f}" for col in range(6)))

def _print_thermal_expansion(method_name, alpha):
    """
    Print thermal expansion vector to console.

    Parameters
    ----------
    method_name : str
        Label for the homogenization method.
    alpha : ndarray, shape (6,) or (6, 1)
        Thermal expansion coefficients in 1/K.
    """
    alpha_flat = np.asarray(alpha).ravel()
    print(f"  Effective thermal expansion alpha_{method_name} (1/K):")
    for i in range(6):
        print(f"    alpha_{i+1} = {alpha_flat[i]:.6e}")

def _print_beta(method_name, beta):
    """
    Print stress-temperature modulus vector to console.

    Parameters
    ----------
    method_name : str
        Label for the homogenization method.
    beta : ndarray, shape (6,) or (6, 1)
        Stress-temperature moduli in Pa/K.
    """
    beta_flat = np.asarray(beta).ravel()
    print(f"  Effective stress-temperature modulus beta_{method_name} (Pa/K):")
    for i in range(6):
        print(f"    beta_{i+1} = {beta_flat[i]:.6e}")

def _print_engineering_props(method_name, props):
    """
    Print engineering properties to console.

    Parameters
    ----------
    method_name : str
        Label for the homogenization method.
    props : dict
        Dictionary with keys 'E1', 'E2', 'E3', 'G23', 'G13', 'G12',
        'nu12', 'nu13', 'nu21', 'nu23', 'nu31', 'nu32'. Moduli in Pa.
    """
    print(f"  Engineering properties ({method_name}):")
    print(f"    E1={props['E1']/1e9:.4f}  E2={props['E2']/1e9:.4f}  E3={props['E3']/1e9:.4f} GPa")
    print(f"    G23={props['G23']/1e9:.4f}  G13={props['G13']/1e9:.4f}  G12={props['G12']/1e9:.4f} GPa")
    print(f"    nu12={props['nu12']:.6f}  nu13={props['nu13']:.6f}  nu23={props['nu23']:.6f}")
    print(f"    nu21={props['nu21']:.6f}  nu31={props['nu31']:.6f}  nu32={props['nu32']:.6f}")

def _print_wave_speeds(method_name, AVP, AVS1, AVS2, MaxAVS, rho):
    """
    Print wave speed anisotropy summary to console.

    Parameters
    ----------
    method_name : str
        Label for the homogenization method.
    AVP : float
        P-wave anisotropy percentage.
    AVS1 : float
        S1-wave anisotropy percentage.
    AVS2 : float
        S2-wave anisotropy percentage.
    MaxAVS : float
        Maximum shear-wave splitting percentage.
    rho : float
        Density in kg/m^3.
    """
    print(f"  Wave speeds ({method_name}, rho={rho:.1f} kg/m³):")
    print(f"    P-wave anisotropy  (AVP)  : {AVP:.2f}%")
    print(f"    S1-wave anisotropy (AVS1) : {AVS1:.2f}%")
    print(f"    S2-wave anisotropy (AVS2) : {AVS2:.2f}%")
    print(f"    Max shear splitting       : {MaxAVS:.2f}%")

def _print_conductivity(method_name, kappa):
    """
    Print a 3x3 thermal conductivity matrix to console.

    Parameters
    ----------
    method_name : str
        Label for the homogenization method.
    kappa : ndarray, shape (3, 3)
        Thermal conductivity matrix in W/(m K).
    """
    print(f"  Effective conductivity kappa_{method_name} (W/(m K)):")
    for row in range(3):
        print(f"    " + "  ".join(f"{kappa[row, col]:10.4f}" for col in range(3)))


