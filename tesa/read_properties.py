"""
Read crystal phase property files and store on Microstructure object.

Property files use Abaqus-style *keyword headers to mark each section.
Comment lines start with #. The parser searches for these keywords
(case-insensitive):

  *phase              → phase name string
  *density                 → single value (kg/m³)
  *stiffness_matrix        → 6×6 matrix (Pa)
  *thermal_expansion       → 6 values (1/K)
  *thermal_conductivity    → 3×3 matrix (W/(m K))

Example property file:
    *phase
    alpha-Quartz

    *density
    2650

    *stiffness_matrix
    88.2e+9  6.5e+9  12.4e+9  18.8e+9  0.0  0.0
    ...

    *thermal_expansion
    5.35e-5
    ...

    *thermal_conductivity
    1.0  0.0  0.0
    ...

    # Units
    # density: kg/m^3
    # stiffness_matrix: Pa

    # References
    # Ohno et al. (2006) ...

"""

import os
import numpy as np


def _find_section(lines, keyword):
    """
    Find the *keyword header line and return the data lines that follow.

    Data lines are non-blank, non-comment, non-keyword lines that appear
    after the header until the next *keyword, comment, or end of file.

    Parameters
    ----------
    lines : list of str
        All lines from the file (already stripped).
    keyword : str
        Keyword to search for (without * prefix), e.g. "density".

    Returns
    -------
    data_lines : list of str
        The data lines following the header, or empty list if not found.
    """
    keyword_lower = keyword.lower()
    found = False
    data_lines = []

    for line in lines:
        stripped = line.strip()

        if found:
            # A new *keyword marks end of section
            if stripped.startswith('*'):
                break
            # Skip blank lines and comment lines
            if stripped == '' or stripped.startswith('#'):
                # If we already collected data, a comment block ends the section
                if data_lines and stripped.startswith('#'):
                    break
                continue
            # This is a data line
            data_lines.append(stripped)
        else:
            # Look for a *keyword line matching the target
            if stripped.startswith('*') and stripped[1:].strip().lower() == keyword_lower:
                found = True

    return data_lines


def read_properties(ms, phase_properties, log_path=None, settings=None):
    """
    Parse crystal property files and populate ms with phase data.

    Uses Abaqus-style *keyword headers — the parser searches for lines
    starting with * followed by the keyword name. Comment lines (#) and
    blank lines are ignored.

    Parameters
    ----------
    ms : Microstructure
        Must have NumberPhases and phase arrays initialized.
    phase_properties : dict
        Mapping of phase number (1-indexed) to file path.
        Example: {1: "property_files/Quartz.txt",
                  2: "property_files/Plagioclase.txt"}
    log_path : str or None
        Path to log.md for writing results.
    settings : dict or None
        Global settings (verbose_console, verbose_log).

    Returns
    -------
    ms : Microstructure
        Updated with phase property data.
    """
    # Unpack settings
    if settings is None:
        settings = {}
    vc = settings.get("verbose_console", "medium")
    vl = settings.get("verbose_log", "medium")

    if vc in ("medium", "high"):
        print("\n── Load phase properties ──")

    # Ensure phase arrays are initialized
    n_phases = ms.NumberPhases
    if ms.PhaseThermalExpansionMatrix is None:
        ms.PhaseThermalExpansionMatrix = [None] * n_phases
    if ms.PhaseThermalConductivityMatrix is None:
        ms.PhaseThermalConductivityMatrix = [None] * n_phases
    # Convert PhaseDensity to list so it can hold None values
    if isinstance(ms.PhaseDensity, np.ndarray):
        ms.PhaseDensity = list(ms.PhaseDensity)

    # Log header
    log_lines = []
    if vl in ("medium", "high"):
        log_lines.append("## Phase Properties\n\n")

    # Parse each phase property file
    for ph_num, ph_file in sorted(phase_properties.items()):
        # Convert to 0-indexed for ms arrays
        idx = ph_num - 1
        if idx < 0 or idx >= n_phases:
            print(f"  WARNING: Phase {ph_num} out of range (1–{n_phases}), skipping")
            continue

        # Store the filename
        ms.PhasePropertyFilename[idx] = ph_file

        if not os.path.isfile(ph_file):
            print(f"  ERROR: Property file not found: {ph_file}")
            continue

        try:
            # Read all lines (handle non-UTF-8 characters like degree symbols)
            with open(ph_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = [line.rstrip() for line in f.readlines()]

            # Track which keywords fell back to defaults
            _defaults_used = []

            # ── Phase name: *phase keyword ──
            name_lines = _find_section(lines, "phase")
            if name_lines:
                phase = name_lines[0].strip()
            else:
                # Default to filename without extension
                phase = os.path.splitext(os.path.basename(ph_file))[0]
                _defaults_used.append(f"phase (using filename: {phase})")
            ms.PhaseName[idx] = phase

            # ── Density: *density keyword ──
            density_lines = _find_section(lines, "density")
            if density_lines:
                density = float(density_lines[0].split()[0])
                ms.PhaseDensity[idx] = density
            else:
                density = 1.0
                ms.PhaseDensity[idx] = density
                _defaults_used.append(f"density (using default: {density} kg/m³)")

            # ── Stiffness matrix: *stiffness_matrix keyword ──
            stiffness_lines = _find_section(lines, "stiffness_matrix")
            if len(stiffness_lines) >= 6:
                stiffness = np.zeros((6, 6))
                for row in range(6):
                    vals = [float(v) for v in stiffness_lines[row].split()]
                    stiffness[row, :len(vals)] = vals
                ms.PhaseStiffnessMatrix[idx] = stiffness
            else:
                # Default to 6×6 identity matrix
                stiffness = np.eye(6)
                ms.PhaseStiffnessMatrix[idx] = stiffness
                _defaults_used.append("stiffness_matrix (using 6×6 identity)")

            # ── Thermal expansion: *thermal_expansion keyword ──
            alpha_lines = _find_section(lines, "thermal_expansion")
            if alpha_lines:
                alpha = np.zeros((6, 1))
                # Values may be one per line or all on one line
                all_vals = []
                for line in alpha_lines:
                    all_vals.extend([float(v) for v in line.split()])
                for i, v in enumerate(all_vals[:6]):
                    alpha[i, 0] = v
                ms.PhaseThermalExpansionMatrix[idx] = alpha
            else:
                # Default to zeros
                alpha = np.zeros((6, 1))
                ms.PhaseThermalExpansionMatrix[idx] = alpha
                _defaults_used.append("thermal_expansion (using zeros)")

            # ── Thermal conductivity: *thermal_conductivity keyword ──
            kappa_lines = _find_section(lines, "thermal_conductivity")
            if len(kappa_lines) >= 3:
                kappa = np.zeros((3, 3))
                for row in range(3):
                    vals = [float(v) for v in kappa_lines[row].split()]
                    kappa[row, :len(vals)] = vals
                ms.PhaseThermalConductivityMatrix[idx] = kappa
            else:
                # Default to 3×3 identity matrix
                kappa = np.eye(3)
                ms.PhaseThermalConductivityMatrix[idx] = kappa
                _defaults_used.append("thermal_conductivity (using 3×3 identity)")

            # ── Print default warnings to console ──
            if _defaults_used and vc in ("medium", "high"):
                for d in _defaults_used:
                    print(f"  WARNING: Phase {ph_num} ({phase}): "
                          f"*{d}")

            # ── Validate stiffness matrix symmetry ──
            if stiffness is not None:
                sym_diff = np.max(np.abs(stiffness - stiffness.T))
                sym_ok = sym_diff < 1e-6 * np.max(np.abs(stiffness))
            else:
                sym_ok = False

            # Build set of defaulted keyword names for display logic
            _defaulted = set(d.split()[0] for d in _defaults_used)

            # ── Console output ──
            if vc in ("medium", "high"):
                print(f"  Phase {ph_num}: {phase if phase else 'MISSING'}")
                print(f"    File    : {os.path.basename(ph_file)}")
                _dflt = " (default)" if "density" in _defaulted else ""
                print(f"    Density : {density:.1f} kg/m³{_dflt}")
                _dflt = " (default)" if "stiffness_matrix" in _defaulted else ""
                if "stiffness_matrix" in _defaulted:
                    print(f"    Cij     : 6×6 identity{_dflt}")
                else:
                    print(f"    Cij     : 6×6 matrix "
                          f"({'symmetric' if sym_ok else 'WARNING: not symmetric'})")
                alpha_nonzero = np.count_nonzero(alpha)
                if "thermal_expansion" in _defaulted:
                    print(f"    Alpha   : zeros (default)")
                else:
                    print(f"    Alpha   : {alpha_nonzero}/6 non-zero components")
                if "thermal_conductivity" in _defaulted:
                    print(f"    Kappa   : 3×3 identity (default)")
                else:
                    print(f"    Kappa   : 3×3 matrix")
                if "stiffness_matrix" not in _defaulted and not sym_ok:
                    print(f"    WARNING: Stiffness matrix max asymmetry = "
                          f"{sym_diff:.3e}")

            # ── Verbose console: print full matrices ──
            if vc == "high":
                if "stiffness_matrix" in _defaulted:
                    print(f"    Stiffness matrix: 6×6 identity (default)")
                else:
                    print(f"    Stiffness matrix (GPa):")
                    for row in range(6):
                        vals = "  ".join(
                            f"{stiffness[row, col]/1e9:10.3f}" for col in range(6))
                        print(f"      {vals}")
                if "thermal_expansion" in _defaulted:
                    print(f"    Thermal expansion: zeros (default)")
                else:
                    print(f"    Thermal expansion (1/K):")
                    for row in range(6):
                        print(f"      {alpha[row, 0]:12.4e}")
                if "thermal_conductivity" in _defaulted:
                    print(f"    Thermal conductivity: 3×3 identity (default)")
                else:
                    print(f"    Thermal conductivity (W/(m K)):")
                    for row in range(3):
                        vals = "  ".join(
                            f"{kappa[row, col]:8.3f}" for col in range(3))
                        print(f"      {vals}")

            # ── Log output ──
            if vl in ("medium", "high"):
                log_lines.append(f"### Phase {ph_num}: {phase}\n\n")
                log_lines.append(f"- **File:** `{os.path.basename(ph_file)}`\n")
                _dflt = " (default)" if "density" in _defaulted else ""
                log_lines.append(f"- **Density:** {density:.1f} kg/m³{_dflt}\n")
                if "stiffness_matrix" in _defaulted:
                    log_lines.append(f"- **Stiffness:** 6×6 identity (default)\n")
                else:
                    log_lines.append(f"- **Stiffness:** 6×6 matrix "
                                     f"({'symmetric' if sym_ok else 'WARNING: not symmetric'})\n")
                if "thermal_expansion" in _defaulted:
                    log_lines.append(f"- **Thermal expansion:** zeros (default)\n")
                else:
                    log_lines.append(f"- **Thermal expansion:** "
                                     f"{alpha_nonzero}/6 non-zero components\n")
                if "thermal_conductivity" in _defaulted:
                    log_lines.append(f"- **Thermal conductivity:** 3×3 identity (default)\n")
                else:
                    log_lines.append(f"- **Thermal conductivity:** 3×3 matrix\n")
                # Log any defaults that were used
                if _defaults_used:
                    log_lines.append(f"\n**Defaults used:**\n\n")
                    for d in _defaults_used:
                        log_lines.append(f"- ⚠ `*{d}`\n")
                log_lines.append("\n")

            # ── High verbosity log: include matrices ──
            if vl == "high":
                if "stiffness_matrix" in _defaulted:
                    log_lines.append("**Stiffness matrix:** 6×6 identity (default)\n\n")
                else:
                    log_lines.append("**Stiffness matrix (GPa):**\n\n```\n")
                    for row in range(6):
                        vals = "  ".join(
                            f"{stiffness[row, col]/1e9:10.3f}" for col in range(6))
                        log_lines.append(f"  {vals}\n")
                    log_lines.append("```\n\n")
                if "thermal_expansion" in _defaulted:
                    log_lines.append("**Thermal expansion:** zeros (default)\n\n")
                else:
                    log_lines.append("**Thermal expansion (1/K):**\n\n```\n")
                    for row in range(6):
                        log_lines.append(f"  {alpha[row, 0]:12.4e}\n")
                    log_lines.append("```\n\n")
                if "thermal_conductivity" in _defaulted:
                    log_lines.append("**Thermal conductivity:** 3×3 identity (default)\n\n")
                else:
                    log_lines.append(
                        "**Thermal conductivity (W/(m K)):**\n\n```\n")
                    for row in range(3):
                        vals = "  ".join(
                            f"{kappa[row, col]:8.3f}" for col in range(3))
                        log_lines.append(f"  {vals}\n")
                    log_lines.append("```\n\n")

        except Exception as e:
            print(f"  ERROR: Failed to parse {ph_file}: {e}")
            log_lines.append(f"### Phase {ph_num}\n\n")
            log_lines.append(f"- **ERROR:** Failed to parse "
                             f"`{os.path.basename(ph_file)}`: {e}\n\n")
            continue

    # Write log
    if log_path and log_lines:
        with open(log_path, "a") as f:
            for line in log_lines:
                f.write(line)

    return ms
