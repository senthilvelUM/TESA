"""
Compute microscale heat flux and temperature gradient fields from AEH results.

Uses the localization tensor kappaHat to compute local heat flux at each
quadrature point given a macroscale temperature gradient (or heat flux).
Also computes the microscale temperature gradient from the heat flux and
the magnitude of both fields.

The MicrofieldHeatConduction output stores (per quadrature point):
  [0]  qx          — x coordinate
  [1]  qy          — y coordinate
  [2-4] q1..q3     — heat flux components
  [5]  |q|         — heat flux magnitude
  [6-8] dT1..dT3   — temperature gradient components
  [9]  |dT|        — temperature gradient magnitude

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np


def perform_aehfe_heat_flux_temp_grad_analysis(ms, macro_field=None,
                                                field_type="temperature_gradient",
                                                verbose=False):
    """
    Compute microscale heat flux/temperature gradient fields.

    Parameters
    ----------
    ms : Microstructure
        Must have kappaHat, quadraturePointThermalConductivity,
        kappaEffectiveAEH, and quadraturePointCoordinates populated.
    macro_field : array-like (3,) or None
        Macroscale loading vector. If None, reads from
        ms.macroscaleLoadInfoHeatConduction.
    field_type : str
        "temperature_gradient" (type 1) or "heat_flux" (type 2).
        Determines how macro_field is interpreted.
    verbose : bool
        Print progress to console.

    Returns
    -------
    ms : Microstructure
        Updated with ms.MicrofieldHeatConduction containing heat flux
        and temperature gradient at quadrature points.
    """
    # Setup variables
    kappaHat = ms.kappaHat
    kappaQuad = ms.quadraturePointThermalConductivity
    kappaEffective = ms.kappaEffectiveAEH

    # Determine analysis type and macro field
    # type 1 = temperature gradient input, type 2 = heat flux input
    if macro_field is not None:
        macroField = np.asarray(macro_field, dtype=float).reshape(3, 1)
        analysisType = 1 if field_type == "temperature_gradient" else 2
        # Store on ms for downstream use
        ms.macroscaleLoadInfoHeatConduction = np.array(
            [analysisType, *macroField.ravel()])
    else:
        # Read from ms (legacy path)
        macroField = np.asarray(
            ms.macroscaleLoadInfoHeatConduction[1:4], dtype=float).reshape(3, 1)
        analysisType = int(ms.macroscaleLoadInfoHeatConduction[0])

    nElements = kappaQuad[0].shape[2]
    nQP = len(kappaQuad)  # 4

    if verbose:
        type_str = "temperature gradient" if analysisType == 1 else "heat flux"
        print(f"  Analysis type: {type_str}")
        print(f"  Macro field: {macroField.ravel()}")

    # If heat flux analysis (type=2), compute temp grad from heat flux
    # If temp grad analysis (type=1), use directly
    if analysisType == 2:
        macroTempGrad = -np.linalg.solve(kappaEffective, np.eye(3)) @ macroField
    else:
        macroTempGrad = macroField

    # Get quadrature point coordinates
    qxyCell = ms.quadraturePointCoordinates
    if isinstance(qxyCell, np.ndarray):
        qx = qxyCell[:, 0]
        qy = qxyCell[:, 1]
    else:
        # Use element-major ordering to match the field value ordering
        # (hf_comp[iQP::nQP] = ...) which interleaves QPs within each element:
        # [QP0_e0, QP1_e0, QP2_e0, QP3_e0, QP0_e1, QP1_e1, ...]
        qx = np.zeros(nElements * nQP)
        qy = np.zeros(nElements * nQP)
        for iQP in range(nQP):
            qx[iQP::nQP] = qxyCell[iQP][:, 0]
            qy[iQP::nQP] = qxyCell[iQP][:, 1]

    nQuadTotal = len(qx)

    if verbose:
        print(f"  Quadrature points: {nQuadTotal} ({nElements} elements × {nQP} QPs)")

    # Initialize MicrofieldHeatConduction: list of 10 arrays
    MF = [None] * 10
    MF[0] = qx.ravel().copy()
    MF[1] = qy.ravel().copy()

    # Compute heat flux: q = -kappaHat @ macroTempGrad
    tmp = [None] * nQP
    for iQP in range(nQP):
        # -kappaHat{iQP} @ macroTempGrad: (3, 3, nE) @ (3, 1) -> (3, 1, nE)
        tmp[iQP] = -np.einsum('ijk,jl->ilk', kappaHat[iQP], macroTempGrad)

    # Store heat flux components (cells 2-4)
    for jComp in range(3):
        hf_comp = np.zeros(nQuadTotal)
        for iQP in range(nQP):
            hf_comp[iQP::nQP] = tmp[iQP][jComp, 0, :]
        MF[2 + jComp] = hf_comp

    # Compute heat flux magnitude (cell 5)
    MF[5] = np.sqrt(MF[2] ** 2 + MF[3] ** 2 + MF[4] ** 2)

    if verbose:
        print(f"  Heat flux range: [{np.min(MF[5]):.6e}, {np.max(MF[5]):.6e}] W/m²")

    # Compute microscale temperature gradient: dT = -inv(kappa) @ q (vectorized)
    # Build batched (nQuadTotal, 3, 3) conductivity array and (nQuadTotal, 3) heat flux RHS
    _kappa_all = np.zeros((nQuadTotal, 3, 3))
    _hf_all = np.column_stack([MF[2], MF[3], MF[4]])  # (nQuadTotal, 3)
    for iQP in range(nQP):
        # kappaQuad[iQP] is (3, 3, nElements) — transpose to (nElements, 3, 3)
        _kappa_all[iQP::nQP] = kappaQuad[iQP].transpose(2, 0, 1)
    # Batch solve: -inv(kappa) @ q = -solve(kappa, q)
    _microTempGrad_all = -np.linalg.solve(_kappa_all, _hf_all[:, :, np.newaxis])[:, :, 0]
    MF[6] = _microTempGrad_all[:, 0]
    MF[7] = _microTempGrad_all[:, 1]
    MF[8] = _microTempGrad_all[:, 2]

    # Compute temperature gradient magnitude (cell 9)
    MF[9] = np.sqrt(MF[6] ** 2 + MF[7] ** 2 + MF[8] ** 2)

    if verbose:
        print(f"  Temp gradient range: [{np.min(MF[9]):.6e}, {np.max(MF[9]):.6e}] K/m")

    # Store results
    ms.MicrofieldHeatConduction = MF

    return ms
