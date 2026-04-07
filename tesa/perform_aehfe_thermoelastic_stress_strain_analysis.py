"""
Compute microscale stress and strain fields from AEH results.

Uses the localization tensors DHat and betaHat to compute local stress
and strain fields at each quadrature point given macroscale strain (or stress)
and temperature change. Also computes principal stresses/strains, max shear,
and mechanical strains (total - thermal).

The Microfield output stores (per quadrature point):
  [0]  qx           — x coordinate
  [1]  qy           — y coordinate
  [2-7]  sigma_11..sigma_12  — stress components (Voigt)
  [8-10] sigma_p1..sigma_p3  — principal stresses (sorted)
  [11] tau_max       — max shear stress = 0.5*(sigma_p1 - sigma_p3)
  [12] tau_xy_max    — max in-plane shear
  [13-18] eps_11..eps_12  — strain components (Voigt)
  [19-21] eps_p1..eps_p3  — principal strains (sorted)
  [39-44] eps_mech_11..eps_mech_12  — mechanical strains (total - thermal)

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np


def perform_aehfe_thermoelastic_stress_strain_analysis(ms):
    """
    Compute microscale stress/strain fields from AEH results.

    Uses the localization tensors DHat and betaHat to compute local stress
    and strain fields at each quadrature point, given macroscale loading
    (stress or strain) and temperature change.

    Parameters
    ----------
    ms : Microstructure
        Must have the following attributes populated:
        DHat, betaHat, quadraturePointElasticStiffnesses,
        quadraturePointStressTemperatureModuli,
        quadraturePointThermalExpansionCoefficients,
        macroscaleLoadInfo, DEffectiveAEH, betaEffectiveAEH,
        and quadraturePointCoordinates.

    Returns
    -------
    ms : Microstructure
        Updated with ms.Microfield, a list of 50 arrays (one per field
        quantity), each of length nQuadTotal. Key indices:
        [0] qx, [1] qy, [2-7] stress components (Voigt),
        [8-10] principal stresses, [11] max shear stress,
        [12] max in-plane shear, [13-18] strain components (Voigt),
        [19-21] principal strains, [39-44] mechanical strains.
    """
    # Setup variables
    DHat = ms.DHat
    betaHat = ms.betaHat
    DQuad = ms.quadraturePointElasticStiffnesses
    betaQuad = ms.quadraturePointStressTemperatureModuli
    alphaQuad = ms.quadraturePointThermalExpansionCoefficients
    macroStrain = np.asarray(ms.macroscaleLoadInfo[1:7], dtype=float).reshape(6, 1)
    macroTemp = float(ms.macroscaleLoadInfo[7])
    stressStrainAnalysisType = int(ms.macroscaleLoadInfo[0])
    DEffective = ms.DEffectiveAEH
    betaEffective = ms.betaEffectiveAEH

    # If stress analysis, compute the strain from the macroscale stress
    if stressStrainAnalysisType == 1:
        macroStrain = np.linalg.solve(DEffective, np.eye(6)) @ (
            macroStrain + betaEffective * macroTemp)

    # Get quadrature point coordinates
    qxyCell = ms.quadraturePointCoordinates
    if isinstance(qxyCell, np.ndarray):
        qx = qxyCell[:, 0]
        qy = qxyCell[:, 1]
    else:
        # Element-major ordering to match field value interleaving
        # (stress_comp[iQP::nQP] = ...) which gives:
        # [QP0_e0, QP1_e0, QP2_e0, QP3_e0, QP0_e1, QP1_e1, ...]
        nE_local = qxyCell[0].shape[0]
        nQP_local = len(qxyCell)
        qx = np.zeros(nE_local * nQP_local)
        qy = np.zeros(nE_local * nQP_local)
        for iQP in range(nQP_local):
            qx[iQP::nQP_local] = qxyCell[iQP][:, 0]
            qy[iQP::nQP_local] = qxyCell[iQP][:, 1]

    nQuadTotal = len(qx)
    nElements = DQuad[0].shape[2]
    nQP = len(DQuad)  # 4

    # Initialize Microfield: list of arrays (50 entries)
    Microfield = [None] * 50
    Microfield[0] = qx.ravel().copy()
    Microfield[1] = qy.ravel().copy()

    # Compute stress at each quadrature point: sigma = DHat @ macroStrain - betaHat * T
    # tmp1..tmp4 for quad points 1..4
    tmp = [None] * nQP
    for iQP in range(nQP):
        # DHat{iQP} @ macroStrain: (6, 6, nE) @ (6, 1) -> (6, 1, nE)
        D_eps = np.einsum('ijk,jl->ilk', DHat[iQP], macroStrain)  # (6, 1, nE)
        # betaHat{iQP} * T: (6, 1, nE) * scalar
        tmp[iQP] = D_eps - betaHat[iQP] * macroTemp  # (6, 1, nE)

    # Store stress components (cells 2-7 = sigma_11..sigma_12)
    for jComp in range(6):
        stress_comp = np.zeros(nQuadTotal)
        for iQP in range(nQP):
            stress_comp[iQP::nQP] = tmp[iQP][jComp, 0, :]
        Microfield[2 + jComp] = stress_comp

    # Compute principal stresses — batch eigenvalue decomposition
    # Build (nQuadTotal, 3, 3) symmetric stress tensor array
    _sMat = np.zeros((nQuadTotal, 3, 3))
    _sMat[:, 0, 0] = Microfield[2]                       # s11
    _sMat[:, 1, 1] = Microfield[3]                       # s22
    _sMat[:, 2, 2] = Microfield[4]                       # s33
    _sMat[:, 1, 2] = _sMat[:, 2, 1] = Microfield[5]     # s23
    _sMat[:, 0, 2] = _sMat[:, 2, 0] = Microfield[6]     # s13
    _sMat[:, 0, 1] = _sMat[:, 1, 0] = Microfield[7]     # s12
    # eigvalsh on (N, 3, 3) returns (N, 3) sorted ascending
    _pStress = np.sort(np.linalg.eigvalsh(_sMat), axis=1)
    Microfield[8] = _pStress[:, 2]    # sigma_p1 (max)
    Microfield[9] = _pStress[:, 1]    # sigma_p2
    Microfield[10] = _pStress[:, 0]   # sigma_p3 (min)

    # Max shear stress: 0.5 * (sigma_p1 - sigma_p3) — cell 11
    Microfield[11] = 0.5 * (Microfield[8] - Microfield[10])

    # Max in-plane shear: sqrt(((s11-s22)/2)^2 + s12^2) — cell 12
    Microfield[12] = np.sqrt(
        ((Microfield[2] - Microfield[3]) / 2.0) ** 2 +
        Microfield[7] ** 2)

    # Compute strains: eps = inv(DQuad) @ (sigma + beta * T)
    # Build batched (nQuadTotal, 6, 6) stiffness array and (nQuadTotal, 6) RHS
    _D_all = np.zeros((nQuadTotal, 6, 6))
    _rhs_all = np.zeros((nQuadTotal, 6))
    _stress_vec = np.column_stack([Microfield[2+j] for j in range(6)])  # (nQuadTotal, 6)
    for iQP in range(nQP):
        # DQuad[iQP] is (6, 6, nElements) — transpose to (nElements, 6, 6) for batch solve
        _D_all[iQP::nQP] = DQuad[iQP].transpose(2, 0, 1)
        # RHS = sigma + beta * T
        _rhs_all[iQP::nQP] = _stress_vec[iQP::nQP] + betaQuad[iQP][:, 0, :].T * macroTemp
    # Batch solve: (nQuadTotal, 6, 6) \ (nQuadTotal, 6, 1) → (nQuadTotal, 6, 1)
    _microStrain_all = np.linalg.solve(_D_all, _rhs_all[:, :, np.newaxis])[:, :, 0]
    # Store strain components (cells 13-18)
    for j in range(6):
        Microfield[13 + j] = _microStrain_all[:, j]

    # Compute principal strains — batch eigenvalue decomposition
    # Build (nQuadTotal, 3, 3) symmetric strain tensor array
    # Note: Voigt ordering for strain is [e11, e22, e33, e23, e13, e12]
    _eMat = np.zeros((nQuadTotal, 3, 3))
    _eMat[:, 0, 0] = Microfield[13]                       # e11
    _eMat[:, 1, 1] = Microfield[14]                       # e22
    _eMat[:, 2, 2] = Microfield[15]                       # e33
    _eMat[:, 1, 2] = _eMat[:, 2, 1] = Microfield[16]     # e23
    _eMat[:, 0, 2] = _eMat[:, 2, 0] = Microfield[17]     # e13
    _eMat[:, 0, 1] = _eMat[:, 1, 0] = Microfield[18]     # e12
    # eigvalsh on (N, 3, 3) returns (N, 3) sorted ascending
    _pStrain = np.sort(np.linalg.eigvalsh(_eMat), axis=1)
    Microfield[19] = _pStrain[:, 2]   # eps_p1 (max)
    Microfield[20] = _pStrain[:, 1]   # eps_p2
    Microfield[21] = _pStrain[:, 0]   # eps_p3 (min)

    # Initialize mechanical strains (cells 39-44) and extra slots
    for i in range(39, 50):
        Microfield[i] = np.zeros(nQuadTotal)

    # Compute mechanical strains = total strains - thermal strains (vectorized)
    _thermal_all = np.zeros((nQuadTotal, 6))
    for iQP in range(nQP):
        # alphaQuad[iQP] is (6, 1, nElements) — extract (nElements, 6) thermal strain
        _thermal_all[iQP::nQP] = alphaQuad[iQP][:, 0, :].T * macroTemp
    for j in range(6):
        Microfield[39 + j] = Microfield[13 + j] - _thermal_all[:, j]

    # Store results
    ms.Microfield = Microfield

    return ms
