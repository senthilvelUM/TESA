"""
Asymptotic Expansion Homogenization (AEH) thermo-elastic analysis.

Main orchestrator that performs the full AEH-FE analysis pipeline:
1. Assign rotated stiffnesses at quadrature points
2. Compute Jacobians
3. Compute strain-displacement matrices
4. Assemble global stiffness matrix
5. Assemble global force vectors (chi and psi)
6. Solve for characteristic functions
7. Compute effective stiffness, beta, and alpha

Copyright 2014-2015, Alden C. Cook.
"""

import time
import numpy as np
from scipy.sparse import hstack as sp_hstack

from .assign_quadrature_point_properties import assign_quadrature_point_properties
from .compute_quadrature_point_jacobian import compute_quadrature_point_jacobian
from .compute_strain_displacement_matrix import compute_strain_displacement_matrix
from .assemble_global_stiffness_matrix import assemble_global_stiffness_matrix
from .assemble_global_force_vector_chi import assemble_global_force_vector_chi
from .assemble_global_force_vector_psi import assemble_global_force_vector_psi
from .solve_characteristic_functions import solve_characteristic_functions
from .compute_macroscale_effective_stiffness import compute_macroscale_effective_stiffness
from .compute_macroscale_effective_thermal_properties import (
    compute_macroscale_effective_thermal_properties
)
from .quadrature_point_coordinates import quadrature_point_coordinates
from . import fem_definitions as FEMDef


# Element-level homogenization method names
_ELEMENT_HOMOG_METHODS = {
    1: 'Nearest Neighbor',
    2: 'Voigt',
    3: 'Reuss',
    4: 'Hill',
    5: 'Geometric Mean',
}


def aehfe_thermoelastic_analysis(ms, verbose=True):
    """
    Perform the full AEH thermo-elastic FE analysis.

    Parameters
    ----------
    ms : Microstructure
        Must have mesh, phase properties, and correction matrices populated.
    verbose : bool
        Print progress messages.

    Returns
    -------
    ms : Microstructure
        Updated with AEH results stored as attributes.
    """
    analysisStart = time.time()

    # Setup AnalysisInfo fields from ms
    node_coordinates = ms.SixNodeCoordinateList
    element_indices = ms.SixNodeElementIndexList

    # Build boundary node pairs: [bl_node, tr_node] (1-based)
    bl_node = np.arange(1, ms.NumberNodes + 1, dtype=int)
    tr_node = np.round(ms.BoundaryNodeRelationsList).astype(int)
    boundary_node_pairs = np.column_stack([bl_node, tr_node])

    data_point_coordinates = ms.DataCoordinateList
    data_point_euler_angles = ms.DataEulerAngle
    data_point_phase = ms.DataPhase.astype(int)

    data_csys_correction_matrix = ms.EBSDCorrectionMatrix
    data_csys_correction_angle = ms.EBSDCorrectionAngle

    element_homog_value = ms.ElementLevelHomogenizationMethodValue
    element_homog_method = _ELEMENT_HOMOG_METHODS.get(element_homog_value, 'Hill')

    phase_stiffness = ms.PhaseStiffnessMatrix
    phase_thermal_expansion = ms.PhaseThermalExpansionMatrix

    # 1. Assign stiffnesses to quadrature points
    if verbose:
        print(f"  Assigning properties ({element_homog_method})...")
    DQuad, betaQuad, alphaQuad = assign_quadrature_point_properties(
        node_coordinates, element_indices,
        data_point_coordinates, data_point_euler_angles, data_point_phase,
        phase_stiffness, phase_thermal_expansion,
        data_csys_correction_matrix, data_csys_correction_angle,
        element_homog_method)

    # 2. Compute Jacobians at quadrature points
    if verbose:
        print("  Computing Jacobians...")
    # element_indices are 1-based; compute_quadrature_point_jacobian expects 0-based
    quadraturePointJacobian, _ = compute_quadrature_point_jacobian(
        node_coordinates, element_indices - 1)

    # 3. Compute strain-displacement matrices
    if verbose:
        print("  Computing strain-displacement matrices...")
    strainDisplacementMatrix = compute_strain_displacement_matrix(
        node_coordinates, element_indices - 1, quadraturePointJacobian)

    # 4. Assemble global stiffness matrix
    if verbose:
        print("  Assembling global stiffness matrix...")
    globalStiffnessMatrix = assemble_global_stiffness_matrix(
        node_coordinates, element_indices, boundary_node_pairs,
        DQuad, quadraturePointJacobian, strainDisplacementMatrix)

    # 5. Assemble global force vectors
    if verbose:
        print("  Assembling global force vectors...")
    # Force vector for chi (6 columns — elastic characteristic functions)
    globalForceVectorChi = assemble_global_force_vector_chi(
        node_coordinates, element_indices, boundary_node_pairs,
        DQuad, quadraturePointJacobian, strainDisplacementMatrix)
    # Force vector for psi (1 column — thermal characteristic function)
    globalForceVectorPsi = assemble_global_force_vector_psi(
        node_coordinates, element_indices, boundary_node_pairs,
        betaQuad, quadraturePointJacobian, strainDisplacementMatrix)

    # 6. Solve for characteristic functions
    # Combine chi (6 cols) and psi (1 col) into 7-column RHS
    if verbose:
        print("  Solving for characteristic functions...")
    globalForceVector = sp_hstack([globalForceVectorChi, globalForceVectorPsi]).tocsc()
    characteristicFunctions = solve_characteristic_functions(
        boundary_node_pairs, globalStiffnessMatrix, globalForceVector)

    # 7. Compute effective properties
    if verbose:
        print("  Computing effective stiffness...")
    DEffective = compute_macroscale_effective_stiffness(
        node_coordinates, element_indices,
        DQuad, quadraturePointJacobian, strainDisplacementMatrix,
        characteristicFunctions)

    if verbose:
        print("  Computing effective thermal properties...")
    betaEffective, alphaEffective = compute_macroscale_effective_thermal_properties(
        node_coordinates, element_indices,
        DQuad, betaQuad, quadraturePointJacobian, strainDisplacementMatrix,
        characteristicFunctions, DEffective)

    # Compute quadrature point coordinates for post-processing
    quadXY = quadrature_point_coordinates(node_coordinates, element_indices - 1)
    nQP = FEMDef.N_QUADRATURE_POINTS
    nElements = element_indices.shape[0]
    tmpx = np.zeros(nQP * nElements)
    tmpy = np.zeros(nQP * nElements)
    for iQP in range(nQP):
        tmpx[iQP::nQP] = quadXY[iQP][:, 0]
        tmpy[iQP::nQP] = quadXY[iQP][:, 1]

    # Store results on ms
    ms.chiCharacteristicFunctions = characteristicFunctions[:, :6]
    ms.psiCharacteristicFunctions = characteristicFunctions[:, 6]
    ms.DEffectiveAEH = DEffective
    ms.betaEffectiveAEH = betaEffective
    ms.alphaEffectiveAEH = alphaEffective
    ms.quadraturePointCoordinates = np.column_stack([tmpx, tmpy])
    ms.quadraturePointElasticStiffnesses = DQuad
    ms.quadraturePointThermalExpansionCoefficients = alphaQuad
    ms.quadraturePointStressTemperatureModuli = betaQuad
    ms.quadraturePointJacobian = quadraturePointJacobian  # (nElements, nQP) — det(J) at each QP

    # Store strain-displacement matrix and combined characteristic functions
    # (needed for computing DHat/betaHat in stress/strain post-processing)
    ms.strainDisplacementMatrix = strainDisplacementMatrix
    ms.combinedCharacteristicFunctions = characteristicFunctions  # (3*nNodes, 7)

    # Display execution time
    elapsed = time.time() - analysisStart
    if verbose:
        print(f"  AEH thermo-elastic analysis complete ({elapsed:.1f}s)")

    return ms
