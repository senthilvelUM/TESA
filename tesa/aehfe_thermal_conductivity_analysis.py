"""
Asymptotic Expansion Homogenization (AEH) thermal conductivity analysis.

Main orchestrator that performs the full AEH-FE thermal conductivity pipeline:
1. Assign rotated thermal conductivities at quadrature points
2. Compute Jacobians
3. Compute shape function derivative matrices
4. Assemble global conductivity matrix
5. Assemble global force vector
6. Solve for characteristic temperature functions
7. Compute effective thermal conductivity (AEH, VRH, Geometric Mean)

Copyright 2014-2015, Alden C. Cook.
"""

import time
import numpy as np

from .assign_quadrature_point_thermal_conductivity import (
    assign_quadrature_point_thermal_conductivity
)
from .compute_quadrature_point_jacobian import compute_quadrature_point_jacobian
from .compute_shape_function_derivative_matrix import (
    compute_shape_function_derivative_matrix
)
from .assemble_global_stiffness_matrix_thermal_conductivity import (
    assemble_global_stiffness_matrix_thermal_conductivity
)
from .assemble_global_force_vector_thermal_conductivity import (
    assemble_global_force_vector_thermal_conductivity
)
from .solve_characteristic_functions_heat_conduction import (
    solve_characteristic_functions_heat_conduction
)
from .compute_macroscale_effective_thermal_conductivity import (
    compute_macroscale_effective_thermal_conductivity
)
from .compute_macroscale_voigt_reuss_hill_thermal_conductivity import (
    compute_macroscale_voigt_reuss_hill_thermal_conductivity
)
from .compute_macroscale_geometric_mean_thermal_conductivity import (
    compute_macroscale_geometric_mean_thermal_conductivity
)
from .quadrature_point_coordinates import quadrature_point_coordinates


# Element-level homogenization method names
_ELEMENT_HOMOG_METHODS = {
    1: 'Nearest Neighbor',
    2: 'Voigt',
    3: 'Reuss',
    4: 'Hill',
    5: 'Geometric Mean',
}


def aehfe_thermal_conductivity_analysis(ms, verbose=True):
    """
    Perform the full AEH thermal conductivity FE analysis.

    Parameters
    ----------
    ms : Microstructure
        Must have mesh, phase properties, and correction matrices populated.
    verbose : bool
        Print progress messages.

    Returns
    -------
    ms : Microstructure
        Updated with thermal conductivity results stored as attributes.
    """
    analysisStart = time.time()

    # Setup fields from ms
    node_coordinates = ms.SixNodeCoordinateList
    element_indices = ms.SixNodeElementIndexList

    # Build boundary node pairs: [bl_node, tr_node] (1-based)
    bl_node = np.arange(1, ms.NumberNodes + 1, dtype=int)
    tr_node = np.round(ms.BoundaryNodeRelationsList).astype(int)
    boundary_node_pairs = np.column_stack([bl_node, tr_node])

    data_point_coordinates = ms.DataCoordinateList
    data_point_euler_angles = ms.DataEulerAngle
    data_point_phase = ms.DataPhase.astype(int)

    data_csys_correction_angle = ms.EBSDCorrectionAngle

    element_homog_value = ms.ElementLevelHomogenizationMethodValue
    element_homog_method = _ELEMENT_HOMOG_METHODS.get(element_homog_value, 'Hill')

    phase_thermal_conductivity = ms.PhaseThermalConductivityMatrix

    # 1. Assign thermal conductivities to quadrature points
    if verbose:
        print(f"  Assigning thermal conductivities ({element_homog_method})...")
    kappaQuad = assign_quadrature_point_thermal_conductivity(
        node_coordinates, element_indices,
        data_point_coordinates, data_point_euler_angles, data_point_phase,
        phase_thermal_conductivity, data_csys_correction_angle,
        element_homog_method)

    # 2. Compute Jacobians at quadrature points
    if verbose:
        print("  Computing Jacobians...")
    quadraturePointJacobian, _ = compute_quadrature_point_jacobian(
        node_coordinates, element_indices - 1)

    # 3. Compute shape function derivative matrices
    if verbose:
        print("  Computing shape function derivative matrices...")
    shapeFunctionDerivativeMatrix = compute_shape_function_derivative_matrix(
        node_coordinates, element_indices - 1, quadraturePointJacobian)

    # 4. Assemble global conductivity matrix
    if verbose:
        print("  Assembling global conductivity matrix...")
    globalStiffnessMatrix = assemble_global_stiffness_matrix_thermal_conductivity(
        node_coordinates, element_indices, boundary_node_pairs,
        kappaQuad, quadraturePointJacobian, shapeFunctionDerivativeMatrix)

    # 5. Assemble global force vector (3 columns — x, y, z temperature gradients)
    if verbose:
        print("  Assembling global force vector...")
    globalForceVector = assemble_global_force_vector_thermal_conductivity(
        node_coordinates, element_indices, boundary_node_pairs,
        kappaQuad, quadraturePointJacobian, shapeFunctionDerivativeMatrix)

    # 6. Solve for characteristic temperature functions
    if verbose:
        print("  Solving for characteristic functions...")
    characteristicFunctions = solve_characteristic_functions_heat_conduction(
        boundary_node_pairs, globalStiffnessMatrix, globalForceVector)

    # 7. Compute AEH effective thermal conductivity
    if verbose:
        print("  Computing AEH effective thermal conductivity...")
    kappaEffective = compute_macroscale_effective_thermal_conductivity(
        node_coordinates, element_indices,
        kappaQuad, quadraturePointJacobian, shapeFunctionDerivativeMatrix,
        characteristicFunctions)

    # Compute VRH bounds using original EBSD data
    if verbose:
        print("  Computing VRH thermal conductivity bounds...")
    kappaVoigt, kappaReuss, kappaHill = \
        compute_macroscale_voigt_reuss_hill_thermal_conductivity(
            ms.OriginalDataEulerAngle, ms.OriginalDataPhase.astype(int),
            phase_thermal_conductivity, data_csys_correction_angle)

    # Compute geometric mean
    if verbose:
        print("  Computing geometric mean thermal conductivity...")
    kappaGeometricMean = compute_macroscale_geometric_mean_thermal_conductivity(
        ms.OriginalDataEulerAngle, ms.OriginalDataPhase.astype(int),
        phase_thermal_conductivity, data_csys_correction_angle)

    # Compute kappaHat localization tensor for post-processing heat flux fields
    if verbose:
        print("  Computing heat flux localization tensors (kappaHat)...")
    from .compute_aeh_heat_flux_info import compute_aeh_heat_flux_info
    kappaHat = compute_aeh_heat_flux_info(
        element_indices, kappaQuad, shapeFunctionDerivativeMatrix,
        characteristicFunctions)

    # Compute quadrature point coordinates for post-processing
    quadXY = quadrature_point_coordinates(node_coordinates, element_indices - 1)

    # Store results on ms
    ms.thermalConductivityCharacteristicFunctions = characteristicFunctions
    ms.kappaEffectiveAEH = kappaEffective
    ms.kappaEffectiveVoigt = kappaVoigt
    ms.kappaEffectiveReuss = kappaReuss
    ms.kappaEffectiveHill = kappaHill
    ms.kappaEffectiveGeoMean = kappaGeometricMean
    ms.quadraturePointThermalConductivity = kappaQuad
    ms.kappaHat = kappaHat
    ms.quadraturePointCoordinates = quadXY
    ms.quadraturePointJacobian = quadraturePointJacobian  # (nElements, nQP) — det(J) at each QP

    # Display execution time
    elapsed = time.time() - analysisStart
    if verbose:
        print(f"  AEH thermal conductivity analysis complete ({elapsed:.1f}s)")

    return ms
