"""
Compute the AEH approximation for macroscale effective thermal conductivity.

Uses the characteristic temperature functions (phi) and quadrature point
conductivities (kappa) to compute the effective thermal conductivity tensor:

kappa_eff = (1/|Y|) * sum_e sum_q  w_q * J_q * (kappa_q - kappa_q @ dN_q @ phi_e) * A_master

where |Y| is the total domain area, and the sum is over all elements and
quadrature points.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from . import fem_definitions as FEMDef


def compute_macroscale_effective_thermal_conductivity(
        node_coordinates,
        element_indices,
        kappaQuad,
        quadrature_point_jacobian,
        shape_function_derivative_matrix,
        characteristic_functions):
    """
    Compute effective thermal conductivity from AEH characteristic functions.

    Parameters
    ----------
    node_coordinates : (nNodes, 2) array
        Nodal coordinates.
    element_indices : (nElements, 6) array of int
        6-node element connectivity (1-based).
    kappaQuad : list of (3, 3, nElements) arrays
        Thermal conductivity at each quadrature point.
    quadrature_point_jacobian : (nElements, nQuadraturePoints) array
        Jacobian determinant at each quadrature point.
    shape_function_derivative_matrix : list of (3, 6, nElements) arrays
        Shape function derivative matrix dN at each quadrature point.
    characteristic_functions : (nNodes, 3) array
        Characteristic temperature functions phi.

    Returns
    -------
    kappaEffective : (3, 3) array
        Macroscale effective thermal conductivity matrix.
    """
    # Initialize output
    kappaEffective = np.zeros((3, 3))

    # Local declarations
    nElements = element_indices.shape[0]
    nQuadraturePoints = FEMDef.N_QUADRATURE_POINTS
    areaMasterElement = 0.5
    weights = FEMDef.w

    # Compute total domain area
    nodeX = node_coordinates[:, 0]
    nodeY = node_coordinates[:, 1]
    n1 = element_indices[:, 0] - 1
    n2 = element_indices[:, 1] - 1
    n3 = element_indices[:, 2] - 1
    areaTotal = np.sum(0.5 * np.abs(
        nodeX[n1] * (nodeY[n2] - nodeY[n3]) +
        nodeX[n2] * (nodeY[n3] - nodeY[n1]) +
        nodeX[n3] * (nodeY[n1] - nodeY[n2])
    ))

    # Construct 3D array of characteristic function values per element
    # charFuncs: (6, 3, nElements) — 6 nodes per element, 3 phi columns
    charFuncs = np.zeros((6, 3, nElements))
    for iElement in range(nElements):
        # Node IDs (1-based) directly index into characteristicFunctions
        iRows = element_indices[iElement, :] - 1  # 0-based
        charFuncs[:, :, iElement] = characteristic_functions[iRows, :3]

    # Compute the macroscale effective thermal conductivity
    for iQP in range(nQuadraturePoints):
        kappa = kappaQuad[iQP]                              # (3, 3, nElements)
        dN = shape_function_derivative_matrix[iQP]           # (3, 6, nElements)

        # kappa @ dN @ phi: (3,3,nE) @ (3,6,nE) @ (6,3,nE) -> (3,3,nE)
        dN_phi = np.einsum('ijk,jlk->ilk', dN, charFuncs)   # (3, 3, nE)
        kappa_dN_phi = np.einsum('ijk,jlk->ilk', kappa, dN_phi)  # (3, 3, nE)

        # quadPointContribution = area * w * (kappa - kappa @ dN @ phi)
        quadPointContribution = areaMasterElement * weights[iQP] * (kappa - kappa_dN_phi)

        # Scale by Jacobian and accumulate, dividing by total area
        J = quadrature_point_jacobian[:, iQP]  # (nElements,)
        scaled = quadPointContribution * J[np.newaxis, np.newaxis, :] / areaTotal

        # Sum over all elements
        kappaEffective += np.sum(scaled, axis=2)

    # Symmetrize
    kappaEffective = 0.5 * (kappaEffective + kappaEffective.T)

    return kappaEffective
