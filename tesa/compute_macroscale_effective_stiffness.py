"""
Compute the AEH approximation for macroscale effective elastic stiffnesses.

Uses the characteristic functions (chi) and quadrature point stiffnesses (D)
to compute the effective stiffness tensor:

D_eff = (1/|Y|) * sum_e sum_q  w_q * J_q * (D_q - D_q @ B_q @ chi_e) * A_master

where |Y| is the total domain area, and the sum is over all elements and
quadrature points.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from . import fem_definitions as FEMDef


def compute_macroscale_effective_stiffness(
        node_coordinates,
        element_indices,
        DQuad,
        quadrature_point_jacobian,
        strain_displacement_matrix,
        characteristic_functions):
    """
    Compute effective stiffness matrix from AEH characteristic functions.

    Parameters
    ----------
    node_coordinates : (nNodes, 2) array
        Nodal coordinates.
    element_indices : (nElements, 6) array of int
        6-node element connectivity (1-based).
    DQuad : list of (6, 6, nElements) arrays
        Stiffness at each quadrature point.
    quadrature_point_jacobian : (nElements, nQuadraturePoints) array
        Jacobian determinant at each quadrature point.
    strain_displacement_matrix : list of (6, 18, nElements) arrays
        B matrix at each quadrature point.
    characteristic_functions : (3*nNodes, 6) array
        Characteristic displacement functions chi.

    Returns
    -------
    DEffective : (6, 6) array
        Macroscale effective stiffness matrix.
    """
    # Initialize output
    DEffective = np.zeros((6, 6))

    # Local declarations
    nElements = element_indices.shape[0]
    nQuadraturePoints = FEMDef.N_QUADRATURE_POINTS
    areaMasterElement = 0.5
    weights = FEMDef.w

    # Compute total domain area using polygon area of corner nodes
    nodeX = node_coordinates[:, 0]
    nodeY = node_coordinates[:, 1]
    # Sum triangle areas: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    n1 = element_indices[:, 0] - 1  # 0-based
    n2 = element_indices[:, 1] - 1
    n3 = element_indices[:, 2] - 1
    areaTotal = np.sum(0.5 * np.abs(
        nodeX[n1] * (nodeY[n2] - nodeY[n3]) +
        nodeX[n2] * (nodeY[n3] - nodeY[n1]) +
        nodeX[n3] * (nodeY[n1] - nodeY[n2])
    ))

    # Construct 3D array of characteristic function values per element
    # charFuncs: (18, 6, nElements) — 18 DOFs per element, 6 chi columns
    charFuncs = np.zeros((18, 6, nElements))
    dofMat = np.array([1, 2, 3])
    for iElement in range(nElements):
        node_ids = element_indices[iElement, :]  # 6 nodes (1-based)
        iRows = np.zeros(18, dtype=int)
        for iNode in range(6):
            base = 3 * (node_ids[iNode] - 1)
            iRows[3 * iNode:3 * iNode + 3] = base + dofMat
        # Convert to 0-based for indexing into characteristicFunctions
        charFuncs[:, :, iElement] = characteristic_functions[iRows - 1, :6]

    # Compute the macroscale effective stiffness matrix
    for iQP in range(nQuadraturePoints):
        D = DQuad[iQP]                              # (6, 6, nElements)
        B = strain_displacement_matrix[iQP]          # (6, 18, nElements)

        # D @ B @ chi: (6, 6, nE) @ (6, 18, nE) @ (18, 6, nE) -> (6, 6, nE)
        B_chi = np.einsum('ijk,jlk->ilk', B, charFuncs)  # (6, 6, nE)
        D_B_chi = np.einsum('ijk,jlk->ilk', D, B_chi)    # (6, 6, nE)

        # quadPointContribution = area * w * (D - D @ B @ chi)
        quadPointContribution = areaMasterElement * weights[iQP] * (D - D_B_chi)

        # Scale by Jacobian and accumulate, dividing by total area
        J = quadrature_point_jacobian[:, iQP]  # (nElements,)
        scaled = quadPointContribution * J[np.newaxis, np.newaxis, :] / areaTotal

        # Sum over all elements
        DEffective += np.sum(scaled, axis=2)

    # Symmetrize
    DEffective = 0.5 * (DEffective + DEffective.T)

    return DEffective
