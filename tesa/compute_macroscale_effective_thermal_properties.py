"""
Compute the AEH approximation for macroscale effective thermal properties.

Uses the thermal characteristic function (psi, column 7 of the combined
characteristic functions) and quadrature point stiffnesses and stress-temperature
moduli to compute the effective beta and alpha:

beta_eff = (1/|Y|) * sum_e sum_q  w_q * J_q * (beta_q - D_q @ B_q @ psi_e) * A_master
alpha_eff = inv(D_eff) @ beta_eff

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from . import fem_definitions as FEMDef


def compute_macroscale_effective_thermal_properties(
        node_coordinates,
        element_indices,
        DQuad,
        betaQuad,
        quadrature_point_jacobian,
        strain_displacement_matrix,
        characteristic_functions,
        DEffective):
    """
    Compute effective stress-temperature moduli and thermal expansion.

    Parameters
    ----------
    node_coordinates : (nNodes, 2) array
        Nodal coordinates.
    element_indices : (nElements, 6) array of int
        6-node element connectivity (1-based).
    DQuad : list of (6, 6, nElements) arrays
        Stiffness at each quadrature point.
    betaQuad : list of (6, 1, nElements) arrays
        Stress-temperature moduli at each quadrature point.
    quadrature_point_jacobian : (nElements, nQuadraturePoints) array
        Jacobian determinant at each quadrature point.
    strain_displacement_matrix : list of (6, 18, nElements) arrays
        B matrix at each quadrature point.
    characteristic_functions : (3*nNodes, 7) array
        Combined characteristic functions: columns 0-5 are chi (elastic),
        column 6 is psi (thermal).
    DEffective : (6, 6) array
        Effective stiffness matrix (from compute_macroscale_effective_stiffness).

    Returns
    -------
    betaEffective : (6, 1) array
        Effective stress-temperature moduli.
    alphaEffective : (6, 1) array
        Effective thermal expansion coefficients.
    """
    # Initialize output
    betaEffective = np.zeros((6, 1))

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

    # Construct 3D array of thermal characteristic function values (psi)
    # charFuncs: (18, 1, nElements) — column 7 (index 6) of characteristic functions
    charFuncs = np.zeros((18, 1, nElements))
    dofMat = np.array([1, 2, 3])
    for iElement in range(nElements):
        node_ids = element_indices[iElement, :]  # 6 nodes (1-based)
        iRows = np.zeros(18, dtype=int)
        for iNode in range(6):
            base = 3 * (node_ids[iNode] - 1)
            iRows[3 * iNode:3 * iNode + 3] = base + dofMat
        # Column 7 (index 6) is the thermal characteristic function psi
        charFuncs[:, 0, iElement] = characteristic_functions[iRows - 1, 6]

    # Compute the macroscale effective beta
    for iQP in range(nQuadraturePoints):
        D = DQuad[iQP]                              # (6, 6, nElements)
        beta = betaQuad[iQP]                         # (6, 1, nElements)
        B = strain_displacement_matrix[iQP]          # (6, 18, nElements)

        # D @ B @ psi: (6, 6, nE) @ (6, 18, nE) @ (18, 1, nE) -> (6, 1, nE)
        B_psi = np.einsum('ijk,jlk->ilk', B, charFuncs)   # (6, 1, nE)
        D_B_psi = np.einsum('ijk,jlk->ilk', D, B_psi)     # (6, 1, nE)

        # quadPointContribution = area * w * (beta - D @ B @ psi)
        quadPointContribution = areaMasterElement * weights[iQP] * (beta - D_B_psi)

        # Scale by Jacobian and accumulate
        J = quadrature_point_jacobian[:, iQP]  # (nElements,)
        scaled = quadPointContribution * J[np.newaxis, np.newaxis, :] / areaTotal

        # Sum over all elements
        betaEffective += np.sum(scaled, axis=2)

    # Compute effective thermal expansion: alpha_eff = inv(D_eff) @ beta_eff
    alphaEffective = np.linalg.solve(DEffective, np.eye(6)) @ betaEffective

    return betaEffective, alphaEffective
