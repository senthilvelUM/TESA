"""
Compute stress/strain localization information from AEH characteristic functions.

Computes the localization tensors DHat and betaHat at each quadrature point:

DHat{q}  = D{q} - D{q} @ B{q} @ chi_e
betaHat{q} = beta{q} - D{q} @ B{q} @ psi_e

These are used in post-processing to compute local stress and strain fields
from the macroscale effective fields.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from . import fem_definitions as FEMDef


def compute_aeh_stress_strain_info(
        element_indices,
        DQuad,
        betaQuad,
        strain_displacement_matrix,
        characteristic_functions):
    """
    Compute stress/strain localization tensors.

    Parameters
    ----------
    element_indices : (nElements, 6) array of int
        6-node element connectivity (1-based).
    DQuad : list of (6, 6, nElements) arrays
        Stiffness at each quadrature point.
    betaQuad : list of (6, 1, nElements) arrays
        Stress-temperature moduli at each quadrature point.
    strain_displacement_matrix : list of (6, 18, nElements) arrays
        B matrix at each quadrature point.
    characteristic_functions : (3*nNodes, 7) array
        Combined characteristic functions: columns 0-5 are chi (elastic),
        column 6 is psi (thermal).

    Returns
    -------
    DHat : list of (6, 6, nElements) arrays
        Localization stiffness tensor at each quadrature point.
    betaHat : list of (6, 1, nElements) arrays
        Localization stress-temperature moduli at each quadrature point.
    """
    # Initialize output as copies of input
    nQuadraturePoints = FEMDef.N_QUADRATURE_POINTS
    DHat = [D.copy() for D in DQuad]
    betaHat = [b.copy() for b in betaQuad]

    # Local declarations
    nElements = element_indices.shape[0]

    # Construct 3D arrays of characteristic function values per element
    # dField: (18, 6, nElements) — chi (elastic, 6 columns)
    # gField: (18, 1, nElements) — psi (thermal, 1 column)
    dField = np.zeros((18, 6, nElements))
    gField = np.zeros((18, 1, nElements))
    dofMat = np.array([1, 2, 3])
    for iElement in range(nElements):
        node_ids = element_indices[iElement, :]  # 6 nodes (1-based)
        iRows = np.zeros(18, dtype=int)
        for iNode in range(6):
            base = 3 * (node_ids[iNode] - 1)
            iRows[3 * iNode:3 * iNode + 3] = base + dofMat
        # Convert to 0-based for indexing
        dField[:, :, iElement] = characteristic_functions[iRows - 1, :6]
        gField[:, 0, iElement] = characteristic_functions[iRows - 1, 6]

    # Compute localization tensors at each quadrature point
    for iQuad in range(nQuadraturePoints):
        B = strain_displacement_matrix[iQuad]   # (6, 18, nElements)
        D = DQuad[iQuad]                         # (6, 6, nElements)

        # D @ B @ chi: (6, 6, nE) @ (6, 18, nE) @ (18, 6, nE) -> (6, 6, nE)
        B_chi = np.einsum('ijk,jlk->ilk', B, dField)     # (6, 6, nE)
        D_B_chi = np.einsum('ijk,jlk->ilk', D, B_chi)    # (6, 6, nE)
        DHat[iQuad] = DHat[iQuad] - D_B_chi

        # D @ B @ psi: (6, 6, nE) @ (6, 18, nE) @ (18, 1, nE) -> (6, 1, nE)
        B_psi = np.einsum('ijk,jlk->ilk', B, gField)     # (6, 1, nE)
        D_B_psi = np.einsum('ijk,jlk->ilk', D, B_psi)    # (6, 1, nE)
        betaHat[iQuad] = betaHat[iQuad] - D_B_psi

    return DHat, betaHat
