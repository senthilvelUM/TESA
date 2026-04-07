"""
Compute heat flux localization information from AEH characteristic functions.

Computes the localization tensor kappaHat at each quadrature point:

kappaHat{q} = kappa{q} - kappa{q} @ dN{q} @ phi_e

This is used in post-processing to compute local heat flux fields
from the macroscale effective temperature gradient.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from . import fem_definitions as FEMDef


def compute_aeh_heat_flux_info(
        element_indices,
        kappaQuad,
        shape_function_derivative_matrix,
        characteristic_functions):
    """
    Compute heat flux localization tensor.

    Parameters
    ----------
    element_indices : (nElements, 6) array of int
        6-node element connectivity (1-based).
    kappaQuad : list of (3, 3, nElements) arrays
        Thermal conductivity at each quadrature point.
    shape_function_derivative_matrix : list of (3, 6, nElements) arrays
        Shape function derivative matrix dN at each quadrature point.
    characteristic_functions : (nNodes, 3) array
        Characteristic temperature functions phi.

    Returns
    -------
    kappaHat : list of (3, 3, nElements) arrays
        Localization conductivity tensor at each quadrature point.
    """
    # Initialize output as copies of input
    nQuadraturePoints = FEMDef.N_QUADRATURE_POINTS
    kappaHat = [k.copy() for k in kappaQuad]

    # Local declarations
    nElements = element_indices.shape[0]

    # Construct 3D array of characteristic function values per element
    # dField: (6, 3, nElements) — 6 nodes per element, 3 phi columns
    dField = np.zeros((6, 3, nElements))
    for iElement in range(nElements):
        # Node IDs (1-based) directly index into characteristicFunctions
        iRows = element_indices[iElement, :] - 1  # 0-based
        dField[:, :, iElement] = characteristic_functions[iRows, :3]

    # Compute localization tensor at each quadrature point
    for iQuad in range(nQuadraturePoints):
        kappa = kappaQuad[iQuad]                          # (3, 3, nElements)
        dN = shape_function_derivative_matrix[iQuad]       # (3, 6, nElements)

        # kappa @ dN @ phi: (3,3,nE) @ (3,6,nE) @ (6,3,nE) -> (3,3,nE)
        dN_phi = np.einsum('ijk,jlk->ilk', dN, dField)    # (3, 3, nE)
        kappa_dN_phi = np.einsum('ijk,jlk->ilk', kappa, dN_phi)  # (3, 3, nE)
        kappaHat[iQuad] = kappaHat[iQuad] - kappa_dN_phi

    return kappaHat
