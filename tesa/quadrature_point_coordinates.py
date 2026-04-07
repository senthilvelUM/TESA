"""
Determine global coordinates of quadrature points in each element.

Uses the 6-node isoparametric shape functions to map from master element
coordinates (r, s) to global coordinates (x, y) at each quadrature point.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from . import fem_definitions as FEM
from .shape_function import shape_function


def quadrature_point_coordinates(node_coordinates, element_indices):
    """
    Determine global coordinates of quadrature points in each element.

    Uses the 6-node isoparametric shape functions to map from master element
    coordinates (r, s) to global coordinates (x, y) at each quadrature point.

    Parameters
    ----------
    node_coordinates : ndarray, shape (n_nodes, 2)
        Global (x, y) coordinates of all nodes.
    element_indices : ndarray, shape (n_elements, 6)
        Node indices for each element (0-based).

    Returns
    -------
    quadXY : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (n_elements, 2) containing the global (x, y) coordinates at
        that quadrature point.
    """
    # Initialize output — global quadrature point coordinates
    nElements = element_indices.shape[0]
    nQuadraturePoints = FEM.N_QUADRATURE_POINTS
    quadXY = [None] * nQuadraturePoints
    for iQuadraturePoint in range(nQuadraturePoints):
        quadXY[iQuadraturePoint] = np.zeros((nElements, 2))

    # Local definitions
    r = FEM.r  # quadrature point coordinates in master element
    s = FEM.s
    elementX = node_coordinates[element_indices, 0]  # (nElements, 6)
    elementY = node_coordinates[element_indices, 1]  # (nElements, 6)

    # Loop over each quadrature point, determining global coordinates
    for jQuadPoint in range(nQuadraturePoints):

        quadXY[jQuadPoint][:, 0] = (
            elementX[:, 0] * shape_function(1, r[jQuadPoint], s[jQuadPoint]) +
            elementX[:, 1] * shape_function(2, r[jQuadPoint], s[jQuadPoint]) +
            elementX[:, 2] * shape_function(3, r[jQuadPoint], s[jQuadPoint]) +
            elementX[:, 3] * shape_function(4, r[jQuadPoint], s[jQuadPoint]) +
            elementX[:, 4] * shape_function(5, r[jQuadPoint], s[jQuadPoint]) +
            elementX[:, 5] * shape_function(6, r[jQuadPoint], s[jQuadPoint]))

        quadXY[jQuadPoint][:, 1] = (
            elementY[:, 0] * shape_function(1, r[jQuadPoint], s[jQuadPoint]) +
            elementY[:, 1] * shape_function(2, r[jQuadPoint], s[jQuadPoint]) +
            elementY[:, 2] * shape_function(3, r[jQuadPoint], s[jQuadPoint]) +
            elementY[:, 3] * shape_function(4, r[jQuadPoint], s[jQuadPoint]) +
            elementY[:, 4] * shape_function(5, r[jQuadPoint], s[jQuadPoint]) +
            elementY[:, 5] * shape_function(6, r[jQuadPoint], s[jQuadPoint]))

    return quadXY
