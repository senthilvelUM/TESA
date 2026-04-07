"""
Compute the shape function derivative matrix (B matrix) at each quadrature
point for 6-node isoparametric triangular elements.

This is the thermal conductivity equivalent of computeStrainDisplacementMatrix.
The output is a 3x6 matrix per quadrature point per element, where:
  - 3 rows = gradient components (dN/dx, dN/dy, 0)
  - 6 columns = 6 nodes
  - Row 3 is all zeros

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from . import fem_definitions as FEM
from .compute_coordinate_derivatives import compute_coordinate_derivatives


def compute_shape_function_derivative_matrix(node_coordinates, element_indices,
                                              quadrature_point_jacobian,
                                              coordinate_derivatives=None):
    """
    Compute the shape function derivative matrix at each quadrature point.

    Builds the 3x6 gradient matrix (dN/dx, dN/dy, 0) for thermal
    conductivity analysis. This is the scalar-field analogue of the
    strain-displacement matrix used in the elastic analysis.

    Parameters
    ----------
    node_coordinates : ndarray, shape (n_nodes, 2)
        Global (x, y) coordinates of all nodes.
    element_indices : ndarray, shape (n_elements, 6)
        Node indices for each element (0-based).
    quadrature_point_jacobian : ndarray, shape (n_elements, n_qp)
        Jacobian determinant at each quadrature point.
    coordinate_derivatives : dict or None, optional
        Pre-computed coordinate derivatives with keys ``'dxds'``,
        ``'dxdr'``, ``'dyds'``, ``'dydr'``. If None, they are computed
        internally.

    Returns
    -------
    BMatrix : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (3, 6, n_elements) containing the shape function derivative
        matrix at that quadrature point. Row 0 is dN/dx, row 1 is dN/dy,
        and row 2 is zeros.
    """
    # Initialize output — quadrature point shape function derivative matrices
    nElements = element_indices.shape[0]
    nQuadraturePoints = FEM.N_QUADRATURE_POINTS
    BMatrix = [None] * nQuadraturePoints

    # Local declarations
    r = FEM.r  # quadrature point coordinates in master element
    s = FEM.s

    # Compute the shape function derivatives with respect to the master element
    # coordinates
    dNdr = np.array([
        4.0 * r + 4.0 * s - 3.0,
        4.0 * r - 1.0,
        np.zeros(nQuadraturePoints),
        -8.0 * r - 4.0 * (s - 1.0),
        4.0 * s,
        -4.0 * s
    ])
    dNds = np.array([
        4.0 * s + 4.0 * r - 3.0,
        np.zeros(nQuadraturePoints),
        4.0 * s - 1.0,
        -4.0 * r,
        4.0 * r,
        -8.0 * s - 4.0 * (r - 1.0)
    ])

    # Compute coordinate derivatives if not provided
    if coordinate_derivatives is None:
        coordinate_derivatives = compute_coordinate_derivatives(
            node_coordinates, element_indices)

    # Copy coordinate derivatives for clarity
    dxds = coordinate_derivatives['dxds']
    dxdr = coordinate_derivatives['dxdr']
    dyds = coordinate_derivatives['dyds']
    dydr = coordinate_derivatives['dydr']

    # Compute the shape function derivative matrix at each quadrature point
    for iQuadraturePoint in range(nQuadraturePoints):

        # Initialize matrices for current quad point in all elements
        BMat = np.zeros((3, 6, nElements))

        # 1/J at current quadrature point for all elements
        invJ = 1.0 / quadrature_point_jacobian[:, iQuadraturePoint]

        # Row #1: dN/dx = (1/J) * (dN/dr * dy/ds - dN/ds * dy/dr)
        BMat[0, 0, :] = invJ * (
            dNdr[0, iQuadraturePoint] * dyds[:, iQuadraturePoint] -
            dNds[0, iQuadraturePoint] * dydr[:, iQuadraturePoint])
        BMat[0, 1, :] = invJ * (
            dNdr[1, iQuadraturePoint] * dyds[:, iQuadraturePoint] -
            dNds[1, iQuadraturePoint] * dydr[:, iQuadraturePoint])
        BMat[0, 2, :] = invJ * (
            dNdr[2, iQuadraturePoint] * dyds[:, iQuadraturePoint] -
            dNds[2, iQuadraturePoint] * dydr[:, iQuadraturePoint])
        BMat[0, 3, :] = invJ * (
            dNdr[3, iQuadraturePoint] * dyds[:, iQuadraturePoint] -
            dNds[3, iQuadraturePoint] * dydr[:, iQuadraturePoint])
        BMat[0, 4, :] = invJ * (
            dNdr[4, iQuadraturePoint] * dyds[:, iQuadraturePoint] -
            dNds[4, iQuadraturePoint] * dydr[:, iQuadraturePoint])
        BMat[0, 5, :] = invJ * (
            dNdr[5, iQuadraturePoint] * dyds[:, iQuadraturePoint] -
            dNds[5, iQuadraturePoint] * dydr[:, iQuadraturePoint])

        # Row #2: dN/dy = (1/J) * (dN/ds * dx/dr - dN/dr * dx/ds)
        BMat[1, 0, :] = invJ * (
            dNds[0, iQuadraturePoint] * dxdr[:, iQuadraturePoint] -
            dNdr[0, iQuadraturePoint] * dxds[:, iQuadraturePoint])
        BMat[1, 1, :] = invJ * (
            dNds[1, iQuadraturePoint] * dxdr[:, iQuadraturePoint] -
            dNdr[1, iQuadraturePoint] * dxds[:, iQuadraturePoint])
        BMat[1, 2, :] = invJ * (
            dNds[2, iQuadraturePoint] * dxdr[:, iQuadraturePoint] -
            dNdr[2, iQuadraturePoint] * dxds[:, iQuadraturePoint])
        BMat[1, 3, :] = invJ * (
            dNds[3, iQuadraturePoint] * dxdr[:, iQuadraturePoint] -
            dNdr[3, iQuadraturePoint] * dxds[:, iQuadraturePoint])
        BMat[1, 4, :] = invJ * (
            dNds[4, iQuadraturePoint] * dxdr[:, iQuadraturePoint] -
            dNdr[4, iQuadraturePoint] * dxds[:, iQuadraturePoint])
        BMat[1, 5, :] = invJ * (
            dNds[5, iQuadraturePoint] * dxdr[:, iQuadraturePoint] -
            dNdr[5, iQuadraturePoint] * dxds[:, iQuadraturePoint])

        # (Row #3 is all zeros)

        # Store the matrices for the current quadrature point
        BMatrix[iQuadraturePoint] = BMat

    return BMatrix
