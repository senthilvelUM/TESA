"""
Compute the strain-displacement matrices at each quadrature point.

These matrices (B matrices) relate nodal displacements to strains
at quadrature points for 6-node isoparametric triangular elements.

The output is a 6x18 matrix per quadrature point per element, where:
  - 6 rows = strain components (e11, e22, e33, 2*e23, 2*e13, 2*e12)
  - 18 columns = 6 nodes x 3 DOFs (u1,v1,w1, u2,v2,w2, ...)
  - Row 3 is all zeros (plane strain: e33 = 0)

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from . import fem_definitions as FEM
from .compute_coordinate_derivatives import compute_coordinate_derivatives


def compute_strain_displacement_matrix(node_coordinates, element_indices,
                                        quadrature_point_jacobian,
                                        coordinate_derivatives=None):
    """
    Compute the strain-displacement matrices at each quadrature point.

    Builds the 6x18 B matrix that relates nodal displacements to strains
    for 6-node quadratic triangular elements with 3 DOFs per node (u, v, w).
    Row layout: e11, e22, e33 (=0, plane strain), 2*e23, 2*e13, 2*e12.

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
    strain_displacement_matrix : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (6, 18, n_elements) containing the strain-displacement matrix
        at that quadrature point.
    """
    # Initialize output — quadrature point strain-displacement matrices
    nElements = element_indices.shape[0]
    nQuadraturePoints = FEM.N_QUADRATURE_POINTS
    strainDisplacementMatrix = [None] * nQuadraturePoints

    # Local declarations
    r = FEM.r  # quadrature point coordinates in master element
    s = FEM.s

    # Compute shape function derivatives w.r.t. master element coordinates
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

    # Compute the strain-displacement matrix at each quadrature point
    for iQuadraturePoint in range(nQuadraturePoints):

        # Initialize strain-displacement matrices for current quad point
        strainDispMat = np.zeros((6, 18, nElements))

        # 1/J at current quadrature point for all elements
        invJ = 1.0 / quadrature_point_jacobian[:, iQuadraturePoint]

        # Row #1 (e11): columns 1,4,7,10,13,16 (0-based: 0,3,6,9,12,15)
        for iNode in range(6):
            col = iNode * 3  # 0-based column: 0, 3, 6, 9, 12, 15
            strainDispMat[0, col, :] = invJ * (
                dNdr[iNode, iQuadraturePoint] * dyds[:, iQuadraturePoint] -
                dNds[iNode, iQuadraturePoint] * dydr[:, iQuadraturePoint])

        # Row #2 (e22): columns 2,5,8,11,14,17 (0-based: 1,4,7,10,13,16)
        for iNode in range(6):
            col = iNode * 3 + 1  # 0-based column: 1, 4, 7, 10, 13, 16
            strainDispMat[1, col, :] = invJ * (
                dNds[iNode, iQuadraturePoint] * dxdr[:, iQuadraturePoint] -
                dNdr[iNode, iQuadraturePoint] * dxds[:, iQuadraturePoint])

        # Row #3 (e33) is all zeros — plane strain

        # Row #4 (2*e23): columns 3,6,9,12,15,18 (0-based: 2,5,8,11,14,17)
        for iNode in range(6):
            col = iNode * 3 + 2  # 0-based column: 2, 5, 8, 11, 14, 17
            strainDispMat[3, col, :] = invJ * (
                dNds[iNode, iQuadraturePoint] * dxdr[:, iQuadraturePoint] -
                dNdr[iNode, iQuadraturePoint] * dxds[:, iQuadraturePoint])

        # Row #5 (2*e13): columns 3,6,9,12,15,18 (0-based: 2,5,8,11,14,17)
        for iNode in range(6):
            col = iNode * 3 + 2  # 0-based column: 2, 5, 8, 11, 14, 17
            strainDispMat[4, col, :] = invJ * (
                dNdr[iNode, iQuadraturePoint] * dyds[:, iQuadraturePoint] -
                dNds[iNode, iQuadraturePoint] * dydr[:, iQuadraturePoint])

        # Row #6 (2*e12): two entries per node (u and v columns)
        for iNode in range(6):
            col_u = iNode * 3      # 0-based: 0, 3, 6, 9, 12, 15
            col_v = iNode * 3 + 1  # 0-based: 1, 4, 7, 10, 13, 16
            strainDispMat[5, col_u, :] = invJ * (
                dNds[iNode, iQuadraturePoint] * dxdr[:, iQuadraturePoint] -
                dNdr[iNode, iQuadraturePoint] * dxds[:, iQuadraturePoint])
            strainDispMat[5, col_v, :] = invJ * (
                dNdr[iNode, iQuadraturePoint] * dyds[:, iQuadraturePoint] -
                dNds[iNode, iQuadraturePoint] * dydr[:, iQuadraturePoint])

        # Store the strain-displacement matrices for the current quadrature point
        strainDisplacementMatrix[iQuadraturePoint] = strainDispMat

    return strainDisplacementMatrix
