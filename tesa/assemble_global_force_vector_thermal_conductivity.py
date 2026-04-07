"""
Assemble the global force vector for the AEH thermal conductivity FE analysis.

Constructs element force vectors from quadrature point contributions
(dN^T @ kappa * J * w), then assembles into a sparse global force matrix
with periodic boundary conditions. The output has 3 columns — one for
each spatial temperature gradient component.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from . import fem_definitions as FEMDef


def assemble_global_force_vector_thermal_conductivity(
        node_coordinates,
        element_indices,
        boundary_node_pairs,
        kappaQuad,
        quadrature_point_jacobian,
        shape_function_derivative_matrix):
    """
    Assemble the global force vector for the AEH thermal conductivity analysis.

    Computes element force vectors via numerical integration
    (dN^T kappa * J * w) over quadrature points, producing 3 right-hand-side
    columns corresponding to the 3 spatial temperature gradient components.

    Parameters
    ----------
    node_coordinates : ndarray, shape (n_nodes, 2)
        Nodal coordinates.
    element_indices : ndarray, shape (n_elements, 6), dtype int
        6-node element connectivity (1-based node IDs).
    boundary_node_pairs : ndarray, shape (n_bnd_pairs, 2), dtype int
        Periodic boundary node pairs ``[bl_node, tr_node]`` (1-based).
    kappaQuad : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (3, 3, n_elements) holding the rotated thermal conductivity at
        that quadrature point.
    quadrature_point_jacobian : ndarray, shape (n_elements, n_qp)
        Jacobian determinant at each quadrature point.
    shape_function_derivative_matrix : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (3, 6, n_elements) holding the shape function derivative (dN)
        matrix at that quadrature point.

    Returns
    -------
    globalForceVector : scipy.sparse.csc_matrix, shape (n_nodes, 3)
        Global force vector with 3 columns (one per spatial direction).
    """
    # Local declarations
    elementIndices = element_indices.copy()
    nElements = elementIndices.shape[0]
    nNodes = node_coordinates.shape[0]
    nQuadraturePoints = FEMDef.N_QUADRATURE_POINTS
    boundNodePairs = boundary_node_pairs
    nBoundNodePairs = boundNodePairs.shape[0]
    areaMasterElement = 0.5
    weights = FEMDef.w

    # Setup edited element index list for periodic boundary conditions
    for iBoundNodePair in range(nBoundNodePairs):
        bl_node = boundNodePairs[iBoundNodePair, 0]
        tr_node = boundNodePairs[iBoundNodePair, 1]
        if tr_node != 0:
            elementIndices[elementIndices == tr_node] = bl_node

    # Construct element force vectors
    # elementForceVector: (6, 3, nElements) — dN^T @ kappa
    elementForceVector = np.zeros((6, 3, nElements))
    for iQP in range(nQuadraturePoints):
        # dN^T @ kappa: (6, 3, nE)
        dN = shape_function_derivative_matrix[iQP]   # (3, 6, nE)
        kappa = kappaQuad[iQP]                         # (3, 3, nE)

        # dN^T @ kappa (batch)
        dNtK = np.einsum('ijk,ilk->jlk', dN, kappa)   # (6, 3, nE)

        # Scale by area * weight * Jacobian
        J = quadrature_point_jacobian[:, iQP]          # (nElements,)
        scale = areaMasterElement * weights[iQP] * J
        quadPointContribution = dNtK * scale[np.newaxis, np.newaxis, :]

        elementForceVector += quadPointContribution

    # Initialize sparse storage and assemble
    max_entries = 6 * 3 * nElements
    iRowNonzero = np.zeros(max_entries, dtype=int)
    jColNonzero = np.zeros(max_entries, dtype=int)
    nonzeroVals = np.zeros(max_entries, dtype=float)
    nonzeroCount = 0

    # jColID: (6, 3) — column indices 1, 2, 3 for each node row
    jColID = np.zeros((6, 3), dtype=int)
    for c in range(3):
        jColID[:, c] = c + 1

    oneMat = np.ones((1, 3), dtype=int)

    for iElement in range(nElements):
        # Setup global row indices — node IDs directly (1 DOF per node)
        node_ids = elementIndices[iElement, :]  # 6 node IDs (1-based)

        # iRowID: (6, 3) — node_id repeated 3 times per row
        iRowID = np.zeros((6, 3), dtype=int)
        for iNode in range(6):
            iRowID[iNode, :] = node_ids[iNode] * oneMat

        # Current element force vector
        currentElementForceVector = elementForceVector[:, :, iElement]

        # Find nonzero components
        nonzeroComponents = np.abs(currentElementForceVector.ravel()) > np.finfo(float).eps
        nNonzeroComponents = np.sum(nonzeroComponents)

        # Ensure arrays are large enough
        if nonzeroCount + nNonzeroComponents > len(iRowNonzero):
            extra = max(nNonzeroComponents, len(iRowNonzero))
            iRowNonzero = np.concatenate([iRowNonzero, np.zeros(extra, dtype=int)])
            jColNonzero = np.concatenate([jColNonzero, np.zeros(extra, dtype=int)])
            nonzeroVals = np.concatenate([nonzeroVals, np.zeros(extra, dtype=float)])

        # Store nonzero entries
        iRowNonzero[nonzeroCount:nonzeroCount + nNonzeroComponents] = \
            iRowID.ravel()[nonzeroComponents]
        jColNonzero[nonzeroCount:nonzeroCount + nNonzeroComponents] = \
            jColID.ravel()[nonzeroComponents]
        nonzeroVals[nonzeroCount:nonzeroCount + nNonzeroComponents] = \
            currentElementForceVector.ravel()[nonzeroComponents]
        nonzeroCount += nNonzeroComponents

    # Trim to actual size
    iRowNonzero = iRowNonzero[:nonzeroCount]
    jColNonzero = jColNonzero[:nonzeroCount]
    nonzeroVals = nonzeroVals[:nonzeroCount]

    # Remove small values and corner entries (nodes 1-4)
    smallValuesID = (np.abs(nonzeroVals) < np.finfo(float).eps) | \
                    (iRowNonzero < 5)
    mask = ~smallValuesID

    # Create sparse matrix (convert from 1-based to 0-based for scipy)
    if np.sum(mask) > 0:
        globalForceVector = coo_matrix(
            (nonzeroVals[mask], (iRowNonzero[mask] - 1, jColNonzero[mask] - 1)),
            shape=(nNodes, 3)
        ).tocsc()
    else:
        globalForceVector = csc_matrix((nNodes, 3))

    return globalForceVector
