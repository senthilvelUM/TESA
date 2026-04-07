"""
Assemble the global stiffness (conductivity) matrix for the AEH
thermal conductivity FE analysis.

Constructs element conductivity matrices from quadrature point contributions
(dN^T @ kappa @ dN * J * w), then assembles into a global sparse matrix with
periodic boundary conditions.

The global matrix has dimensions nNodes x nNodes (1 DOF per node: temperature).

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from scipy.sparse import coo_matrix
from . import fem_definitions as FEMDef


def assemble_global_stiffness_matrix_thermal_conductivity(
        node_coordinates,
        element_indices,
        boundary_node_pairs,
        kappaQuad,
        quadrature_point_jacobian,
        shape_function_derivative_matrix):
    """
    Assemble the global conductivity matrix for the AEH thermal analysis.

    Computes element conductivity matrices via numerical integration
    (dN^T kappa dN * J * w) over quadrature points, then assembles into a
    sparse global matrix. Periodic boundary conditions are enforced by
    substituting top-right boundary node DOFs with their bottom-left
    counterparts. One DOF per node (temperature).

    Parameters
    ----------
    node_coordinates : ndarray, shape (n_nodes, 2)
        Nodal coordinates.
    element_indices : ndarray, shape (n_elements, 6), dtype int
        6-node element connectivity (1-based node IDs).
    boundary_node_pairs : ndarray, shape (n_bnd_pairs, 2), dtype int
        Periodic boundary node pairs ``[bl_node, tr_node]`` (1-based).
        Zero entries in the ``tr_node`` column are skipped.
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
    globalStiffnessMatrix : scipy.sparse.csc_matrix, shape (n_nodes, n_nodes)
        Assembled global conductivity matrix in CSC format.
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

    # Construct element stiffness matrices
    # elementStiffnessMatrix: (6, 6, nElements) — 6 nodes per element, 1 DOF each
    elementStiffnessMatrix = np.zeros((6, 6, nElements))
    for iQP in range(nQuadraturePoints):
        # dN^T @ kappa @ dN for all elements at this quad point
        # shapeFunctionDerivativeMatrix[iQP]: (3, 6, nElements)
        # kappaQuad[iQP]: (3, 3, nElements)
        dN = shape_function_derivative_matrix[iQP]   # (3, 6, nE)
        kappa = kappaQuad[iQP]                         # (3, 3, nE)

        # dN^T @ kappa: (6, 3, nE)
        dNtK = np.einsum('ijk,ilk->jlk', dN, kappa)
        # (dN^T @ kappa) @ dN: (6, 6, nE)
        dNtKdN = np.einsum('ijk,jlk->ilk', dNtK, dN)

        # Scale by area * weight * Jacobian
        J = quadrature_point_jacobian[:, iQP]       # (nElements,)
        scale = areaMasterElement * weights[iQP] * J
        quadPointContribution = dNtKdN * scale[np.newaxis, np.newaxis, :]

        elementStiffnessMatrix += quadPointContribution

    # Initialize sparse storage and assemble
    max_entries = 6 * 6 * nElements + 4 + nBoundNodePairs
    iRowNonzero = np.zeros(max_entries, dtype=int)
    jColNonzero = np.zeros(max_entries, dtype=int)
    nonzeroVals = np.zeros(max_entries, dtype=float)
    nonzeroCount = 0

    for iElement in range(nElements):
        # Setup global indices — node IDs directly (1 DOF per node)
        node_ids = elementIndices[iElement, :]  # 6 node IDs (1-based)

        # iRowID: (6, 6) — row i has node_ids[i] repeated 6 times
        iRowID = np.repeat(node_ids.reshape(6, 1), 6, axis=1)
        # jColID: (6, 6) — transpose
        jColID = iRowID.T

        # Symmetrize element stiffness
        currentElementStiffnessMatrix = 0.5 * (
            elementStiffnessMatrix[:, :, iElement] +
            elementStiffnessMatrix[:, :, iElement].T)

        # Find nonzero components
        nonzeroComponents = np.abs(currentElementStiffnessMatrix.ravel()) > np.finfo(float).eps
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
            currentElementStiffnessMatrix.ravel()[nonzeroComponents]
        nonzeroCount += nNonzeroComponents

    # Trim to actual size
    iRowNonzero = iRowNonzero[:nonzeroCount]
    jColNonzero = jColNonzero[:nonzeroCount]
    nonzeroVals = nonzeroVals[:nonzeroCount]

    # Remove corner entries (nodes 1-4) and set diagonal
    cornerValsID = (iRowNonzero < 5) | (jColNonzero < 5)
    nonzeroVals[cornerValsID] = 0.0

    # Add diagonal entries for boundary node pairs (tr_node)
    boundDiagRowID = boundNodePairs[:, 1].copy()
    # Filter out zeros (unpaired nodes)
    boundDiagRowID = boundDiagRowID[boundDiagRowID > 0]

    # Concatenate: corner diagonal (1-4) + element entries + boundary diagonal
    corner_ids = np.arange(1, 5, dtype=int)
    iRowNonzero = np.concatenate([corner_ids, iRowNonzero, boundDiagRowID])
    jColNonzero = np.concatenate([corner_ids, jColNonzero, boundDiagRowID])
    nonzeroVals = np.concatenate([np.ones(4), nonzeroVals,
                                   np.ones(len(boundDiagRowID))])

    # Remove small values
    smallValuesID = np.abs(nonzeroVals) < np.finfo(float).eps
    mask = ~smallValuesID

    # Create sparse matrix (convert from 1-based to 0-based for scipy)
    globalStiffnessMatrix = coo_matrix(
        (nonzeroVals[mask], (iRowNonzero[mask] - 1, jColNonzero[mask] - 1)),
        shape=(nNodes, nNodes)
    ).tocsc()

    return globalStiffnessMatrix
