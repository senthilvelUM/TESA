"""
Assemble the global stiffness matrix for the AEH Finite Element Analysis.

Constructs element stiffness matrices from quadrature point contributions
(B^T @ D @ B * J * w), then assembles into a global sparse matrix with
periodic boundary conditions enforced by node pair substitution.

The global matrix has dimensions (3*nNodes) x (3*nNodes), where the factor
of 3 accounts for 3 displacement degrees of freedom per node (u, v, w).

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from scipy.sparse import coo_matrix
from . import fem_definitions as FEMDef


def assemble_global_stiffness_matrix(
        node_coordinates,
        element_indices,
        boundary_node_pairs,
        DQuad,
        quadrature_point_jacobian,
        strain_displacement_matrix):
    """
    Assemble the global FE stiffness matrix for the thermo-elastic AEH analysis.

    Computes element stiffness matrices via numerical integration
    (B^T D B * J * w) over quadrature points, then assembles into a sparse
    global matrix. Periodic boundary conditions are enforced by substituting
    top-right boundary node DOFs with their bottom-left counterparts.

    Parameters
    ----------
    node_coordinates : ndarray, shape (n_nodes, 2)
        Nodal coordinates.
    element_indices : ndarray, shape (n_elements, 6), dtype int
        6-node element connectivity (1-based node IDs).
    boundary_node_pairs : ndarray, shape (n_bnd_pairs, 2), dtype int
        Periodic boundary node pairs ``[bl_node, tr_node]`` (1-based).
        Zero entries in the ``tr_node`` column are skipped.
    DQuad : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (6, 6, n_elements) holding the rotated stiffness matrix at
        that quadrature point.
    quadrature_point_jacobian : ndarray, shape (n_elements, n_qp)
        Jacobian determinant at each quadrature point.
    strain_displacement_matrix : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (6, 18, n_elements) holding the strain-displacement (B) matrix
        at that quadrature point.

    Returns
    -------
    globalStiffnessMatrix : scipy.sparse.csc_matrix, shape (3*n_nodes, 3*n_nodes)
        Assembled global stiffness matrix in CSC format.
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
    # Replace boundary node references: tr_node -> bl_node
    for iBoundNodePair in range(nBoundNodePairs):
        bl_node = boundNodePairs[iBoundNodePair, 0]
        tr_node = boundNodePairs[iBoundNodePair, 1]
        if tr_node != 0:  # skip unpaired nodes
            elementIndices[elementIndices == tr_node] = bl_node

    # Construct the element stiffness matrices
    # elementStiffnessMatrix: (18, 18, nElements)
    elementStiffnessMatrix = np.zeros((18, 18, nElements))
    for iQP in range(nQuadraturePoints):
        # B^T @ D @ B for all elements at this quad point
        # strainDisplacementMatrix[iQP]: (6, 18, nElements)
        # DQuad[iQP]: (6, 6, nElements)
        B = strain_displacement_matrix[iQP]       # (6, 18, nElements)
        D = DQuad[iQP]                              # (6, 6, nElements)

        # B^T @ D: (18, 6, nE) @ ... but we need batch multiply
        BtD = np.einsum('ijk,ilk->jlk', B, D)     # (18, 6, nE)
        BtDB = np.einsum('ijk,jlk->ilk', BtD, B)  # (18, 18, nE)

        # Scale by area * weight * Jacobian
        J = quadrature_point_jacobian[:, iQP]      # (nElements,)
        scale = areaMasterElement * weights[iQP] * J  # (nElements,)
        quadPointContribution = BtDB * scale[np.newaxis, np.newaxis, :]

        elementStiffnessMatrix += quadPointContribution

    # Initialize sparse storage and assemble the global stiffness matrix
    # Pre-allocate arrays for COO format
    max_entries = 18 * 18 * nElements + 12 + 3 * nBoundNodePairs
    iRowNonzero = np.zeros(max_entries, dtype=int)
    jColNonzero = np.zeros(max_entries, dtype=int)
    nonzeroVals = np.zeros(max_entries, dtype=float)
    nonzeroCount = 0

    # dofMat: maps local node position to DOF offset [1,2,3] repeated
    dofMat = np.tile(np.array([1, 2, 3]).reshape(3, 1), (1, 18))  # (3, 18)

    for iElement in range(nElements):
        # Setup global indices of element stiffness entries
        # iRowID: (18, 18) — row indices for this element's stiffness
        node_ids = elementIndices[iElement, :]  # 6 node IDs (1-based)

        # Build row index matrix (18x18)
        iRowID = np.zeros((18, 18), dtype=int)
        for iNode in range(6):
            base = 3 * (node_ids[iNode] - 1)
            iRowID[3 * iNode:3 * iNode + 3, :] = base + dofMat

        # Build column index matrix (18x18)
        jColID = np.zeros((18, 18), dtype=int)
        for iNode in range(6):
            base = 3 * (node_ids[iNode] - 1)
            jColID[:, 3 * iNode:3 * iNode + 3] = base + dofMat.T

        # Symmetrize element stiffness
        currentElementStiffnessMatrix = 0.5 * (
            elementStiffnessMatrix[:, :, iElement] +
            elementStiffnessMatrix[:, :, iElement].T)

        # Find nonzero components
        nonzeroComponents = np.abs(currentElementStiffnessMatrix.ravel()) > np.sqrt(np.finfo(float).eps)
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

    # Remove any existing corner entries (DOFs 1-12, i.e., first 4 nodes * 3 DOFs)
    # and set diagonal entries for corner and boundary nodes
    cornerValsID = (iRowNonzero < 13) | (jColNonzero < 13)
    nonzeroVals[cornerValsID] = 0.0

    # Add diagonal entries for boundary node pairs (tr_node DOFs)
    boundDiagRowID = np.zeros(3 * nBoundNodePairs, dtype=int)
    for ib in range(nBoundNodePairs):
        tr = boundNodePairs[ib, 1]
        if tr != 0:
            boundDiagRowID[3 * ib] = 3 * (tr - 1) + 1
            boundDiagRowID[3 * ib + 1] = 3 * (tr - 1) + 2
            boundDiagRowID[3 * ib + 2] = 3 * (tr - 1) + 3

    # Filter out zero entries from boundary diagonal (unpaired nodes)
    boundDiagRowID = boundDiagRowID[boundDiagRowID > 0]

    # Concatenate: corner diagonal (1-12) + element entries + boundary diagonal
    corner_ids = np.arange(1, 13, dtype=int)
    iRowNonzero = np.concatenate([corner_ids, iRowNonzero, boundDiagRowID])
    jColNonzero = np.concatenate([corner_ids, jColNonzero, boundDiagRowID])
    nonzeroVals = np.concatenate([np.ones(12), nonzeroVals,
                                   np.ones(len(boundDiagRowID))])

    # Remove small values
    smallValuesID = np.abs(nonzeroVals) < np.sqrt(np.finfo(float).eps)
    mask = ~smallValuesID

    # Create sparse matrix (convert from 1-based to 0-based for scipy)
    globalStiffnessMatrix = coo_matrix(
        (nonzeroVals[mask], (iRowNonzero[mask] - 1, jColNonzero[mask] - 1)),
        shape=(3 * nNodes, 3 * nNodes)
    ).tocsc()

    return globalStiffnessMatrix
