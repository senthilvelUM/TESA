"""
Assemble the global force vector for the AEH elastic FE analysis.

Constructs element force vectors from quadrature point contributions
(B^T @ D * J * w), then assembles into a sparse global force matrix
with periodic boundary conditions. The output has 6 columns — one for
each independent strain component (e11, e22, e33, e23, e13, e12).

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from scipy.sparse import coo_matrix
from . import fem_definitions as FEMDef


def assemble_global_force_vector_chi(
        node_coordinates,
        element_indices,
        boundary_node_pairs,
        DQuad,
        quadrature_point_jacobian,
        strain_displacement_matrix):
    """
    Assemble the global force vector for the AEH elastic characteristic functions.

    Computes element force vectors via numerical integration (B^T D * J * w)
    over quadrature points, producing 6 right-hand-side columns corresponding
    to the 6 independent strain components (e11, e22, e33, e23, e13, e12).

    Parameters
    ----------
    node_coordinates : ndarray, shape (n_nodes, 2)
        Nodal coordinates.
    element_indices : ndarray, shape (n_elements, 6), dtype int
        6-node element connectivity (1-based node IDs).
    boundary_node_pairs : ndarray, shape (n_bnd_pairs, 2), dtype int
        Periodic boundary node pairs ``[bl_node, tr_node]`` (1-based).
    DQuad : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (6, 6, n_elements) holding the rotated stiffness at that
        quadrature point.
    quadrature_point_jacobian : ndarray, shape (n_elements, n_qp)
        Jacobian determinant at each quadrature point.
    strain_displacement_matrix : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (6, 18, n_elements) holding the B matrix at that quadrature
        point.

    Returns
    -------
    globalForceVector : scipy.sparse.csc_matrix, shape (3*n_nodes, 6)
        Global force vector with 6 columns (one per strain component).
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
    # elementForceVector: (18, 6, nElements) — B^T @ D for each quad point
    elementForceVector = np.zeros((18, 6, nElements))
    for iQP in range(nQuadraturePoints):
        # B^T @ D: (18, 6, nE)
        B = strain_displacement_matrix[iQP]       # (6, 18, nElements)
        D = DQuad[iQP]                              # (6, 6, nElements)

        # B^T @ D (batch)
        BtD = np.einsum('ijk,ilk->jlk', B, D)     # (18, 6, nE)

        # Scale by area * weight * Jacobian
        J = quadrature_point_jacobian[:, iQP]      # (nElements,)
        scale = areaMasterElement * weights[iQP] * J  # (nElements,)
        quadPointContribution = BtD * scale[np.newaxis, np.newaxis, :]

        elementForceVector += quadPointContribution

    # Initialize sparse storage and assemble the global force vector
    max_entries = 18 * 6 * nElements
    iRowNonzero = np.zeros(max_entries, dtype=int)
    jColNonzero = np.zeros(max_entries, dtype=int)
    nonzeroVals = np.zeros(max_entries, dtype=float)
    nonzeroCount = 0

    # dofMat: (3, 6) — DOF offsets [1,2,3] repeated 6 times
    oneMat = np.ones((3, 6), dtype=int)
    dofMat = np.tile(np.array([1, 2, 3]).reshape(3, 1), (1, 6))  # (3, 6)

    # jColID: (18, 6) — column indices (1 through 6 for each DOF row)
    jColID = np.zeros((18, 6), dtype=int)
    for c in range(6):
        jColID[:, c] = c + 1

    for iElement in range(nElements):
        # Setup global row indices for element force vector
        node_ids = elementIndices[iElement, :]  # 6 node IDs (1-based)

        iRowID = np.zeros((18, 6), dtype=int)
        for iNode in range(6):
            base = 3 * (node_ids[iNode] - 1)
            iRowID[3 * iNode:3 * iNode + 3, :] = base * oneMat + dofMat

        # Current element force vector
        currentElementForceVector = elementForceVector[:, :, iElement]

        # Find nonzero components
        nonzeroComponents = np.abs(currentElementForceVector.ravel()) > np.sqrt(np.finfo(float).eps)
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

    # Remove small values and corner entries (DOFs 1-12)
    smallValuesID = (np.abs(nonzeroVals) < np.sqrt(np.finfo(float).eps)) | \
                    (iRowNonzero < 13)
    mask = ~smallValuesID

    # Create sparse matrix (convert from 1-based to 0-based for scipy)
    globalForceVector = coo_matrix(
        (nonzeroVals[mask], (iRowNonzero[mask] - 1, jColNonzero[mask] - 1)),
        shape=(3 * nNodes, 6)
    ).tocsc()

    return globalForceVector
