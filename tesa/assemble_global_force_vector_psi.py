"""
Assemble the global force vector Psi for the AEH thermal FE analysis.

Constructs element force vectors from quadrature point contributions
(B^T @ beta * J * w), then assembles into a sparse global force vector
with periodic boundary conditions. The output is a single column —
the RHS for the thermal characteristic function psi.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from scipy.sparse import coo_matrix
from . import fem_definitions as FEMDef


def assemble_global_force_vector_psi(
        node_coordinates,
        element_indices,
        boundary_node_pairs,
        betaQuad,
        quadrature_point_jacobian,
        strain_displacement_matrix):
    """
    Assemble the global force vector for the AEH thermal characteristic function psi.

    Computes element force vectors via numerical integration
    (B^T beta * J * w) over quadrature points, producing a single
    right-hand-side column for the thermal characteristic function.

    Parameters
    ----------
    node_coordinates : ndarray, shape (n_nodes, 2)
        Nodal coordinates.
    element_indices : ndarray, shape (n_elements, 6), dtype int
        6-node element connectivity (1-based node IDs).
    boundary_node_pairs : ndarray, shape (n_bnd_pairs, 2), dtype int
        Periodic boundary node pairs ``[bl_node, tr_node]`` (1-based).
    betaQuad : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (6, 1, n_elements) holding the stress-temperature moduli at
        that quadrature point.
    quadrature_point_jacobian : ndarray, shape (n_elements, n_qp)
        Jacobian determinant at each quadrature point.
    strain_displacement_matrix : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (6, 18, n_elements) holding the B matrix at that quadrature
        point.

    Returns
    -------
    globalForceVector : scipy.sparse.csc_matrix, shape (3*n_nodes, 1)
        Global force vector for the thermal characteristic function psi.
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
    # elementForceVector: (18, 1, nElements) — B^T @ beta
    elementForceVector = np.zeros((18, 1, nElements))
    for iQP in range(nQuadraturePoints):
        # B^T @ beta: (18, 6, nE)^T @ (6, 1, nE) -> (18, 1, nE)
        B = strain_displacement_matrix[iQP]       # (6, 18, nElements)
        beta = betaQuad[iQP]                        # (6, 1, nElements)

        # B^T @ beta (batch)
        BtBeta = np.einsum('ijk,ilk->jlk', B, beta)  # (18, 1, nE)

        # Scale by area * weight * Jacobian
        J = quadrature_point_jacobian[:, iQP]      # (nElements,)
        scale = areaMasterElement * weights[iQP] * J  # (nElements,)
        quadPointContribution = BtBeta * scale[np.newaxis, np.newaxis, :]

        elementForceVector += quadPointContribution

    # Initialize sparse storage and assemble the global force vector
    max_entries = 18 * nElements
    iRowNonzero = np.zeros(max_entries, dtype=int)
    jColNonzero = np.zeros(max_entries, dtype=int)
    nonzeroVals = np.zeros(max_entries, dtype=float)
    nonzeroCount = 0

    # dofMat: (3, 1) — DOF offsets [1,2,3]
    oneMat = np.ones((3, 1), dtype=int)
    dofMat = np.array([1, 2, 3]).reshape(3, 1)
    jColID = np.ones((18, 1), dtype=int)

    for iElement in range(nElements):
        # Setup global row indices
        node_ids = elementIndices[iElement, :]  # 6 node IDs (1-based)

        iRowID = np.zeros((18, 1), dtype=int)
        for iNode in range(6):
            base = 3 * (node_ids[iNode] - 1)
            iRowID[3 * iNode:3 * iNode + 3, :] = base * oneMat + dofMat

        # Current element force vector
        currentElementForceVector = elementForceVector[:, 0, iElement]

        # Find nonzero components
        nonzeroComponents = np.abs(currentElementForceVector) > np.sqrt(np.finfo(float).eps)
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
            currentElementForceVector[nonzeroComponents]
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
    if np.sum(mask) > 0:
        globalForceVector = coo_matrix(
            (nonzeroVals[mask], (iRowNonzero[mask] - 1, jColNonzero[mask] - 1)),
            shape=(3 * nNodes, 1)
        ).tocsc()
    else:
        from scipy.sparse import csc_matrix
        globalForceVector = csc_matrix((3 * nNodes, 1))

    return globalForceVector
