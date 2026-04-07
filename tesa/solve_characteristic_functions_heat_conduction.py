"""
Solve for the characteristic functions in the AEH thermal conductivity formulation.

Solves the linear system K @ phi = F where K is the global conductivity matrix
and F is the global force vector (3 columns for 3 spatial temperature gradient
components). Uses sparse direct solver.

After solving, enforces periodic boundary conditions by copying the solution
at boundary node pairs: phi(tr_node) = phi(bl_node).

Note: 1 DOF per node (temperature), unlike the elastic version (3 DOFs).

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import issparse


def solve_characteristic_functions_heat_conduction(
        boundary_node_pairs,
        global_stiffness_matrix,
        global_force_vector):
    """
    Solve K phi = F for the AEH thermal conductivity characteristic functions.

    Solves the sparse linear system column-by-column using a direct solver,
    then enforces periodic boundary conditions by copying the solution from
    bottom-left nodes to their top-right counterparts. Uses 1 DOF per node
    (temperature), unlike the elastic solver which uses 3 DOFs per node.

    Parameters
    ----------
    boundary_node_pairs : ndarray, shape (n_bnd_pairs, 2), dtype int
        Periodic boundary node pairs ``[bl_node, tr_node]`` (1-based).
    global_stiffness_matrix : scipy.sparse matrix, shape (n_nodes, n_nodes)
        Global conductivity matrix.
    global_force_vector : scipy.sparse matrix or ndarray, shape (n_nodes, 3)
        Global force vector with 3 right-hand-side columns.

    Returns
    -------
    characteristicFunctions : ndarray, shape (n_nodes, 3)
        Characteristic temperature functions phi.
    """
    K = global_stiffness_matrix
    F = global_force_vector

    # Convert F to dense if sparse
    if issparse(F):
        F_dense = F.toarray()
    else:
        F_dense = np.asarray(F, dtype=float)

    nNodes = K.shape[0]
    nRHS = F_dense.shape[1]

    # Solve K @ phi = F for each RHS column
    characteristicFunctions = np.zeros((nNodes, nRHS))
    for iRHS in range(nRHS):
        characteristicFunctions[:, iRHS] = spsolve(K, F_dense[:, iRHS])

    # Update periodic boundary information:
    # phi(tr_node) = phi(bl_node)  — 1 DOF per node (temperature)
    boundNodePairs = boundary_node_pairs
    nBoundNodePairs = boundNodePairs.shape[0]
    for iBoundNodePair in range(nBoundNodePairs):
        bl_node = boundNodePairs[iBoundNodePair, 0]
        tr_node = boundNodePairs[iBoundNodePair, 1]
        if tr_node != 0:
            # 1-based node IDs to 0-based indices
            characteristicFunctions[tr_node - 1, :] = \
                characteristicFunctions[bl_node - 1, :]

    return characteristicFunctions
