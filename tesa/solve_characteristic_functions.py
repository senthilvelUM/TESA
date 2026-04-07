"""
Solve for the characteristic functions in the AEH formulation.

Solves the linear system K @ chi = F where K is the global stiffness matrix
and F is the global force vector (6 columns for 6 independent strain components).
Uses sparse direct solver (Cholesky factorization via cholmod2).

After solving, enforces periodic boundary conditions by copying the solution
at boundary node pairs: chi(tr_node) = chi(bl_node).
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import issparse


def solve_characteristic_functions(
        boundary_node_pairs,
        global_stiffness_matrix,
        global_force_vector):
    """
    Solve K chi = F for the AEH elastic characteristic displacement functions.

    Solves the sparse linear system column-by-column using a direct solver,
    then enforces periodic boundary conditions by copying the solution from
    bottom-left nodes to their top-right counterparts.

    Parameters
    ----------
    boundary_node_pairs : ndarray, shape (n_bnd_pairs, 2), dtype int
        Periodic boundary node pairs ``[bl_node, tr_node]`` (1-based).
    global_stiffness_matrix : scipy.sparse matrix, shape (3*n_nodes, 3*n_nodes)
        Global FE stiffness matrix.
    global_force_vector : scipy.sparse matrix or ndarray, shape (3*n_nodes, 6)
        Global force vector with 6 right-hand-side columns.

    Returns
    -------
    characteristicFunctions : ndarray, shape (3*n_nodes, 6)
        Characteristic displacement functions chi.
    """
    K = global_stiffness_matrix
    F = global_force_vector

    # Convert F to dense if sparse
    if issparse(F):
        F_dense = F.toarray()
    else:
        F_dense = np.asarray(F, dtype=float)

    nDOF = K.shape[0]
    nRHS = F_dense.shape[1]

    # Solve K @ chi = F for each RHS column
    characteristicFunctions = np.zeros((nDOF, nRHS))
    for iRHS in range(nRHS):
        characteristicFunctions[:, iRHS] = spsolve(K, F_dense[:, iRHS])

    # Update periodic boundary information:
    # chi(tr_node DOFs) = chi(bl_node DOFs)
    boundNodePairs = boundary_node_pairs
    nBoundNodePairs = boundNodePairs.shape[0]
    for iBoundNodePair in range(nBoundNodePairs):
        bl_node = boundNodePairs[iBoundNodePair, 0]
        tr_node = boundNodePairs[iBoundNodePair, 1]
        if tr_node != 0:
            for jDOF in range(3):
                # 1-based node IDs to 0-based DOF indices
                tr_dof = 3 * (tr_node - 1) + jDOF
                bl_dof = 3 * (bl_node - 1) + jDOF
                characteristicFunctions[tr_dof, :] = \
                    characteristicFunctions[bl_dof, :]

    return characteristicFunctions
