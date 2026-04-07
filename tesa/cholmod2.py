"""
Sparse symmetric positive definite solver.

Computes LL' factorization of A(p,p) where p is a fill-reducing ordering,
then solves Ax = b. Uses scipy.sparse.linalg.spsolve which internally
applies fill-reducing orderings via SuiteSparse CHOLMOD/UMFPACK.
"""

import numpy as np
from scipy.sparse import issparse, csc_matrix
from scipy.sparse.linalg import spsolve


def cholmod2(A, b):
    """
    Solve Ax = b for a sparse symmetric positive definite system.

    Converts A to CSC format if needed, then solves using
    ``scipy.sparse.linalg.spsolve``. Multiple right-hand sides are
    handled column-by-column.

    Parameters
    ----------
    A : scipy.sparse matrix, shape (n, n)
        Symmetric positive definite coefficient matrix.
    b : ndarray, shape (n,) or (n, k)
        Right-hand side vector or matrix. If 2-D, each column is solved
        independently.

    Returns
    -------
    x : ndarray, shape (n,) or (n, k)
        Solution vector(s), same shape as ``b``.
    """
    # Ensure A is sparse CSC format (optimal for direct solvers)
    if not issparse(A):
        A = csc_matrix(A)
    elif A.format != 'csc':
        A = A.tocsc()

    b = np.asarray(b, dtype=float)

    # Handle multiple right-hand sides
    if b.ndim == 1:
        x = spsolve(A, b)
    else:
        # Solve column by column
        x = np.zeros_like(b)
        for k in range(b.shape[1]):
            x[:, k] = spsolve(A, b[:, k])

    return x
