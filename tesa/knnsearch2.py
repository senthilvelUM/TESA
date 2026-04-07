"""
Linear k-nearest neighbor (KNN) search.

Searches the reference data set R to find the k-nearest neighbors of
each query point in Q. Returns indices and distances.

When Q and R are the same (or R is None), the search excludes self-matches
by setting the self-distance to infinity.

By Yi Cao at Cranfield University on 25 March 2008.
"""

import numpy as np


def knnsearch2(Q, R=None, K=1):
    """
    Linear k-nearest neighbor search.

    Parameters
    ----------
    Q : (N, M) array
        Query points (N points in M-dimensional space).
    R : (L, M) array or None
        Reference data set. If None, R = Q (self-search).
    K : int
        Number of nearest neighbors to find. Default is 1.

    Returns
    -------
    idx : (N, K) array (int)
        Indices into R of the K nearest neighbors for each query point.
        0-based indices.
    D : (N, K) array (float)
        Euclidean distances to the K nearest neighbors.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)

    # Check inputs — determine if Q and R are identical
    fident = False
    if R is None:
        R = Q
        fident = True
    else:
        R = np.asarray(R, dtype=float)
        if R.ndim == 1:
            R = R.reshape(1, -1)
        if R.size == 0:
            fident = True
            R = Q
    if not fident:
        fident = np.array_equal(Q, R)

    N, M = Q.shape
    L = R.shape[0]
    idx = np.zeros((N, K), dtype=int)
    D = np.zeros((N, K), dtype=float)

    if K == 1:
        # Loop for each query point
        for k in range(N):
            d = np.zeros(L)
            for t in range(M):
                d = d + (R[:, t] - Q[k, t]) ** 2
            if fident:
                d[k] = np.inf
            idx[k, 0] = np.argmin(d)
            D[k, 0] = d[idx[k, 0]]
    else:
        for k in range(N):
            d = np.zeros(L)
            for t in range(M):
                d = d + (R[:, t] - Q[k, t]) ** 2
            if fident:
                d[k] = np.inf
            t_sorted = np.argsort(d)
            idx[k, :] = t_sorted[:K]
            D[k, :] = d[t_sorted[:K]]

    D = np.sqrt(D)

    return idx, D
