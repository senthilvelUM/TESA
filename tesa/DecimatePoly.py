"""
Reduce complexity of a 2D closed polygon by removing vertices.
Author: Anton Semechko (a.semechko@gmail.com), Jan 2011
"""

import numpy as np


def _poly_perim(C):
    """
    Compute polygon perimeter and minimum edge length.

    Parameters
    ----------
    C : ndarray, shape (N, 2)
        Closed polygon vertices (first vertex == last vertex).

    Returns
    -------
    P : float
        Total perimeter length.
    Emin : float
        Length of the shortest edge.
    """
    dE = C[1:, :] - C[:-1, :]
    dE_len = np.sqrt(np.sum(dE**2, axis=1))
    P = np.sum(dE_len)
    Emin = np.min(dE_len) if len(dE_len) > 0 else 0.0
    return P, Emin


def _poly_area(C):
    """
    Compute the absolute area of a polygon.

    Parameters
    ----------
    C : ndarray, shape (N, 2)
        Closed polygon vertices (first vertex == last vertex).

    Returns
    -------
    area : float
        Absolute area of the polygon.
    """
    dx = C[1:, 0] - C[:-1, 0]
    dy = C[1:, 1] + C[:-1, 1]
    return abs(np.sum(dx * dy) / 2.0)


def _recompute_errors(V):
    """
    Compute the squared distance error for removing the middle vertex.

    Measures how far the middle vertex deviates from the line segment
    connecting its two neighbors.

    Parameters
    ----------
    V : ndarray, shape (3, 2)
        Three consecutive vertices: V[0] = previous, V[1] = candidate
        for removal, V[2] = next.

    Returns
    -------
    err : float
        Squared perpendicular distance from V[1] to the segment V[0]-V[2].
    """
    D31 = V[2, :] - V[0, :]
    D21 = V[1, :] - V[0, :]
    dE_new2 = np.sum(D31**2)

    # Closest point on new edge
    t = np.sum(D21 * D31) / dE_new2 if dE_new2 > 0 else 0.0
    t = max(0.0, min(1.0, t))
    p = V[0, :] + t * D31

    # Squared distance
    return float(np.sum((p - V[1, :])**2))


def DecimatePoly(C, opt=None):
    """
    Reduce polygon complexity by iteratively removing vertices.

    Removes vertices whose removal introduces the smallest boundary
    offset error, continuing until no vertex can be removed within
    tolerance or the minimum vertex count is reached.

    Parameters
    ----------
    C : ndarray, shape (N, 2)
        Closed polygon vertices (first vertex == last vertex).
    opt : list of length 2 or None, optional
        Decimation options:
        - [B_tol, 1]: remove vertices with offset error < B_tol.
        - [P_tol, 2]: retain fraction P_tol of original vertices.
        - None: uses half the minimum edge length as B_tol.

    Returns
    -------
    C_out : ndarray, shape (M, 2)
        Simplified closed polygon (first vertex == last vertex).
    i_rem : ndarray, shape (N,), dtype bool
        True at indices where vertices were removed from the original.
    """
    C = np.asarray(C, dtype=float).copy()
    N = C.shape[0]
    i_rem = np.zeros(N, dtype=bool)

    if N <= 4:
        return C.copy(), i_rem

    # Tolerance parameter, perimeter and area of input polygon
    _, Emin = _poly_perim(C)
    B_tol = Emin / 2.0
    No = N - 1

    if opt is not None:
        opt = list(opt)
        B_tol = opt[0]
    else:
        opt = [B_tol, 1]

    Nmin = 3
    if opt[1] == 2:
        Nmin = round((N - 1) * opt[0])
        if (N - 1) == Nmin:
            return C.copy(), i_rem
        if Nmin < 3:
            Nmin = 3

    # Remove the repeating end-point
    C = C[:-1, :]
    N = C.shape[0]

    # Compute distance offset errors
    C_prev = np.roll(C, 1, axis=0)   # circshift(C, [1, 0])
    C_next = np.roll(C, -1, axis=0)  # circshift(C, [-1, 0])
    D31 = C_next - C_prev
    D21 = C - C_prev
    dE_new2 = np.sum(D31**2, axis=1)

    # Closest point on potential new edge
    t = np.sum(D21 * D31, axis=1) / np.maximum(dE_new2, 1e-30)
    t = np.clip(t, 0.0, 1.0)
    V = C_prev + t[:, np.newaxis] * D31

    # Squared distance errors
    Err_D2 = np.sum((V - C)**2, axis=1)

    # Distance error accumulation array
    DEAA = np.zeros(N)

    # Keep track of retained vertex indices (into the N-length open polygon)
    idx_ret = list(range(N))

    # Begin decimation
    while True:
        n = len(idx_ret)

        # Find vertices whose removal satisfies the criterion
        idx_i = np.where(Err_D2 < B_tol)[0]
        if len(idx_i) == 0 and n > Nmin and opt[1] == 2:
            B_tol = B_tol * np.sqrt(1.5)
            continue

        if len(idx_i) == 0 or n == Nmin:
            break

        # Vertex with smallest net error
        i_min = idx_i[np.argmin(Err_D2[idx_i])]

        # Update error accumulation
        DEAA[i_min] = DEAA[i_min] + np.sqrt(Err_D2[i_min])

        nn = len(Err_D2)
        i1 = (i_min - 1) % nn
        i3 = (i_min + 1) % nn

        DEAA[i1] = DEAA[i_min]
        DEAA[i3] = DEAA[i_min]

        # Recompute errors for neighbors
        # C is indexed by current retained positions — use a local C array
        C_local = np.array([C[idx_ret[j]] for j in range(len(idx_ret))])

        # Find positions of i_min, i1, i3 in the current retained set
        # Actually, Err_D2, DEAA, idx_ret are all kept in sync by deletion
        # So i_min, i1, i3 are direct indices into these arrays
        nn = len(idx_ret)
        i1 = (i_min - 1) % nn
        i3 = (i_min + 1) % nn

        i1_1 = (i1 - 1) % nn
        i1_3 = i3

        i3_1 = i1
        i3_3 = (i3 + 1) % nn

        # Get vertex coordinates from original C using idx_ret mapping
        err_D1 = _recompute_errors(np.array([
            C[idx_ret[i1_1]],
            C[idx_ret[i1]],
            C[idx_ret[i1_3]]
        ]))
        err_D3 = _recompute_errors(np.array([
            C[idx_ret[i3_1]],
            C[idx_ret[i3]],
            C[idx_ret[i3_3]]
        ]))

        # Update errors with accumulation
        Err_D2[i1] = (np.sqrt(err_D1) + DEAA[i1])**2
        Err_D2[i3] = (np.sqrt(err_D3) + DEAA[i3])**2

        # Remove the vertex
        Err_D2 = np.delete(Err_D2, i_min)
        DEAA = np.delete(DEAA, i_min)
        del idx_ret[i_min]

    # Reconstruct output polygon (closed)
    C_out = np.vstack([C[idx_ret], C[idx_ret[0]:idx_ret[0] + 1]])

    # Build removal mask
    retained_set = set(idx_ret)
    for i in range(N):
        if i not in retained_set:
            i_rem[i] = True
    # Last point mirrors first
    i_rem[-1] = i_rem[0]  # original N included the closing point

    # Actually i_rem was sized for the original closed polygon (N+1)
    # The MATLAB code: i_rem(idx_ret)=true; i_rem=~i_rem; i_rem(end)=i_rem(1);
    # So i_rem marks RETAINED vertices as True, then inverts to get REMOVED
    # Let's redo this properly for the original closed polygon size
    i_rem_full = np.zeros(N + 1, dtype=bool)
    for idx in idx_ret:
        i_rem_full[idx] = True
    i_rem_full = ~i_rem_full
    i_rem_full[-1] = i_rem_full[0]

    return C_out, i_rem_full
