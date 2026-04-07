"""
Mesh cleanup: merge coincident nodes, remove degenerate elements, compact.

Extracted from the conforming mesh pipeline for reuse across mesh types
and during iteration quality checks.
"""

import numpy as np
from .triarea import triarea


def cleanup_mesh(p, t, verbose=False):
    """
    Merge coincident nodes, remove degenerate elements, and compact.

    The mesh may contain near-coincident nodes (from overlapping fixed +
    interior point sets in DistMesh). These create zero-area elements.
    Strategy: use degenerate elements to identify duplicate node pairs,
    merge them, remove the degenerate elements, then compact.

    Parameters
    ----------
    p : ndarray (N, 2)
        Node coordinates.
    t : ndarray (M, 3)
        Element connectivity (0-based).
    verbose : bool
        If True, print progress for each step.

    Returns
    -------
    p_clean : ndarray
        Cleaned node coordinates.
    t_clean : ndarray
        Cleaned element connectivity.
    stats : dict
        Cleanup statistics with keys: n_degen, n_merged, n_removed_elem,
        n_removed_node, nn_before, ne_before, nn_after, ne_after.
    """
    p = p.copy()
    t = t.copy()
    nn_pre = p.shape[0]
    ne_pre = t.shape[0]

    # Step 1: Find degenerate elements (area < 1e-8) and extract duplicate node pairs
    areas = np.abs(triarea(p, t))
    degen_mask = areas < 1e-8
    n_degen = int(np.sum(degen_mask))

    remap = np.arange(nn_pre, dtype=int)
    n_merged = 0
    if n_degen > 0:
        # For each degenerate element, find the shortest edge → duplicate pair
        degen_idx = np.where(degen_mask)[0]
        for di in degen_idx:
            ns = t[di]
            edges = [(ns[0], ns[1]), (ns[1], ns[2]), (ns[0], ns[2])]
            elens = [np.linalg.norm(p[a] - p[b]) for a, b in edges]
            shortest = np.argmin(elens)
            na, nb = edges[shortest]
            lo, hi = (min(na, nb), max(na, nb))
            remap[hi] = lo
        # Resolve chains
        for ni in range(nn_pre):
            while remap[remap[ni]] != remap[ni]:
                remap[ni] = remap[remap[ni]]
        n_merged = int(np.sum(remap != np.arange(nn_pre)))
        # Apply remap to all elements
        t = remap[t]
        if verbose:
            print(f"    Step 1: Found {n_degen} degenerate element(s), "
                  f"merged {n_merged} coincident node(s)")
    else:
        if verbose:
            print(f"    Step 1: No degenerate elements found")

    # Step 2: Remove degenerate elements (now have duplicate node indices after merge)
    distinct = np.array([len(set(row)) == 3 for row in t])
    areas2 = np.abs(triarea(p, t))
    good = distinct & (areas2 > 1e-8)
    n_removed_elem = ne_pre - int(np.sum(good))
    t = t[good]
    if verbose:
        if n_removed_elem > 0:
            print(f"    Step 2: Removed {n_removed_elem} degenerate element(s)")
        else:
            print(f"    Step 2: No elements to remove")

    # Step 3: Compact — remove unreferenced nodes, renumber sequentially
    used = np.unique(t.ravel())
    n_removed_node = 0
    if len(used) < nn_pre:
        renum = np.full(nn_pre, -1, dtype=int)
        renum[used] = np.arange(len(used))
        p = p[used]
        t = renum[t]
    n_removed_node = nn_pre - p.shape[0]
    if verbose:
        if n_removed_node > 0:
            print(f"    Step 3: Removed {n_removed_node} unreferenced node(s)")
        else:
            print(f"    Step 3: No unreferenced nodes")
        print(f"    Result: {p.shape[0]} nodes, {t.shape[0]} elements "
              f"(was {nn_pre} nodes, {ne_pre} elements)")

    stats = {
        "n_degen": n_degen,
        "n_merged": n_merged,
        "n_removed_elem": n_removed_elem,
        "n_removed_node": n_removed_node,
        "nn_before": nn_pre,
        "ne_before": ne_pre,
        "nn_after": p.shape[0],
        "ne_after": t.shape[0],
    }

    return p, t, stats
