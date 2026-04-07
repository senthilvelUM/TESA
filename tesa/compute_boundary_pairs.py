"""
Compute periodic boundary node pairs for a rectangular mesh.

For AEH analysis, periodic boundary conditions require pairing:
  - Right edge nodes with left edge nodes (same y-coordinate)
  - Top edge nodes with bottom edge nodes (same x-coordinate)
  - Corner nodes all tied to the bottom-left corner

Before pairing, boundary nodes are snapped to exact edge coordinates
and right/top edge nodes are adjusted to match left/bottom coordinates
exactly. This ensures every boundary node gets a valid pair.

The output is a 1D array where bnd_rel[i] = j means node i is the
dependent (right/top) node paired with independent (left/bottom) node j.
bnd_rel[i] = 0 means node i is not a dependent boundary node.

All indices are 1-based (matching the AEH solver convention).
"""

import numpy as np


def compute_boundary_pairs(p, tol=None):
    """
    Compute periodic boundary node pairs for a rectangular domain.

    Preprocessing:
    1. Snap boundary nodes to exact edge coordinates
    2. Force right edge y-coords to match left edge y-coords
    3. Force top edge x-coords to match bottom edge x-coords

    Parameters
    ----------
    p : (nNodes, 2) array
        Nodal coordinates (MODIFIED IN PLACE — boundary nodes are snapped).
    tol : float or None
        Tolerance for identifying boundary nodes. If None, auto-computed.

    Returns
    -------
    bnd_rel : (nNodes,) array of int
        Boundary relation list (1-based). bnd_rel[i] = j means node i+1
        is paired with node j (1-based). 0 = not a dependent node.
    """
    p = np.asarray(p, dtype=float)
    nNodes = p.shape[0]

    # Domain extents
    xmin, xmax = p[:, 0].min(), p[:, 0].max()
    ymin, ymax = p[:, 1].min(), p[:, 1].max()
    width = xmax - xmin
    height = ymax - ymin

    # Auto-compute tolerance for identifying boundary nodes
    if tol is None:
        edge_tol = max(width, height) * 0.001
        left_y = np.sort(p[np.abs(p[:, 0] - xmin) < edge_tol, 1])
        if len(left_y) > 1:
            min_spacing = np.min(np.diff(np.unique(np.round(left_y, 8))))
            if min_spacing > 0:
                tol = min_spacing * 0.3
            else:
                tol = edge_tol
        else:
            tol = edge_tol

    # ── Step 1: Identify boundary nodes ─────────────────────────────────
    left_mask   = np.abs(p[:, 0] - xmin) < tol
    right_mask  = np.abs(p[:, 0] - xmax) < tol
    bottom_mask = np.abs(p[:, 1] - ymin) < tol
    top_mask    = np.abs(p[:, 1] - ymax) < tol

    # ── Step 2: Snap boundary nodes to exact edge coordinates ───────────
    # All left edge nodes get x = xmin exactly
    p[left_mask, 0] = xmin
    # All right edge nodes get x = xmax exactly
    p[right_mask, 0] = xmax
    # All bottom edge nodes get y = ymin exactly
    p[bottom_mask, 1] = ymin
    # All top edge nodes get y = ymax exactly
    p[top_mask, 1] = ymax

    # ── Step 3: Force matching coordinates between paired edges ─────────
    # Exclude corner nodes from edge-to-edge matching
    bl_mask = left_mask & bottom_mask
    br_mask = right_mask & bottom_mask
    tl_mask = left_mask & top_mask
    tr_mask = right_mask & top_mask

    # Right edge y-coords must match left edge y-coords
    left_interior = np.where(left_mask & ~bottom_mask & ~top_mask)[0]
    right_interior = np.where(right_mask & ~bottom_mask & ~top_mask)[0]

    if len(left_interior) > 0 and len(right_interior) > 0:
        left_y_sorted = np.sort(p[left_interior, 1])
        right_y_sorted_idx = np.argsort(p[right_interior, 1])

        if len(left_interior) == len(right_interior):
            # Same count: directly assign sorted y-coords
            for k, ri in enumerate(right_interior[right_y_sorted_idx]):
                p[ri, 1] = left_y_sorted[k]
        else:
            # Different counts: match each right node to nearest left y-coord
            for ri in right_interior:
                dists = np.abs(p[ri, 1] - left_y_sorted)
                p[ri, 1] = left_y_sorted[np.argmin(dists)]

    # Top edge x-coords must match bottom edge x-coords
    bottom_interior = np.where(bottom_mask & ~left_mask & ~right_mask)[0]
    top_interior = np.where(top_mask & ~left_mask & ~right_mask)[0]

    if len(bottom_interior) > 0 and len(top_interior) > 0:
        bottom_x_sorted = np.sort(p[bottom_interior, 0])
        top_x_sorted_idx = np.argsort(p[top_interior, 0])

        if len(bottom_interior) == len(top_interior):
            for k, ti in enumerate(top_interior[top_x_sorted_idx]):
                p[ti, 0] = bottom_x_sorted[k]
        else:
            for ti in top_interior:
                dists = np.abs(p[ti, 0] - bottom_x_sorted)
                p[ti, 0] = bottom_x_sorted[np.argmin(dists)]

    # Snap corner nodes to exact corner coordinates
    p[np.where(bl_mask)[0], :] = [xmin, ymin]
    p[np.where(br_mask)[0], :] = [xmax, ymin]
    p[np.where(tl_mask)[0], :] = [xmin, ymax]
    p[np.where(tr_mask)[0], :] = [xmax, ymax]

    # ── Step 4: Pair boundary nodes ─────────────────────────────────────
    # Find bottom-left corner node
    bl_nodes = np.where(bl_mask)[0]
    if len(bl_nodes) == 0:
        dist_to_bl = (p[:, 0] - xmin) ** 2 + (p[:, 1] - ymin) ** 2
        bl_node = np.argmin(dist_to_bl)
    else:
        bl_node = bl_nodes[0]

    # Initialize output
    bnd_rel = np.zeros(nNodes, dtype=int)

    # Re-extract coordinates after snapping
    left_nodes = np.where(left_mask & ~bottom_mask & ~top_mask)[0]
    right_nodes = np.where(right_mask & ~bottom_mask & ~top_mask)[0]
    left_y_vals = p[left_nodes, 1]

    # Pair right → left (same y-coordinate, now exactly matching)
    for ri in right_nodes:
        y_ri = p[ri, 1]
        dists = np.abs(left_y_vals - y_ri)
        best = np.argmin(dists)
        bnd_rel[ri] = left_nodes[best] + 1  # 1-based

    # Pair top → bottom (same x-coordinate, now exactly matching)
    bottom_nodes = np.where(bottom_mask & ~left_mask & ~right_mask)[0]
    top_nodes = np.where(top_mask & ~left_mask & ~right_mask)[0]
    bottom_x_vals = p[bottom_nodes, 0]

    for ti in top_nodes:
        x_ti = p[ti, 0]
        dists = np.abs(bottom_x_vals - x_ti)
        best = np.argmin(dists)
        bnd_rel[ti] = bottom_nodes[best] + 1  # 1-based

    # Pair corner nodes → bottom-left
    for ci in np.where(br_mask)[0]:
        if ci != bl_node:
            bnd_rel[ci] = bl_node + 1
    for ci in np.where(tl_mask)[0]:
        if ci != bl_node:
            bnd_rel[ci] = bl_node + 1
    for ci in np.where(tr_mask)[0]:
        if ci != bl_node:
            bnd_rel[ci] = bl_node + 1

    return bnd_rel
