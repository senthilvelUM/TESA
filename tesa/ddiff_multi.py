"""
Multi-grain distance difference function.
"""

import numpy as np
from .dpoly import dpoly
from .ddiff import ddiff


def ddiff_multi(p, Gm, ng, in_grains):
    """
    Signed distance for a grain with holes subtracted.

    Parameters
    ----------
    p         : (N, 2) query points
    Gm        : list of (Mi, 2) grain boundary arrays (open — will be closed)
    ng        : index of the target grain (0-based)
    in_grains : list/array of grain indices that are holes inside grain ng (0-based)

    Returns
    -------
    d : (N,) signed distance (negative inside grain ng but outside holes)
    """
    p = np.asarray(p, dtype=float)

    # Distance to target grain (close polygon)
    pv_ng = np.vstack([Gm[ng], Gm[ng][0:1, :]])
    d1 = dpoly(p, pv_ng)

    # Distance to each hole grain
    d2_list = []
    for ig in in_grains:
        if Gm[ig] is None or (hasattr(Gm[ig], 'shape') and Gm[ig].shape[0] == 0):
            # Hole grain has no meshed boundary (e.g. tiny grain lost all eligible snap nodes).
            # Skip this hole rather than crash — the outer grain's geometry is still valid.
            continue
        pv_ig = np.vstack([Gm[ig], Gm[ig][0:1, :]])
        d2_list.append(dpoly(p, pv_ig))

    if len(d2_list) == 0:
        return d1

    # Stack and take column-wise: d2 shape (N, n_holes)
    d2 = np.column_stack(d2_list) if len(d2_list) > 1 else d2_list[0].reshape(-1, 1)
    # ddiff operates row-wise: max(d1, -d2) across the hole columns
    # For multiple holes: d = max(d1, -min(d2_columns))
    # Actually MATLAB ddiff does max([d1, -d2], [], 2) which is max across columns
    d2_min = np.min(d2, axis=1) if d2.ndim > 1 else d2.ravel()
    return ddiff(d1, d2_min)
