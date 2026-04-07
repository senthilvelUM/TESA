import numpy as np
from .inpoly import inpoly


def find_grain_holes(G, verbose=False):
    """
    Find grains that are completely contained inside other grains (holes).

    Uses bounding-box pre-filtering to avoid expensive inpoly calls:
    grain j can only be a hole inside grain i if bbox(j) is fully
    contained within bbox(i).  For typical EBSD maps this eliminates
    >99% of candidate pairs.

    Parameters
    ----------
    G : list of ndarray
        List of grain boundary polygons, each shape (Ni, 2).
    verbose : bool, optional
        If True, print progress per grain. Default is False.

    Returns
    -------
    Holes : list of list
        Holes[i] is a list of grain indices that are holes inside grain i.
    holes : list
        Flat list of all grain indices that are holes.
    """
    n = len(G)
    Holes = [[] for _ in range(n)]
    holes = []

    # Precompute bounding boxes for all grains: [xmin, ymin, xmax, ymax]
    bboxes = np.empty((n, 4))
    for k in range(n):
        pts = G[k]
        bboxes[k, 0] = pts[:, 0].min()   # xmin
        bboxes[k, 1] = pts[:, 1].min()   # ymin
        bboxes[k, 2] = pts[:, 0].max()   # xmax
        bboxes[k, 3] = pts[:, 1].max()   # ymax

    # Precompute bounding-box areas for quick size comparison
    bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    _n_inpoly = 0
    for i in range(n):
        if verbose:
            print(f'Finding grain holes {i + 1} / {n}')

        # Bounding box of the enclosing candidate
        xi_min, yi_min, xi_max, yi_max = bboxes[i]

        for j in range(n):
            if i == j:
                continue

            # Skip if grain j's bbox is larger than grain i's bbox (can't be inside)
            if bbox_areas[j] >= bbox_areas[i]:
                continue

            # Skip if grain j's bbox is NOT fully inside grain i's bbox
            if (bboxes[j, 0] < xi_min or bboxes[j, 1] < yi_min or
                    bboxes[j, 2] > xi_max or bboxes[j, 3] > yi_max):
                continue

            # Bounding box of grain j is inside grain i — run full inpoly test
            _n_inpoly += 1
            polygon = G[i]
            points = G[j]
            result = inpoly(points, polygon)
            if np.all(result):
                Holes[i].append(j)
                holes.append(j)

    if verbose:
        print(f'  Bounding-box filter: {_n_inpoly} inpoly calls out of {n*(n-1)} pairs')

    return Holes, holes
