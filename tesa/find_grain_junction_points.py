import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from .find_grain_holes import find_grain_holes


def find_grain_junction_points(ms):
    """
    Find grain junction (triple/multi) points from grain boundary data.

    Parameters
    ----------
    ms : Microstructure
        Microstructure object. Modified in-place (GrainHoles, HoleGrains,
        GrainJunctionPoints are set).

    Returns
    -------
    jpts : ndarray, shape (M, 2)
        Junction points.
    """
    G = ms.Grains

    # Find holes if not already found
    if ms.HoleGrains is None or len(ms.HoleGrains) == 0:
        Holes, holes = find_grain_holes(G)
        ms.GrainHoles = Holes
        ms.HoleGrains = holes
    else:
        holes = ms.HoleGrains

    # Find data limits and store all points
    minx = 1e6
    maxx = 0
    miny = 1e6
    maxy = 0
    apts_list = []

    for i in range(len(G)):
        # unique rows, stable order
        _, idx = np.unique(G[i], axis=0, return_index=True)
        idx = np.sort(idx)
        G[i] = G[i][idx]

        x = G[i][:, 0]
        y = G[i][:, 1]
        minx = min(np.min(x), minx)
        miny = min(np.min(y), miny)
        maxx = max(np.max(x), maxx)
        maxy = max(np.max(y), maxy)
        apts_list.append(np.column_stack([x, y]))

    apts = np.vstack(apts_list)

    # Find first level junction points
    eps_val = np.sqrt(np.finfo(float).eps)

    c1 = apts[:, 0] < eps_val
    c2 = apts[:, 0] > maxx - eps_val
    c3 = apts[:, 1] < eps_val
    c4 = apts[:, 1] > maxy - eps_val

    bpts = apts[c1 | c2 | c3 | c4, :]

    # Remove unique rows (keep duplicates): MATLAB [~,id,~]=unique(bpts,'rows'); bpts(id,:)=[];
    _, unique_idx = np.unique(bpts, axis=0, return_index=True)
    mask = np.ones(len(bpts), dtype=bool)
    mask[unique_idx] = False
    bpts = bpts[mask]

    # setdiff with corners
    corners = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, miny]])
    if len(bpts) > 0:
        # Remove corner points from bpts
        corner_set = set(map(tuple, corners))
        keep = np.array([tuple(row) not in corner_set for row in bpts])
        bpts = bpts[keep] if np.any(keep) else np.empty((0, 2))

    # Prepend proper corners and unique remaining boundary points
    proper_corners = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])
    if len(bpts) > 0:
        _, uidx = np.unique(bpts, axis=0, return_index=True)
        bpts = bpts[np.sort(uidx)]
        bpts = np.vstack([proper_corners, bpts])
    else:
        bpts = proper_corners.copy()

    # jpts = points that appear 3+ times (remove unique twice)
    jpts = apts.copy()
    # First unique removal
    _, unique_idx = np.unique(jpts, axis=0, return_index=True)
    mask = np.ones(len(jpts), dtype=bool)
    mask[unique_idx] = False
    jpts = jpts[mask]
    # Second unique removal
    if len(jpts) > 0:
        _, unique_idx = np.unique(jpts, axis=0, return_index=True)
        mask = np.ones(len(jpts), dtype=bool)
        mask[unique_idx] = False
        jpts = jpts[mask]
    # Final unique
    if len(jpts) > 0:
        _, uidx = np.unique(jpts, axis=0, return_index=True)
        jpts = jpts[np.sort(uidx)]

    # Boundary left points
    blpts = bpts[bpts[:, 0] < minx + eps_val, :]
    if len(blpts) > 0:
        blpts = np.vstack([blpts, np.column_stack([blpts[:, 0] + maxx, blpts[:, 1]])])

    # Boundary right points
    brpts = bpts[bpts[:, 0] > maxx - eps_val, :]
    if len(brpts) > 0:
        brpts = np.vstack([brpts, np.column_stack([brpts[:, 0] - maxx, brpts[:, 1]])])

    # Boundary bottom points
    bbpts = bpts[bpts[:, 1] < miny + eps_val, :]
    if len(bbpts) > 0:
        bbpts = np.vstack([bbpts, np.column_stack([bbpts[:, 0], bbpts[:, 1] + maxy])])

    # Boundary top points
    btpts = bpts[bpts[:, 1] > maxy - eps_val, :]
    if len(btpts) > 0:
        btpts = np.vstack([btpts, np.column_stack([btpts[:, 0], btpts[:, 1] - maxy])])

    # Combine all junction points
    parts = [jpts]
    for arr in [blpts, brpts, bbpts, btpts]:
        if len(arr) > 0:
            parts.append(arr)
    jpts = np.vstack(parts)

    # Unique rows
    _, uidx = np.unique(jpts, axis=0, return_index=True)
    jpts = jpts[np.sort(uidx)]

    # Find second level junction points (hole junction points)
    apts_list2 = []
    harea = 0.0
    if holes is not None:
        for i in range(len(holes)):
            h_idx = holes[i]
            apts_list2.append(G[h_idx])
            poly = G[h_idx]
            # polyarea using shoelace formula
            harea += _polyarea(poly[:, 0], poly[:, 1])

    if len(apts_list2) > 0:
        hpts_all = np.vstack(apts_list2)
    else:
        hpts_all = np.empty((0, 2))

    # Find duplicate points in hole boundaries
    hpts = hpts_all.copy()
    if len(hpts) > 0:
        _, unique_idx = np.unique(hpts, axis=0, return_index=True)
        mask = np.ones(len(hpts), dtype=bool)
        mask[unique_idx] = False
        hpts = hpts[mask]

    if len(hpts) > 0:
        for i in range(len(hpts)):
            sarea = 0.0
            for j in range(len(holes)):
                h_idx = holes[j]
                pg = G[h_idx]
                # Remove rows matching hpts[i]
                keep = ~np.all(pg == hpts[i], axis=1)
                pg = pg[keep]
                if len(pg) > 0:
                    sarea += _polyarea(pg[:, 0], pg[:, 1])
            if abs(sarea - harea) > eps_val:
                jpts = np.vstack([jpts, hpts[i:i+1, :]])

    ms.GrainJunctionPoints = jpts

    return jpts


def _polyarea(x, y):
    """
    Compute polygon area using the shoelace formula.

    Parameters
    ----------
    x : ndarray, shape (N,)
        X-coordinates of polygon vertices.
    y : ndarray, shape (N,)
        Y-coordinates of polygon vertices.

    Returns
    -------
    area : float
        Absolute area of the polygon.
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
