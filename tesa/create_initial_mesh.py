"""
Create the initial FE mesh for the microstructure.
"""

import numpy as np
from scipy.spatial.distance import cdist
from .find_grain_holes import find_grain_holes
from .drectangle import drectangle
from .hmatrix import hmatrix
from .dpoly import dpoly
from .distmesh2d import distmesh2d


def create_initial_mesh(ms):
    """
    Create the initial FE mesh.

    Generates an equilateral-triangle grid over the domain, thins it by the
    mesh size function, runs distmesh2d to optimize, then snaps to junction
    points and domain corners. Stores the result in ms.InitialMesh.

    Parameters
    ----------
    ms : Microstructure
        Microstructure state with fields: GrainJunctionPoints,
        MeshSizeFunctionGrid, MeshSizeFunctionGridLimits,
        GrainsSmoothed, GrainHoles, HoleGrains, InitialMesh.

    Returns
    -------
    ms : Microstructure
        Updated with InitialMesh = [pfix, pinit] where pfix is the fixed
        node array (corners + junction points) and pinit is the interior
        node array.
    """
    # Data
    jpts = ms.GrainJunctionPoints
    xx = ms.MeshSizeFunctionGrid[0]
    yy = ms.MeshSizeFunctionGrid[1]
    hh = ms.MeshSizeFunctionGrid[2]
    minx = ms.MeshSizeFunctionGridLimits[0]
    maxx = ms.MeshSizeFunctionGridLimits[1]
    miny = ms.MeshSizeFunctionGridLimits[2]
    maxy = ms.MeshSizeFunctionGridLimits[3]
    Gs = ms.GrainsSmoothed

    if ms.GrainHoles is None or len(ms.GrainHoles) == 0:
        Holes, holes = find_grain_holes(Gs)
        ms.GrainHoles = Holes
        ms.HoleGrains = holes
    else:
        Holes = ms.GrainHoles

    # Create initial distribution
    h0 = np.min(hh)
    geps = 0.001 * h0
    bbox = np.array([[minx, miny], [maxx, maxy]])
    print(f"  h0 (min element size) = {h0:.6f}")

    x_vals = np.arange(bbox[0, 0], bbox[1, 0] + h0 / 2, h0)
    y_vals = np.arange(bbox[0, 1], bbox[1, 1] + h0 * np.sqrt(3) / 4, h0 * np.sqrt(3) / 2)
    xg, yg = np.meshgrid(x_vals, y_vals)
    xg[1::2, :] = xg[1::2, :] + h0 / 2
    pinit = np.column_stack([xg.ravel(), yg.ravel()])
    print(f"  Initial grid: {xg.shape[1]}×{xg.shape[0]} = {pinit.shape[0]} points")

    pinit = pinit[drectangle(pinit, minx, maxx, miny, maxy) < geps, :]
    print(f"  After domain clip: {pinit.shape[0]} points")
    r0 = 1.0 / hmatrix(pinit, xx, yy, None, hh) ** 2
    pinit = pinit[np.random.rand(pinit.shape[0]) < r0 / np.max(r0), :]
    print(f"  After density thinning: {pinit.shape[0]} points")

    rect_poly = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]])
    pinit = pinit[dpoly(pinit, rect_poly) < geps, :]

    fd = lambda p: drectangle(p, minx / maxx, maxx / maxx, miny / maxx, maxy / maxx)
    fh_func = lambda p: hmatrix(p, xx / maxx, yy / maxx, None, hh)
    bbox_norm = bbox / maxx
    pfix = np.unique(np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]), axis=0)

    # Remove pfix from pinit
    pfix_set = set(map(tuple, np.round(pfix, 12)))
    mask = np.array([tuple(np.round(row, 12)) not in pfix_set for row in pinit])
    pinit = pinit[mask]

    h0_norm = h0 / maxx
    pfix_norm = pfix / maxx
    pinit_norm = pinit / maxx
    Gs_norm = [g / maxx for g in Gs]

    print(f"  Running distmesh2d ({pinit_norm.shape[0]} interior + {pfix_norm.shape[0]} fixed)...")
    pinit_out, _ = distmesh2d(ms, fd, fd, fh_func, h0_norm, bbox_norm,
                               pfix_norm, pinit_norm, 3, Gs_norm, Holes)

    # Scale back
    pinit = pinit_out * maxx

    # Snap closest points to junction points
    D = cdist(pinit, jpts)
    id_snap = np.argmin(D, axis=0)
    pinit[id_snap, :] = jpts

    # Snap corners
    corners = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])
    D = cdist(pinit, corners)
    id_corners = np.argmin(D, axis=0)
    pinit[id_corners, :] = corners

    # Build boundary matching points
    pl = pinit[pinit[:, 0] < minx + h0 / 5, :].copy()
    pl[:, 0] = minx
    pallfix = pl.copy()
    pl_r = pl.copy()
    pl_r[:, 0] = maxx
    pallfix = np.vstack([pallfix, pl_r])

    pb = pinit[pinit[:, 1] < miny + h0 / 5, :].copy()
    pb[:, 1] = miny
    pallfix = np.vstack([pallfix, pb])
    pb_t = pb.copy()
    pb_t[:, 1] = maxy
    pallfix = np.vstack([pallfix, pb_t])

    mask = ((pinit[:, 0] <= maxx - h0 / 5) & (pinit[:, 1] <= maxy - h0 / 5) &
            (pinit[:, 0] >= minx + h0 / 5) & (pinit[:, 1] >= miny + h0 / 5))
    pinit = pinit[mask, :]
    pinit = np.vstack([pinit, pallfix])
    pinit = np.unique(pinit, axis=0)

    pfix = np.unique(np.vstack([corners, jpts]), axis=0)
    pfix_set = set(map(tuple, np.round(pfix, 12)))
    mask = np.array([tuple(np.round(row, 12)) not in pfix_set for row in pinit])
    pinit = pinit[mask]

    ms.InitialMesh = [pfix, pinit]

    return ms
