"""
Plot heat flux or temperature gradient vectors on the microstructure.

Overlays quiver arrows on a magnitude contour background.
The arrows show the direction and relative magnitude of the in-plane
vector field (q1, q2) or (dT1, dT2) on a regular interpolation grid.

Uses grain-aware RBF interpolation so that arrow magnitudes are
consistent with the grain-aware RBF background colormap.
"""

import numpy as np
from .dpoly import dpoly


def plot_field_vectors(ms, comp, arrow_length=1.5):
    """
    Compute heat flux or temperature gradient vector arrows on a regular grid.

    Uses grain-aware RBF interpolation of vector components to grid points,
    matching the same RBF method used by _plot_rbf_field for the background.

    Parameters
    ----------
    ms : Microstructure
        Must have MicrofieldHeatConduction, SixNodeCoordinateList,
        GrainsMeshed (or Grains), and optionally GrainHoles.
    comp : int
        Component selector:
        comp < 5  -> heat flux vectors (q1, q2)
        comp >= 5 -> temperature gradient vectors (dT1, dT2)
    arrow_length : float
        Arrow length scaling factor. Default 1.5.
        Larger values produce longer arrows.

    Returns
    -------
    xx : (ny, nx) array — shifted grid x-coordinates
    yy : (ny, nx) array — shifted grid y-coordinates
    vv1 : (ny, nx) array — scaled arrow x-components
    vv2 : (ny, nx) array — scaled arrow y-components
    maxNorm : float — max norm of scaled arrows (for axis padding)
    """
    # Extract field data and node coordinates
    f = ms.MicrofieldHeatConduction
    p = ms.SixNodeCoordinateList

    # Select vector components based on comp
    if comp < 5:
        v1_all = f[2]   # q1
        v2_all = f[3]   # q2
    else:
        v1_all = f[6]   # dT1
        v2_all = f[7]   # dT2

    # Quadrature point coordinates
    qx = f[0]
    qy = f[1]

    # Get max norm from original quadrature point data
    maxNorm = np.max(np.sqrt(v1_all**2 + v2_all**2))
    if maxNorm < 1e-30:
        maxNorm = 1.0

    # Setup interpolation grid
    xmin = np.min(p[:, 0])
    xmax = np.max(p[:, 0])
    ymin = np.min(p[:, 1])
    ymax = np.max(p[:, 1])
    nx = 25
    ny = max(3, int(round(nx * (ymax - ymin) / (xmax - xmin))))

    # First pass: compute grid spacing
    xx_temp, _ = np.meshgrid(np.linspace(xmin, xmax, nx),
                              np.linspace(ymin, ymax, ny))
    sep = np.sort(np.unique(xx_temp.ravel()))
    sep = sep[1] - sep[0]

    # Second pass: inset grid by sep/3 to avoid boundary artifacts
    xx, yy = np.meshgrid(np.linspace(xmin + sep / 3, xmax - sep / 3, nx),
                          np.linspace(ymin + sep / 3, ymax - sep / 3, ny))
    sep = np.sort(np.unique(xx.ravel()))
    sep = sep[1] - sep[0]

    # ── Grain-aware RBF interpolation of vector components to grid ────
    # For each grain: find QPs inside, build RBF, evaluate at grid points.
    # This matches the RBF method used by _plot_rbf_field for the background.
    Gm = getattr(ms, 'GrainsMeshed', None)
    if Gm is None:
        Gm = getattr(ms, 'Grains', [])
    Holes = getattr(ms, 'GrainHoles', None)

    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    n_grid = grid_pts.shape[0]
    raw_v1 = np.zeros(n_grid)
    raw_v2 = np.zeros(n_grid)
    grid_assigned = np.zeros(n_grid, dtype=bool)

    qp_xy = np.column_stack([qx, qy])
    qp_remaining = np.ones(len(qx), dtype=bool)

    for n in range(len(Gm)):
        g = Gm[n]
        if g is None:
            continue
        g = np.asarray(g)
        if g.size == 0:
            continue

        # Close the polygon
        closed = np.vstack([g, g[0:1]])

        # Build distance function (with holes if needed)
        has_holes = False
        if Holes is not None:
            if isinstance(Holes, dict):
                has_holes = Holes.get(n) is not None and len(Holes[n]) > 0
            elif isinstance(Holes, list) and n < len(Holes):
                has_holes = Holes[n] is not None and len(Holes[n]) > 0
        if has_holes:
            from .ddiff_multi import ddiff_multi
            fd = lambda pts, _n=n, _h=Holes[n]: ddiff_multi(pts, Gm, _n, _h)
        else:
            fd = lambda pts, _c=closed: dpoly(pts, _c)

        # Find QPs inside this grain
        d_qp = fd(qp_xy[qp_remaining])
        in_grain_local = d_qp <= 0
        in_grain_global = np.where(qp_remaining)[0][in_grain_local]

        xg = qx[in_grain_global]
        yg = qy[in_grain_global]
        v1g = v1_all[in_grain_global]
        v2g = v2_all[in_grain_global]
        qp_remaining[in_grain_global] = False

        if len(xg) < 4:
            continue

        # Find grid points inside this grain
        d_grid = fd(grid_pts[~grid_assigned])
        in_grid_local = d_grid <= 0
        in_grid_global = np.where(~grid_assigned)[0][in_grid_local]

        if len(in_grid_global) == 0:
            continue

        # Downsample if too many QPs
        if len(xg) > 33000:
            step = 3
            xg, yg = xg[::step], yg[::step]
            v1g, v2g = v1g[::step], v2g[::step]

        # Build multiquadric RBF matrix: A(i,j) = sqrt(r_ij + 1)
        dx = xg[:, None] - xg[None, :]
        dy = yg[:, None] - yg[None, :]
        r = np.sqrt(dx**2 + dy**2)
        A = np.sqrt(r + 1.0)

        # Solve for RBF coefficients — one solve per component
        try:
            lam_v1 = np.linalg.solve(A, v1g)
            lam_v2 = np.linalg.solve(A, v2g)
        except np.linalg.LinAlgError:
            lam_v1, _, _, _ = np.linalg.lstsq(A, v1g, rcond=None)
            lam_v2, _, _, _ = np.linalg.lstsq(A, v2g, rcond=None)

        # Evaluate RBF at grid points inside this grain
        for idx in in_grid_global:
            px, py = grid_pts[idx, 0], grid_pts[idx, 1]
            r_node = np.sqrt((px - xg)**2 + (py - yg)**2)
            phi_vals = np.sqrt(r_node + 1.0)
            raw_v1[idx] = np.dot(lam_v1, phi_vals)
            raw_v2[idx] = np.dot(lam_v2, phi_vals)
            grid_assigned[idx] = True

    # Fallback for any unassigned grid points (near grain boundaries)
    if not np.all(grid_assigned):
        from scipy.spatial import cKDTree
        tree = cKDTree(qp_xy)
        unassigned = np.where(~grid_assigned)[0]
        _, nn_idx = tree.query(grid_pts[unassigned])
        raw_v1[unassigned] = v1_all[nn_idx]
        raw_v2[unassigned] = v2_all[nn_idx]

    raw_v1 = raw_v1.reshape(xx.shape)
    raw_v2 = raw_v2.reshape(xx.shape)

    # Scale arrows: arrow_length * sep * RBF(v) / maxNorm
    vv1 = arrow_length * sep * raw_v1 / maxNorm
    vv2 = arrow_length * sep * raw_v2 / maxNorm

    # Shift arrow tails to center on grid points
    xx = xx - 0.5 * vv1
    yy = yy - 0.5 * vv2

    # Recompute maxNorm for axis padding
    maxNorm = np.max(np.sqrt(vv1.ravel()**2 + vv2.ravel()**2))

    return xx, yy, vv1, vv2, maxNorm
