import numpy as np
from .dpoly import dpoly
from .ddiff_multi import ddiff_multi


def plfs(xx, yy, gamma, kappatol, Gs, cg, Holes):
    """
    Return local feature size (lfs) via medial axis on a Cartesian grid.

    Parameters
    ----------
    xx : ndarray
        X coordinates grid (2D meshgrid output).
    yy : ndarray
        Y coordinates grid (2D meshgrid output).
    gamma : float
        Medial axis detection parameter.
    kappatol : float
        Curvature tolerance.
    Gs : list of ndarray
        Grain boundary polygons.
    cg : int
        Index of the current grain (0-based).
    Holes : list or None
        Hole indices for this grain, or None/empty if no holes.

    Returns
    -------
    p : ndarray, shape (M, 2)
        Medial axis points in original grid coordinates.
    """
    # setup unit spacing grid
    nx = len(np.unique(xx.ravel()))
    ny = len(np.unique(yy.ravel()))
    xxu_1d = np.linspace(0, nx - 1, nx)
    yyu_1d = np.linspace(0, ny - 1, ny)
    xxu, yyu = np.meshgrid(xxu_1d, yyu_1d)

    mx = np.max(xxu.ravel()) / (np.max(xx.ravel()) - np.min(xx.ravel()))
    my = np.max(yyu.ravel()) / (np.max(yy.ravel()) - np.min(yy.ravel()))

    # Scale grain boundaries to unit grid
    Gs_scaled = []
    for i in range(len(Gs)):
        g = Gs[i].copy()
        g[:, 0] = mx * g[:, 0] - mx * np.min(xx.ravel())
        g[:, 1] = my * g[:, 1] - my * np.min(yy.ravel())
        Gs_scaled.append(g)

    # compute distance function
    if Holes is None or len(Holes) == 0:
        closed_poly = np.vstack([Gs_scaled[cg], Gs_scaled[cg][0:1, :]])
        def fd(pts):
            return dpoly(pts, closed_poly)
    else:
        def fd(pts):
            return ddiff_multi(pts, Gs_scaled, cg, Holes)

    rows = xxu.shape[0]
    cols = xxu.shape[1]
    too_big = False

    try:
        dd = fd(np.column_stack([xxu.ravel(), yyu.ravel()]))
        dd = dd.reshape(rows, cols)
    except Exception:
        too_big = True
        xt = xxu.ravel()
        yt = yyu.ravel()
        nt = len(xt)
        q = int(np.ceil(nt / 4))

        print(f'Computing distances for medial axis 1 4')
        dd = fd(np.column_stack([xt[:q], yt[:q]]))

        print(f'Computing distances for medial axis 2 4')
        dd2 = fd(np.column_stack([xt[q:2*q], yt[q:2*q]]))
        dd = np.concatenate([dd, dd2])

        print(f'Computing distances for medial axis 3 4')
        dd3 = fd(np.column_stack([xt[2*q:3*q], yt[2*q:3*q]]))
        dd = np.concatenate([dd, dd3])

        print(f'Computing distances for medial axis 4 4')
        dd4 = fd(np.column_stack([xt[3*q:], yt[3*q:]]))
        dd = np.concatenate([dd, dd4])

        dd = dd.reshape(rows, cols)

    inside = dd < 0

    # Compute gradient and curvature
    d_dx = np.zeros_like(dd)
    d_ddx = np.zeros_like(dd)
    dx_val = 1.0
    dy_val = 1.0

    # x-direction derivatives (along columns, axis=1) — vectorized
    d_dx[:, 1:-1] = (dd[:, 2:] - dd[:, :-2]) / (2 * dx_val)
    d_ddx[:, 1:-1] = (dd[:, 2:] - 2 * dd[:, 1:-1] + dd[:, :-2]) / dx_val**2

    d_dx[:, 0] = (dd[:, 1] - dd[:, 0]) / dx_val
    d_dx[:, -1] = (-dd[:, -1] + dd[:, -2]) / dx_val

    d_dxy = d_dx.copy()

    d_ddx[:, 0] = (d_dx[:, 1] - d_dx[:, 0]) / dx_val
    d_ddx[:, -1] = (-d_dx[:, -1] + d_dx[:, -2]) / dx_val

    d_dy = np.zeros_like(dd)
    d_ddy = np.zeros_like(dd)

    # y-direction derivatives (along rows, axis=0) — vectorized
    d_dy[1:-1, :] = (dd[2:, :] - dd[:-2, :]) / (2 * dy_val)
    d_ddy[1:-1, :] = (dd[2:, :] - 2 * dd[1:-1, :] + dd[:-2, :]) / dy_val**2
    # d_dxy has sequential dependency (row i uses updated row i-1) — keep as loop
    for i in range(1, dd.shape[0] - 1):
        d_dxy[i, :] = (d_dxy[i + 1, :] - d_dxy[i - 1, :]) / (2 * dy_val)

    d_dy[0, :] = (dd[1, :] - dd[0, :]) / dy_val
    d_dy[-1, :] = (-dd[-1, :] + dd[-2, :]) / dy_val
    d_ddy[0, :] = (d_dy[1, :] - d_dy[0, :]) / dy_val
    d_ddy[-1, :] = (-d_dy[-1, :] + d_dy[-2, :]) / dy_val

    denom = (d_dx**2 + d_dy**2) ** (3.0 / 2.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = (d_ddx * d_dy**2 - 2.0 * d_dy * d_dx * d_dxy + d_ddy * d_dx**2) / denom
    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)

    # Find medial axis points
    # Pre-allocate
    p_list = []

    # x-direction scan
    for i in range(dd.shape[0]):
        for j in range(3, dd.shape[1] - 4):  # MATLAB 4:size-4 => 0-based 3:size-5
            if not inside[i, j]:
                continue
            d1 = dd[i, j - 2]
            d2 = dd[i, j - 1]
            d3 = dd[i, j]
            d4 = dd[i, j + 1]
            d5 = dd[i, j + 2]
            d6 = dd[i, j + 3]

            p1 = np.polyfit([-2, -1, 0], [d1, d2, d3], 2)
            p2 = np.polyfit([1, 2, 3], [d4, d5, d6], 2)

            diff_poly = p2 - p1
            s = np.roots(diff_poly)
            if len(s) == 0:
                continue

            # Keep only real roots
            st = []
            for k in range(len(s)):
                if np.isreal(s[k]):
                    st.append(np.real(s[k]))
            s = np.array(st)
            if len(s) == 0:
                continue

            if not np.any((s >= 0) & (s <= 1)):
                continue

            k_poly = np.polyder(diff_poly)
            kvals = np.polyval(k_poly, s)
            so = s[kvals == np.max(kvals)]
            so = so[0]

            if not (so >= 0 and so <= 1):
                continue

            kappa1 = kappa[i, j - 1]
            kappa2 = kappa[i, j + 2]
            alpha = d_dx[i, j] * d_dx[i, j + 1] + d_dy[i, j] * d_dy[i, j + 1]

            if alpha < 1 - (gamma**2) * max(kappa1**2, kappa2**2, kappatol**2) / 2:
                p_list.append([xxu[i, j] + so, yyu[i, j]])

    # y-direction scan
    for j in range(dd.shape[1]):
        for i in range(3, dd.shape[0] - 4):  # MATLAB 4:size-4 => 0-based 3:size-5
            if not inside[i, j]:
                continue
            d1 = dd[i - 2, j]
            d2 = dd[i - 1, j]
            d3 = dd[i, j]
            d4 = dd[i + 1, j]
            d5 = dd[i + 2, j]
            d6 = dd[i + 3, j]

            p1 = np.polyfit([-2, -1, 0], [d1, d2, d3], 2)
            p2 = np.polyfit([1, 2, 3], [d4, d5, d6], 2)

            diff_poly = p2 - p1
            s = np.roots(diff_poly)
            if len(s) == 0:
                continue

            st = []
            for k in range(len(s)):
                if np.isreal(s[k]):
                    st.append(np.real(s[k]))
            s = np.array(st)
            if len(s) == 0:
                continue

            if not np.any((s >= 0) & (s <= 1)):
                continue

            k_poly = np.polyder(diff_poly)
            kvals = np.polyval(k_poly, s)
            so = s[kvals == np.max(kvals)]
            so = so[0]

            if not (so >= 0 and so <= 1):
                continue

            kappa1 = kappa[i - 1, j]
            kappa2 = kappa[i + 2, j]
            alpha = d_dx[i, j] * d_dx[i + 1, j] + d_dy[i, j] * d_dy[i + 1, j]

            if alpha < 1 - (gamma**2) * max(kappa1**2, kappa2**2, kappatol**2) / 2:
                p_list.append([xxu[i, j], yyu[i, j] + so])

    # Convert back to regular grid coordinates
    # Restore Gs (not strictly needed since we used Gs_scaled, but reconstruct fd)
    Gs_orig = Gs  # original Gs was not modified
    if Holes is None or len(Holes) == 0:
        closed_poly = np.vstack([Gs_orig[cg], Gs_orig[cg][0:1, :]])
        def fd_orig(pts):
            return dpoly(pts, closed_poly)
    else:
        def fd_orig(pts):
            return ddiff_multi(pts, Gs_orig, cg, Holes)

    if len(p_list) == 0:
        return np.empty((0, 2))

    p = np.array(p_list)

    # Convert from unit grid back to original coordinates
    p[:, 0] = (p[:, 0] + mx * np.min(xx.ravel())) / mx
    p[:, 1] = (p[:, 1] + my * np.min(yy.ravel())) / my

    # Compute grid spacing
    dx_sorted = np.sort(np.unique(xx.ravel()))
    dx_step = np.abs(dx_sorted[1] - dx_sorted[0])
    dy_sorted = np.sort(np.unique(yy.ravel()))
    dy_step = np.abs(dy_sorted[1] - dy_sorted[0])

    dp = fd_orig(p)
    p = p[dp < -1.0 * max(dx_step, dy_step), :]

    return p
