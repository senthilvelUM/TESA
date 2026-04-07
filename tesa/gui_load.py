"""
EBSD data loader and grain isolator.
All GUI dialogs replaced with function parameters.
"""

import os
import numpy as np
from scipy.spatial.distance import cdist
from .knnsearch2 import knnsearch2
from .inpoly import inpoly
from .hdrload import hdrload
# ctfload moved to not_needed/ — import lazily only if .ctf file is loaded
from .DecimatePoly import DecimatePoly
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union


def gui_load(ms, file_str, angle_cols=None, coord_cols=None, phase_col=None,
             csys_angle=90, remove_small_grains=False, min_grain_pixels=10,
             allowed_phases=None):
    """
    Load EBSD data and isolate grains.

    Reads EBSD data from a supported file format, normalizes pixel
    coordinates, removes stray pixels, identifies grains by connected-
    component analysis, and extracts ordered grain boundary polygons.
    This is the legacy loader with console output; see ``_parse_ebsd``
    in ``load_ebsd.py`` for the pipeline version.

    Parameters
    ----------
    ms : Microstructure
        Microstructure state object to populate in place.
    file_str : str
        Path to data file (.mat, .txt, .ang, or .ctf).
    angle_cols : list of int or None
        Column indices (0-based) for Euler angles [phi1, Phi, phi2].
        Default is [0, 1, 2] for .ang files.
    coord_cols : list of int or None
        Column indices (0-based) for x, y coordinates.
        Default is [3, 4] for .ang files.
    phase_col : int or None
        Column index (0-based) for phase number.
        Default is 5 for .ang files.
    csys_angle : float
        Orientation of reference frame in degrees. Default is 90.
    remove_small_grains : bool
        Whether to merge small grains into a neighbor. Default is False.
    min_grain_pixels : int
        Minimum number of data points per grain to keep. Default is 10.
    allowed_phases : list of int or None
        Phase numbers that small grains can be merged into. If None, all
        phases are allowed.

    Returns
    -------
    None
        The function modifies `ms` in place.
    """
    eps_val = np.sqrt(np.finfo(float).eps)

    # Get the filename extension
    _, FileName_with_ext = os.path.split(file_str)
    FileName, ext = os.path.splitext(FileName_with_ext)

    print("  Loading, please wait...")

    # Load .mat file (saved analysis)
    if ext == '.mat':
        import scipy.io
        mat_data = scipy.io.loadmat(file_str, squeeze_me=True)
        # Attempt to restore ms from mat file
        if 'ms' in mat_data:
            for key, val in mat_data['ms'].item()._fieldnames:
                setattr(ms, key, val)
        if ms.DisplayData is None or ms.DataCoordinateList is None:
            print("  Error: There is insufficient data in this file")
            return
        print("  Loading complete.")
        return

    # Load .txt file
    elif ext == '.txt':
        data = np.loadtxt(file_str)
        if data.size > 0:
            ms.DataEulerAngle = data[:, 0:3].copy()
            ms.DataCoordinateList = np.column_stack([
                data[:, 3],
                np.abs(-1.0 * data[:, 4] + np.max(data[:, 4]))
            ])
            ms.DataPhase = data[:, 5].astype(int)
            if np.any(ms.DataPhase == 0):
                ms.DataPhase = ms.DataPhase + 1
            ms.FileType = 'txt'
            ms.Filename = FileName + ext
            ms.NumberPhases = int(np.max(ms.DataPhase))
            ms.NumberDataPoints = ms.DataCoordinateList.shape[0]

        if ms.NumberPhases > 8:
            print("  Error: There can only be 8 distinct phases.")
            return

        # Find volume fractions
        ms.PhaseVolumeFraction = np.zeros(ms.NumberPhases)
        for i in range(ms.NumberPhases):
            ms.PhaseVolumeFraction[i] = np.sum(ms.DataPhase == (i + 1)) / len(ms.DataPhase)

        # Setup DisplayData
        x = ms.DataCoordinateList[:, 0]
        y = ms.DataCoordinateList[:, 1]
        xs = abs(x[1] - x[0]) / 2.0
        yt = np.unique(y)
        ys = abs(yt[1] - yt[0]) / 2.0
        xd = np.vstack([x - xs, x + xs, x + xs, x - xs, x - xs])
        yd = np.vstack([y - ys, y - ys, y + ys, y + ys, y - ys])
        colors_mat = np.array([ms.Colors[k] for k in range(len(ms.Colors))])
        c = colors_mat[ms.DataPhase - 1, :]
        c = c.reshape(1, len(ms.DataPhase), 3)
        ms.DisplayData = [xd, yd, c]

        # Setup DisplayDataIndices
        ms.PhaseColorValue = [None] * ms.NumberPhases
        ms.DisplayDataIndices = [None] * ms.NumberPhases
        for i in range(int(np.max(ms.DataPhase))):
            ms.PhaseColorValue[i] = ms.Colors[i]
            ms.DisplayDataIndices[i] = np.where(ms.DataPhase == (i + 1))[0]

        # Initialize phase attributes
        ms.PhaseName = [''] * ms.NumberPhases
        ms.PhasePropertyFilename = [''] * ms.NumberPhases
        ms.PhaseStiffnessMatrix = [None] * ms.NumberPhases
        ms.PhaseDensity = np.zeros(ms.NumberPhases)
        ms.PhaseAnisotropySphericalWaveSpeeds = [None] * ms.NumberPhases
        ms.PhaseAnisotropyAVP = [None] * ms.NumberPhases
        ms.PhaseAnisotropyAVSH = [None] * ms.NumberPhases
        ms.PhaseAnisotropyAVSV = [None] * ms.NumberPhases
        ms.PhaseAnisotropyMaxAVS = [None] * ms.NumberPhases

    # Load .ang file
    elif ext == '.ang':
        # Read data (skip header lines starting with #)
        lines = []
        with open(file_str, 'r') as fID:
            for line in fID:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    try:
                        vals = [float(v) for v in stripped.split()]
                        lines.append(vals)
                    except ValueError:
                        continue

        if len(lines) == 0:
            print("  Error: No data found in file.")
            return

        data = np.array(lines)

        if data.size > 0:
            # Column defaults (converting from MATLAB 1-based to Python 0-based)
            if angle_cols is None:
                angle_col = 0  # MATLAB default: 1
            else:
                angle_col = angle_cols[0]
            if coord_cols is None:
                coord_col = 3  # MATLAB default: 4
            else:
                coord_col = coord_cols[0]
            if phase_col is None:
                phase_col_idx = 5  # MATLAB default: 6
            else:
                phase_col_idx = phase_col

            # Select columns: angles (3), coords (2), phase (1)
            selected_cols = [angle_col, angle_col + 1, angle_col + 2,
                             coord_col, coord_col + 1, phase_col_idx]
            data = data[:, selected_cols]
            phase_col_idx = 5  # now phase is column 5 (0-based)

            # Sort by y coordinates, then by x (columns 4, 3 in 0-based)
            sort_idx = np.lexsort((data[:, 3], data[:, 4]))
            data = data[sort_idx, :]

            # Renumber phases sequentially
            uniquePhaseNum = np.unique(data[:, 5])
            for idx, val in enumerate(uniquePhaseNum):
                data[data[:, 5] == val, 5] = idx + 1

            # Force unit grid spacing in x and y
            if abs(data[0, 3] - data[1, 3]) != 1 and abs(data[0, 3] - data[1, 3]) > 0:
                sep = abs(data[0, 3] - data[1, 3])
                data[:, 3] = np.round(data[:, 3] / sep)

            y_vals = data[:, 4]
            y_unique_nonmin = y_vals[y_vals != np.min(y_vals)]
            if len(y_unique_nonmin) > 0:
                y_second = np.min(y_unique_nonmin)
                if abs(y_vals[0] - y_second) != 1 and abs(y_vals[0] - y_second) > 0:
                    sep = abs(y_vals[0] - y_second)
                    data[:, 4] = np.round(data[:, 4] / sep)

            # Select portion of map
            ipx = np.min(data[:, 3])
            fpx = np.max(data[:, 3])
            ncx = int(fpx - ipx + 1)

            ipy = np.min(data[:, 4])
            fpy = np.max(data[:, 4])
            ncy = int(fpy - ipy + 1)

            data = data[(data[:, 3] >= ipx - 0.5) & (data[:, 3] <= fpx + 0.5) &
                        (data[:, 4] >= ipy - 0.5) & (data[:, 4] <= fpy + 0.5), :]
            data[:, 3] = np.round(data[:, 3] - np.min(data[:, 3]))
            data[:, 4] = -1.0 * data[:, 4] + np.max(data[:, 4])
            data[:, 4] = np.round(data[:, 4] - np.min(data[:, 4]))

            if np.any(data[:, phase_col_idx] < eps_val):
                data[:, phase_col_idx] = data[:, phase_col_idx] + 1

            ms.DataEulerAngle = data[:, 0:3].copy()
            ms.DataCoordinateList = 1e-6 * np.round(1e6 * data[:, 3:5])
            ms.DataCoordinateList = 0.1 * np.round(10.0 * ms.DataCoordinateList) + 0.5
            ms.DataPhase = data[:, phase_col_idx].astype(int)
            ms.NumberDataPoints = data.shape[0]
            ms.NumberPhases = int(np.round(np.max(ms.DataPhase)))
            ms.FileType = 'ang'
            ms.Filename = FileName + ext

        if ms.NumberPhases > 8:
            print("  Error: There can only be 8 distinct phases.")
            return

        # Remove stray pixels
        a1 = ms.DataEulerAngle[:, 0].reshape(ncx, ncy, order='F').T
        a2 = ms.DataEulerAngle[:, 1].reshape(ncx, ncy, order='F').T
        a3 = ms.DataEulerAngle[:, 2].reshape(ncx, ncy, order='F').T
        ph = ms.DataPhase.reshape(ncx, ncy, order='F').T
        count = 0

        for i in range(1, ncy - 1):
            print(f"  Removing stray pixels {i + 2} {ncy}")
            for j in range(1, ncx - 1):
                mm = np.array([a1[i, j], a2[i, j], a3[i, j], ph[i, j]])
                um = np.array([a1[i - 1, j], a2[i - 1, j], a3[i - 1, j], ph[i - 1, j]])
                ur = np.array([a1[i - 1, j + 1], a2[i - 1, j + 1], a3[i - 1, j + 1], ph[i - 1, j + 1]])
                mr = np.array([a1[i, j + 1], a2[i, j + 1], a3[i, j + 1], ph[i, j + 1]])
                if (not (np.array_equal(mm, um) or np.array_equal(mm, mr))) and np.array_equal(mm, ur):
                    a1[i, j] = a1[i - 1, j]
                    a2[i, j] = a2[i - 1, j]
                    a3[i, j] = a3[i - 1, j]
                    ph[i, j] = ph[i - 1, j]
                    count += 1

                mm = np.array([a1[i, j], a2[i, j], a3[i, j], ph[i, j]])
                ul = np.array([a1[i - 1, j - 1], a2[i - 1, j - 1], a3[i - 1, j - 1], ph[i - 1, j - 1]])
                ml = np.array([a1[i, j - 1], a2[i, j - 1], a3[i, j - 1], ph[i, j - 1]])
                if (not (np.array_equal(mm, um) or np.array_equal(mm, ml))) and np.array_equal(mm, ul):
                    a1[i, j] = a1[i - 1, j]
                    a2[i, j] = a2[i - 1, j]
                    a3[i, j] = a3[i - 1, j]
                    ph[i, j] = ph[i - 1, j]
                    count += 1

                mm = np.array([a1[i, j], a2[i, j], a3[i, j], ph[i, j]])
                bm = np.array([a1[i + 1, j], a2[i + 1, j], a3[i + 1, j], ph[i + 1, j]])
                br = np.array([a1[i + 1, j + 1], a2[i + 1, j + 1], a3[i + 1, j + 1], ph[i + 1, j + 1]])
                if (not (np.array_equal(mm, bm) or np.array_equal(mm, mr))) and np.array_equal(mm, br):
                    a1[i, j] = a1[i + 1, j]
                    a2[i, j] = a2[i + 1, j]
                    a3[i, j] = a3[i + 1, j]
                    ph[i, j] = ph[i + 1, j]
                    count += 1

                mm = np.array([a1[i, j], a2[i, j], a3[i, j], ph[i, j]])
                bl = np.array([a1[i + 1, j - 1], a2[i + 1, j - 1], a3[i + 1, j - 1], ph[i + 1, j - 1]])
                if (not (np.array_equal(mm, bm) or np.array_equal(mm, ml))) and np.array_equal(mm, bl):
                    a1[i, j] = a1[i + 1, j]
                    a2[i, j] = a2[i + 1, j]
                    a3[i, j] = a3[i + 1, j]
                    ph[i, j] = ph[i + 1, j]
                    count += 1

        # Reshape back: MATLAB reshape(a1',[],1) = transpose then column-major flatten
        ms.DataEulerAngle = np.column_stack([a1.T.ravel(order='F'),
                                              a2.T.ravel(order='F'),
                                              a3.T.ravel(order='F')])
        ms.DataPhase = ph.T.ravel(order='F').astype(int)
        print(f"  A total of {count} stray pixels were changed.")

        # Find volume fractions
        ms.PhaseVolumeFraction = np.zeros(ms.NumberPhases)
        for i in range(ms.NumberPhases):
            ms.PhaseVolumeFraction[i] = np.sum(ms.DataPhase == (i + 1)) / len(ms.DataPhase)

        # Setup DisplayData
        x = ms.DataCoordinateList[:, 0]
        y = ms.DataCoordinateList[:, 1]
        xs = 0.5
        ys = 0.5
        xd = np.vstack([x - xs, x + xs, x + xs, x - xs, x - xs])
        yd = np.vstack([y - ys, y - ys, y + ys, y + ys, y - ys])
        colors_mat = np.array([ms.Colors[k] for k in range(len(ms.Colors))])
        c = colors_mat[ms.DataPhase - 1, :]
        c = c.reshape(1, len(ms.DataPhase), 3)
        ms.DisplayData = [xd, yd, c]

        # Setup DisplayDataIndices
        ms.PhaseColorValue = [None] * ms.NumberPhases
        ms.DisplayDataIndices = [None] * ms.NumberPhases
        for i in range(int(np.max(ms.DataPhase))):
            ms.PhaseColorValue[i] = ms.Colors[i]
            ms.DisplayDataIndices[i] = np.where(ms.DataPhase == (i + 1))[0]

        # Initialize phase attributes
        ms.PhaseName = [''] * ms.NumberPhases
        ms.PhasePropertyFilename = [''] * ms.NumberPhases
        ms.PhaseStiffnessMatrix = [None] * ms.NumberPhases
        ms.PhaseDensity = np.zeros(ms.NumberPhases)
        ms.PhaseAnisotropySphericalWaveSpeeds = [None] * ms.NumberPhases
        ms.PhaseAnisotropyAVP = [None] * ms.NumberPhases
        ms.PhaseAnisotropyAVSH = [None] * ms.NumberPhases
        ms.PhaseAnisotropyAVSV = [None] * ms.NumberPhases
        ms.PhaseAnisotropyMaxAVS = [None] * ms.NumberPhases

    elif ext == '.ctf':
        # CTF loading is commented out in original MATLAB; skip
        print("  CTF file loading not implemented in this version.")
        return

    else:
        print("  Error: The selected file is not valid for this analysis.")
        return

    print("  File loaded successfully.")

    # Save CSYS correction angle
    ms.CSYSAngle = csys_angle

    # -----------------------------------------------------------------------
    # Find grains based on phase and Euler angles
    # -----------------------------------------------------------------------
    print("  Isolating grains...")
    data = np.column_stack([ms.DataCoordinateList.astype(float),
                             ms.DataEulerAngle.astype(float),
                             ms.DataPhase.astype(float).reshape(-1, 1)])

    # Group data points by unique (euler1, euler2, euler3, phase) using a single pass.
    # Much faster than the original per-grain scan for maps with many grains.
    _grain_keys = data[:, 2:]
    _unique_keys, _inverse = np.unique(_grain_keys, axis=0, return_inverse=True)
    num_grains = _unique_keys.shape[0]
    grain_data = [None] * num_grains
    for i in range(num_grains):
        grain_data[i] = data[_inverse == i, :]

    # Split grain_data into grains (connected component splitting)
    if ext not in ['.png', '.bmp', '.ppm', '.jpg', '.eps']:
        print("  Splitting grains...")
        new_grain_data = []
        grain_count = 0
        for i in range(num_grains):
            print(f"  Splitting grains {i + 1} of {num_grains}")
            gxy = grain_data[i][:, 0:2]
            gid = np.arange(gxy.shape[0])
            aid = np.array([], dtype=int)

            while True:
                tid = np.setdiff1d(gid, aid)
                if len(tid) == 0:
                    break
                cxy = gxy[tid[0:1], :]
                cid = np.array([tid[0]])
                oxy = np.array([[np.nan, np.nan]])

                while True:
                    old_cid = cid.copy()
                    # setdiff rows
                    oxy_set = set(map(tuple, np.round(oxy, 12)))
                    cxy_mask = np.array([tuple(np.round(row, 12)) not in oxy_set for row in cxy])
                    cxy = cxy[cxy_mask] if np.any(cxy_mask) else cxy
                    oxy = cxy.copy()

                    nuid = np.setdiff1d(gid, cid)
                    if len(nuid) == 0:
                        grain_count += 1
                        new_grain_data.append(grain_data[i][cid, :])
                        aid = np.unique(np.concatenate([aid, cid]))
                        break

                    K = min(8, gxy[nuid, :].shape[0])
                    id_knn, d_knn = knnsearch2(cxy, gxy[nuid, :], K)

                    id_flat = id_knn.ravel()
                    d_flat = d_knn.ravel()
                    valid = d_flat < np.sqrt(2) + eps_val
                    nid = np.unique(nuid[id_flat[valid]])
                    nid = np.setdiff1d(nid, cid)

                    if nid.size > 0:
                        cxy = np.unique(np.vstack([cxy, gxy[nid, :]]), axis=0)
                    cid = np.unique(np.concatenate([cid, nid]))

                    if np.array_equal(np.sort(cid), np.sort(old_cid)):
                        grain_count += 1
                        new_grain_data.append(grain_data[i][cid, :])
                        aid = np.unique(np.concatenate([aid, cid]))
                        break

                if np.all(np.isin(gid, aid)):
                    break

        num_grains = len(new_grain_data)
        grain_data = new_grain_data

    # Find grain boundaries and order them
    ms.Grains = [None] * num_grains
    ms.AllGrainNodes = np.zeros((0, 2))
    print("  Ordering grain boundary points...")

    for i in range(num_grains):
        xg = grain_data[i][:, 0]
        yg = grain_data[i][:, 1]

        # Build corner coordinates of each pixel
        xcor = np.column_stack([xg - 0.5, xg + 0.5, xg + 0.5, xg - 0.5]).T
        ycor = np.column_stack([yg - 0.5, yg - 0.5, yg + 0.5, yg + 0.5]).T
        xcor = 0.1 * np.round(10.0 * xcor.ravel())
        ycor = 0.1 * np.round(10.0 * ycor.ravel())
        pcor = np.column_stack([xcor, ycor])
        porig = pcor.copy()

        # Remove points appearing 4 times (interior corners), keep boundary corners
        _, idx1 = np.unique(pcor, axis=0, return_index=True)
        mask1 = np.ones(pcor.shape[0], dtype=bool)
        mask1[idx1] = False
        pcor = pcor[mask1, :]

        _, idx2 = np.unique(pcor, axis=0, return_index=True)
        mask2 = np.ones(pcor.shape[0], dtype=bool)
        mask2[idx2] = False
        pcor = pcor[mask2, :]

        _, idx3 = np.unique(pcor, axis=0, return_index=True)
        mask3 = np.ones(pcor.shape[0], dtype=bool)
        mask3[idx3] = False
        pcor = pcor[mask3, :]

        # setdiff: keep points from porig not in pcor
        pcor_set = set(map(tuple, np.round(pcor, 12)))
        pcor_mask = np.array([tuple(np.round(row, 12)) not in pcor_set for row in porig])
        pcor = porig[pcor_mask, :]
        # unique
        pcor = np.unique(pcor, axis=0)

        # Mid-edge points
        xmid = np.column_stack([xg, xg + 0.5, xg, xg - 0.5]).T
        ymid = np.column_stack([yg - 0.5, yg, yg + 0.5, yg]).T
        xmid = 0.1 * np.round(10.0 * xmid.ravel())
        ymid = 0.1 * np.round(10.0 * ymid.ravel())
        pmid = np.column_stack([xmid, ymid])
        porig_mid = pmid.copy()

        _, idx_m = np.unique(pmid, axis=0, return_index=True)
        mask_m = np.ones(pmid.shape[0], dtype=bool)
        mask_m[idx_m] = False
        pmid = pmid[mask_m, :]

        # setdiff
        pmid_set = set(map(tuple, np.round(pmid, 12)))
        pmid_mask = np.array([tuple(np.round(row, 12)) not in pmid_set for row in porig_mid])
        pmid = porig_mid[pmid_mask, :]
        pmid = np.unique(pmid, axis=0)

        pall = np.unique(1e-6 * np.round(1e6 * np.vstack([pcor, pmid])), axis=0)

        # Order boundary points
        pord = -1.0 * np.ones_like(pall)
        ps = pall.copy()
        pord[0, :] = pall[0, :]
        pall[0, :] = 2e10
        dp = np.sqrt((pall[:, 0] - pord[0, 0]) ** 2 + (pall[:, 1] - pord[0, 1]) ** 2)
        ip = np.where(np.abs(dp - 0.5) < eps_val)[0]
        if len(ip) == 0:
            # Skip erroneous grain
            print(f"  Warning: Erroneous grain {i + 1}")
            ms.Grains[i] = np.zeros((0, 2))
            continue
        pord[1, :] = pall[ip[0], :]
        pall[ip[0], :] = 2e10

        for j in range(1, pord.shape[0] - 1):
            dp = np.sqrt((pall[:, 0] - pord[j, 0]) ** 2 + (pall[:, 1] - pord[j, 1]) ** 2)
            ip = np.where(np.abs(dp - 0.5) < eps_val)[0]
            if len(ip) == 0:
                break
            if len(ip) > 1:
                # Take the first one
                pord[j + 1, :] = pall[ip[0], :]
                pall[ip[0], :] = 2e10
            else:
                pord[j + 1, :] = pall[ip[0], :]
                pall[ip[0], :] = 2e10

        # Remove unused entries
        pord = pord[np.abs(pord[:, 0] + 1.0) >= eps_val, :]

        # Remove halfway points (keep only integer-coordinate points)
        mask_int = (np.abs(np.mod(pord[:, 0], 1)) < eps_val) & \
                   (np.abs(np.mod(pord[:, 1], 1)) < eps_val)
        pord = pord[mask_int, :]

        ms.Grains[i] = pord
        if pord.shape[0] > 0:
            ms.AllGrainNodes = np.vstack([ms.AllGrainNodes, pord])

    ms.AllGrainNodes = np.unique(ms.AllGrainNodes, axis=0)

    # Arrange grains smallest to largest
    gsize = np.zeros((num_grains, 2))
    for i in range(num_grains):
        gsize[i, :] = [ms.Grains[i].shape[0] if ms.Grains[i] is not None else 0, i]
    sort_idx = np.argsort(gsize[:, 0])
    gsize = gsize[sort_idx, :]
    G = [None] * num_grains
    minGrainPoints = np.inf
    maxGrainPoints = -np.inf
    for i in range(num_grains):
        G[i] = ms.Grains[int(gsize[i, 1])]
        if G[i] is not None and G[i].shape[0] > 0:
            id_in = inpoly(ms.DataCoordinateList, G[i])
            npts = np.sum(id_in)
            minGrainPoints = min(npts, minGrainPoints)
            maxGrainPoints = max(npts, maxGrainPoints)
    ms.Grains = G

    # Remove small grains
    if remove_small_grains:
        print(f"  Min. number of points in a grain = {minGrainPoints}")
        print(f"  Max. number of points in a grain = {maxGrainPoints}")
        if allowed_phases is None:
            allowed_phases = list(range(1, int(np.max(ms.DataPhase)) + 1))
        allowed_phases = np.array(allowed_phases)

        print("  Removing small grains...")
        ang = ms.DataEulerAngle.copy()
        ph_arr = ms.DataPhase.copy()
        rcount = 0

        Gb = [g.copy() if g is not None else None for g in ms.Grains]
        for i in range(num_grains - 1):
            if ms.Grains[i] is None or ms.Grains[i].shape[0] == 0:
                continue
            n_in = np.sum(inpoly(ms.DataCoordinateList, ms.Grains[i]))
            if n_in > min_grain_pixels:
                continue

            gc = np.mean(ms.Grains[i], axis=0)
            for j in range(i + 1, num_grains - 1):
                if Gb[j] is None or Gb[j].shape[0] == 0:
                    continue
                # Check if grains share boundary points or centroid is inside
                gi_set = set(map(tuple, np.round(ms.Grains[i], 12)))
                gbj_set = set(map(tuple, np.round(Gb[j], 12)))
                shared = len(gi_set & gbj_set)
                gc_in = inpoly(gc.reshape(1, -1), ms.Grains[j])
                if shared > 1 or (gc_in.size > 0 and gc_in[0]):
                    id_j = np.where(inpoly(ms.DataCoordinateList, Gb[j]))[0]
                    if len(id_j) == 0:
                        continue
                    from scipy.stats import mode as scipy_mode_fn
                    grain_phase = int(scipy_mode_fn(ms.DataPhase[id_j], keepdims=False).mode)
                    if grain_phase not in allowed_phases:
                        continue
                    new_ang = ang[id_j[0], :]
                    new_ph = ph_arr[id_j[0]]
                    id_i = inpoly(ms.DataCoordinateList, ms.Grains[i])
                    ms.DataEulerAngle[id_i, 0] = new_ang[0]
                    ms.DataEulerAngle[id_i, 1] = new_ang[1]
                    ms.DataEulerAngle[id_i, 2] = new_ang[2]
                    ms.DataPhase[id_i] = new_ph

                    # Union of polygons — matches MATLAB's polybool('union',...)
                    try:
                        poly_i = ShapelyPolygon(ms.Grains[i])
                        poly_j = ShapelyPolygon(ms.Grains[j])
                        union_poly = unary_union([poly_i, poly_j])
                        if union_poly.geom_type == 'Polygon':
                            coords_u = np.array(union_poly.exterior.coords[:-1])
                            ms.Grains[j] = coords_u
                        elif union_poly.geom_type == 'MultiPolygon':
                            largest = max(union_poly.geoms, key=lambda g: g.area)
                            coords_u = np.array(largest.exterior.coords[:-1])
                            ms.Grains[j] = coords_u
                        else:
                            ms.Grains[j] = Gb[j]
                    except Exception:
                        ms.Grains[j] = Gb[j]  # fallback if union fails
                    ms.Grains[i] = None
                    rcount += 1
                    break

            print(f"  Removing small grains {i + 1} {num_grains}")

        print(f"  {rcount} small grains removed.")

    # Rebuild grain list (remove None entries)
    G_new = []
    ms.AllGrainNodes = np.zeros((0, 2))
    for i in range(len(ms.Grains)):
        if ms.Grains[i] is not None and ms.Grains[i].shape[0] > 0:
            _, idx_u = np.unique(ms.Grains[i], axis=0, return_index=True)
            idx_u = np.sort(idx_u)
            G_new.append(ms.Grains[i][idx_u, :])

    # Interpolate along straight edges to add intermediate boundary points
    ms.Grains = [None] * len(G_new)
    for i in range(len(G_new)):
        g = G_new[i]
        pts = np.zeros((0, 2))
        for j in range(g.shape[0] - 1):
            if g[j, 0] == g[j + 1, 0]:
                y_interp = np.linspace(g[j, 1], g[j + 1, 1],
                                        int(abs(g[j + 1, 1] - g[j, 1])) + 1)
                x_interp = g[j, 0] * np.ones_like(y_interp)
                pts = np.vstack([pts, np.column_stack([x_interp, y_interp])])
            elif g[j, 1] == g[j + 1, 1]:
                x_interp = np.linspace(g[j, 0], g[j + 1, 0],
                                        int(abs(g[j + 1, 0] - g[j, 0])) + 1)
                y_interp = g[j, 1] * np.ones_like(x_interp)
                pts = np.vstack([pts, np.column_stack([x_interp, y_interp])])
        if pts.shape[0] > 0:
            _, idx_u = np.unique(pts, axis=0, return_index=True)
            idx_u = np.sort(idx_u)
            ms.Grains[i] = pts[idx_u, :]
        else:
            ms.Grains[i] = g
        ms.AllGrainNodes = np.vstack([ms.AllGrainNodes, ms.Grains[i]])

    ms.AllGrainNodes = np.unique(ms.AllGrainNodes, axis=0)

    # Arrange grains smallest to largest and remove duplicates
    num_grains = len(ms.Grains)
    gsize = np.zeros((num_grains, 2))
    for i in range(num_grains):
        gsize[i, :] = [ms.Grains[i].shape[0], i]
    sort_idx = np.argsort(gsize[:, 0])
    gsize = gsize[sort_idx, :]
    G_sorted = []
    for i in range(num_grains):
        G_sorted.append(ms.Grains[int(gsize[i, 1])])
    ms.Grains = G_sorted

    # Remove grains that don't contain any data points
    G_final = []
    xy = ms.DataCoordinateList.copy()
    for i in range(len(ms.Grains)):
        id_in = inpoly(xy, ms.Grains[i])
        if np.any(id_in):
            G_final.append(ms.Grains[i])
            xy = xy[~id_in, :]
    ms.Grains = G_final

    # Store Normalized Grains
    ms.NumberGrains = len(ms.Grains)
    ms.GrainsNormalized = [None] * ms.NumberGrains
    maxx_disp = np.max(ms.DisplayData[0]) if ms.DisplayData is not None else 1.0
    for i in range(len(ms.Grains)):
        closed = np.vstack([ms.Grains[i], ms.Grains[i][0:1, :]])
        dec, _ = DecimatePoly(closed, np.array([1e-9, 1.0]))
        ms.GrainsNormalized[i] = dec[:-1, :] / maxx_disp

    # Store original EBSD data
    ms.OriginalDataCoordinateList = ms.DataCoordinateList.copy()
    ms.OriginalDataEulerAngle = ms.DataEulerAngle.copy()
    ms.OriginalDataPhase = ms.DataPhase.copy()

    # Assign grain colors based on Euler Angles and phases
    ms.Colors = {0: np.array([1, 0, 0]), 1: np.array([0, 1, 0]), 2: np.array([0, 0, 1])}
    # Extend Colors for more phases
    for pi in range(3, ms.NumberPhases):
        ms.Colors[pi] = np.array([0.5, 0.5, 0.5])

    num_final = len(ms.Grains)
    ms.GrainColors = np.zeros((num_final, 3))
    grain_set = np.zeros((num_final, 4))

    xy = ms.DataCoordinateList.copy()
    ang_data = ms.DataEulerAngle.copy()
    ph_data = ms.DataPhase.copy()

    print("  Finding grain sets...")
    for i in range(num_final):
        print(f"  Finding grain sets {i + 1} {num_final}")
        id_in = np.where(inpoly(xy, ms.Grains[i]))[0]
        if len(id_in) > 0:
            grain_set[i, :] = np.concatenate([ang_data[id_in[0], :], [ph_data[id_in[0]]]])
            xy = np.delete(xy, id_in, axis=0)
            ang_data = np.delete(ang_data, id_in, axis=0)
            ph_data = np.delete(ph_data, id_in)

    csep = 1.0 / ms.NumberPhases if ms.NumberPhases > 0 else 1.0
    for i in range(ms.NumberPhases):
        id_ph = grain_set[:, 3] == (i + 1)
        if not np.any(id_ph):
            continue
        phset = grain_set[id_ph, 0:3].copy()
        max_val = np.max(phset)
        if max_val > 0:
            for j in range(3):
                phset[:, j] = phset[:, j] / max_val
        phset[:, 0:3] = phset[:, 0:3] * csep + i * csep
        ms.GrainColors[id_ph, :] = phset

    print("  Assigning grain colors...")
    for i in range(ms.NumberPhases):
        id_ph = grain_set[:, 3] == (i + 1)
        if not np.any(id_ph):
            continue
        unique_angles = np.unique(grain_set[id_ph, 0:3], axis=0)
        num_shades = unique_angles.shape[0]
        shades_vals = np.linspace(0.5, 1.0, num_shades)
        color_base = ms.Colors[i]
        shades = np.column_stack([color_base[0] * shades_vals,
                                   color_base[1] * shades_vals,
                                   color_base[2] * shades_vals])

        cgset = grain_set[id_ph, 0:3].copy()
        for j in range(cgset.shape[0]):
            print(f"  Assigning grain colors {i + 1} {j + 1} {cgset.shape[0]}")
            # Find which unique angle this row matches
            diffs = np.sum(np.abs(unique_angles - cgset[j, :]), axis=1)
            loc = np.argmin(diffs)
            cgset[j, :] = shades[loc, :]
        ms.GrainColors[id_ph, :] = cgset

    print("  gui_load complete.")
    return ms
