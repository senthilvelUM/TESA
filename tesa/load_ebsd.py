"""
EBSD data loader and grain isolator.

Public entry point:  load_ebsd(job, log_path=None)
Private core loader: _parse_ebsd(ms, file_str, ...)
"""

import io
import os
import contextlib
import numpy as np
from scipy.spatial.distance import cdist
from .knnsearch2 import knnsearch2
from .inpoly import inpoly
from .hdrload import hdrload
# ctfload moved to not_needed/ — import lazily only if .ctf file is loaded
from .DecimatePoly import DecimatePoly
from .Microstructure import Microstructure
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union


def _grain_has_holes(grain_xy):
    """
    Check if a grain's pixel region has interior holes.

    Builds a binary mask from grain pixel coordinates and uses
    binary_fill_holes to detect interior holes. Uses floor() to map
    half-integer EBSD coordinates to integer grid indices, avoiding
    banker's rounding artifacts.

    Parameters
    ----------
    grain_xy : ndarray, shape (n_pixels, 2)
        Grain pixel coordinates (x, y).

    Returns
    -------
    has_holes : bool
        True if the grain region contains interior holes.
    """
    from scipy.ndimage import binary_fill_holes
    xg = grain_xy[:, 0]
    yg = grain_xy[:, 1]
    ix_raw = np.floor(xg).astype(int)
    iy_raw = np.floor(yg).astype(int)
    x0, y0 = ix_raw.min() - 1, iy_raw.min() - 1
    x1, y1 = ix_raw.max() + 1, iy_raw.max() + 1
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    mask[iy_raw - y0, ix_raw - x0] = True
    filled = binary_fill_holes(mask)
    return np.any(filled & ~mask)


def _recompute_outer_boundary(grain_xy):
    """
    Recompute outer boundary points from a hole-filled pixel mask.

    For grains with interior holes, the original pixel set produces boundary
    points on both outer and inner (hole) boundaries. This function fills
    the holes with binary_fill_holes, then recomputes corners and mid-edges
    from the filled mask, yielding only outer boundary points.

    Parameters
    ----------
    grain_xy : ndarray, shape (n_pixels, 2)
        Grain pixel coordinates (x, y).

    Returns
    -------
    pall : ndarray, shape (M, 2)
        Outer boundary points in the original coordinate space.
    """
    from scipy.ndimage import binary_fill_holes
    xg = grain_xy[:, 0]
    yg = grain_xy[:, 1]
    # Build filled pixel mask (holes filled in)
    ix_raw = np.floor(xg).astype(int)
    iy_raw = np.floor(yg).astype(int)
    x0, y0 = ix_raw.min() - 1, iy_raw.min() - 1
    x1, y1 = ix_raw.max() + 1, iy_raw.max() + 1
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    mask[iy_raw - y0, ix_raw - x0] = True
    filled = binary_fill_holes(mask)
    # Extract filled pixel centers in original coordinate space
    fy, fx = np.where(filled)
    # Compute fractional offset (0.0 for integer coords, 0.5 for half-integer)
    frac_offset = xg[0] - np.floor(xg[0])
    xf = (fx + x0).astype(float) + frac_offset
    yf = (fy + y0).astype(float) + frac_offset
    # Recompute corner and mid-edge boundary points from filled pixel set
    # (same logic as the main boundary extraction loop)
    xcor = np.column_stack([xf - 0.5, xf + 0.5, xf + 0.5, xf - 0.5]).T
    ycor = np.column_stack([yf - 0.5, yf - 0.5, yf + 0.5, yf + 0.5]).T
    xcor = 0.1 * np.round(10.0 * xcor.ravel())
    ycor = 0.1 * np.round(10.0 * ycor.ravel())
    pcor = np.column_stack([xcor, ycor])
    _ucor, _inv_cor, _cnt_cor = np.unique(pcor, axis=0, return_inverse=True, return_counts=True)
    pcor = np.unique(pcor[_cnt_cor[_inv_cor] < 4], axis=0)
    xmid = np.column_stack([xf, xf + 0.5, xf, xf - 0.5]).T
    ymid = np.column_stack([yf - 0.5, yf, yf + 0.5, yf]).T
    xmid = 0.1 * np.round(10.0 * xmid.ravel())
    ymid = 0.1 * np.round(10.0 * ymid.ravel())
    pmid = np.column_stack([xmid, ymid])
    _umid, _inv_mid, _cnt_mid = np.unique(pmid, axis=0, return_inverse=True, return_counts=True)
    pmid = np.unique(pmid[_cnt_mid[_inv_mid] == 1], axis=0)
    pall = np.unique(1e-6 * np.round(1e6 * np.vstack([pcor, pmid])), axis=0)
    return pall


def _parse_ebsd(ms, file_str, angle_cols=None, coord_cols=None, phase_col=None,
              csys_angle=90, remove_small_grains=False, min_grain_pixels=10,
              allowed_phases=None, verbose=False):
    """
    Core EBSD loader -- parse file, normalize grid, and isolate grains.

    Reads EBSD data from a supported file format, normalizes pixel
    coordinates to unit spacing, removes stray pixels, identifies grains
    by connected-component analysis, and extracts ordered grain boundary
    polygons. Populates the Microstructure object with all grain and
    phase data.

    Parameters
    ----------
    ms : Microstructure
        Microstructure state object to populate.
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
        Whether to merge grains with fewer than `min_grain_pixels` pixels
        into a neighbor. Default is False.
    min_grain_pixels : int
        Minimum number of data points per grain to keep. Default is 10.
    allowed_phases : list of int or None
        Phase numbers that small grains can be merged into. If None, all
        phases are allowed.
    verbose : bool
        If True, print progress messages to console. Default is False.

    Returns
    -------
    None
        The function modifies `ms` in place.
    """
    eps_val = np.sqrt(np.finfo(float).eps)

    # Get the filename extension
    _, FileName_with_ext = os.path.split(file_str)
    FileName, ext = os.path.splitext(FileName_with_ext)

    if verbose:
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
        ms.PhaseThermalExpansionMatrix = [None] * ms.NumberPhases
        ms.PhaseThermalConductivityMatrix = [None] * ms.NumberPhases
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

            # Extract EBSD step size in physical units (for reporting only — not used in analysis)
            # Read raw x/y spacing before coordinates are normalised to unit pixel spacing.
            _raw_x = abs(data[0, 3] - data[1, 3]) if data.shape[0] > 1 else 0.0
            _raw_y_diffs = np.diff(np.unique(data[:, 4]))
            _raw_y = float(_raw_y_diffs[0]) if len(_raw_y_diffs) > 0 else 0.0
            ms.EBSDStepSize = _raw_x if _raw_x > 0 else (_raw_y if _raw_y > 0 else 1.0)

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

        if verbose:
            print(f"  Parsed {data.shape[0]} data points ({ncx} x {ncy}), {ms.NumberPhases} phases")

        if ms.NumberPhases > 8:
            print("  Error: There can only be 8 distinct phases.")
            return

        if verbose:
            print("  Removing stray pixels...")

        # Remove stray pixels (vectorized)
        # A stray pixel matches a diagonal neighbor but not the adjacent
        # horizontal/vertical neighbors — this removes staircase artifacts
        a1 = ms.DataEulerAngle[:, 0].reshape(ncx, ncy, order='F').T
        a2 = ms.DataEulerAngle[:, 1].reshape(ncx, ncy, order='F').T
        a3 = ms.DataEulerAngle[:, 2].reshape(ncx, ncy, order='F').T
        ph = ms.DataPhase.reshape(ncx, ncy, order='F').T
        count = 0

        # Stack into (ncy, ncx, 4) array for vectorized comparison
        _data = np.stack([a1, a2, a3, ph.astype(float)], axis=-1)

        # Interior slices: [1:-1, 1:-1]
        mm = _data[1:-1, 1:-1]  # center pixel
        um = _data[:-2, 1:-1]   # up-middle
        bm = _data[2:, 1:-1]    # below-middle
        ml = _data[1:-1, :-2]   # middle-left
        mr = _data[1:-1, 2:]    # middle-right
        ur = _data[:-2, 2:]     # up-right
        ul = _data[:-2, :-2]    # up-left
        br = _data[2:, 2:]      # below-right
        bl = _data[2:, :-2]     # below-left

        # Helper: check if all 4 components match (vectorized element-wise)
        def _eq(a, b):
            return np.all(a == b, axis=-1)

        # Check 1: matches up-right but not up-middle or middle-right → replace with up-middle
        mask1 = _eq(mm, ur) & ~_eq(mm, um) & ~_eq(mm, mr)
        if np.any(mask1):
            a1[1:-1, 1:-1][mask1] = a1[:-2, 1:-1][mask1]
            a2[1:-1, 1:-1][mask1] = a2[:-2, 1:-1][mask1]
            a3[1:-1, 1:-1][mask1] = a3[:-2, 1:-1][mask1]
            ph[1:-1, 1:-1][mask1] = ph[:-2, 1:-1][mask1]
            count += int(np.sum(mask1))
            # Refresh center after modification
            _data[1:-1, 1:-1] = np.stack([a1[1:-1, 1:-1], a2[1:-1, 1:-1],
                                           a3[1:-1, 1:-1], ph[1:-1, 1:-1].astype(float)], axis=-1)
            mm = _data[1:-1, 1:-1]

        # Check 2: matches up-left but not up-middle or middle-left → replace with up-middle
        mask2 = _eq(mm, ul) & ~_eq(mm, um) & ~_eq(mm, ml)
        if np.any(mask2):
            a1[1:-1, 1:-1][mask2] = a1[:-2, 1:-1][mask2]
            a2[1:-1, 1:-1][mask2] = a2[:-2, 1:-1][mask2]
            a3[1:-1, 1:-1][mask2] = a3[:-2, 1:-1][mask2]
            ph[1:-1, 1:-1][mask2] = ph[:-2, 1:-1][mask2]
            count += int(np.sum(mask2))
            _data[1:-1, 1:-1] = np.stack([a1[1:-1, 1:-1], a2[1:-1, 1:-1],
                                           a3[1:-1, 1:-1], ph[1:-1, 1:-1].astype(float)], axis=-1)
            mm = _data[1:-1, 1:-1]

        # Check 3: matches below-right but not below-middle or middle-right → replace with below-middle
        mask3 = _eq(mm, br) & ~_eq(mm, bm) & ~_eq(mm, mr)
        if np.any(mask3):
            a1[1:-1, 1:-1][mask3] = a1[2:, 1:-1][mask3]
            a2[1:-1, 1:-1][mask3] = a2[2:, 1:-1][mask3]
            a3[1:-1, 1:-1][mask3] = a3[2:, 1:-1][mask3]
            ph[1:-1, 1:-1][mask3] = ph[2:, 1:-1][mask3]
            count += int(np.sum(mask3))
            _data[1:-1, 1:-1] = np.stack([a1[1:-1, 1:-1], a2[1:-1, 1:-1],
                                           a3[1:-1, 1:-1], ph[1:-1, 1:-1].astype(float)], axis=-1)
            mm = _data[1:-1, 1:-1]

        # Check 4: matches below-left but not below-middle or middle-left → replace with below-middle
        mask4 = _eq(mm, bl) & ~_eq(mm, bm) & ~_eq(mm, ml)
        if np.any(mask4):
            a1[1:-1, 1:-1][mask4] = a1[2:, 1:-1][mask4]
            a2[1:-1, 1:-1][mask4] = a2[2:, 1:-1][mask4]
            a3[1:-1, 1:-1][mask4] = a3[2:, 1:-1][mask4]
            ph[1:-1, 1:-1][mask4] = ph[2:, 1:-1][mask4]
            count += int(np.sum(mask4))

        # Reshape back: MATLAB reshape(a1',[],1) = transpose then column-major flatten
        ms.DataEulerAngle = np.column_stack([a1.T.ravel(order='F'),
                                              a2.T.ravel(order='F'),
                                              a3.T.ravel(order='F')])
        ms.DataPhase = ph.T.ravel(order='F').astype(int)
        if verbose:
            print(f"  Stray pixels: {count} changed")

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
        ms.PhaseThermalExpansionMatrix = [None] * ms.NumberPhases
        ms.PhaseThermalConductivityMatrix = [None] * ms.NumberPhases
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

    if verbose:
        print("  File loaded successfully.")

    # Save CSYS correction angle
    ms.CSYSAngle = csys_angle

    # -----------------------------------------------------------------------
    # Find grains based on phase and Euler angles
    # -----------------------------------------------------------------------
    if verbose:
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

    if verbose:
        print(f"  Found {num_grains} unique grain orientations")

    # Split grain_data into grains (connected component splitting)
    if ext not in ['.png', '.bmp', '.ppm', '.jpg', '.eps']:
        if verbose:
            print("  Splitting disconnected grains...")
        from scipy.spatial import cKDTree as _cKDTree
        new_grain_data = []
        grain_count = 0
        _sqrt2_tol = np.sqrt(2) + eps_val
        for i in range(num_grains):
            if verbose and (i % 50 == 0 or i == num_grains - 1):
                print(f"    Grain {i+1}/{num_grains}...", end='\r')
            gxy = grain_data[i][:, 0:2]
            n_pts = gxy.shape[0]

            # Fast path: small grains don't need splitting
            if n_pts <= 2:
                grain_count += 1
                new_grain_data.append(grain_data[i])
                continue

            # Build KD-tree once for all points in this grain
            tree = _cKDTree(gxy)

            # Find connected components using KD-tree neighbor queries
            assigned = np.zeros(n_pts, dtype=bool)

            while not np.all(assigned):
                # Start a new component from the first unassigned point
                unassigned_idx = np.where(~assigned)[0]
                seed = unassigned_idx[0]
                component = set([seed])
                frontier = set([seed])

                while frontier:
                    new_frontier = set()
                    # Query neighbors of all frontier points at once
                    frontier_pts = gxy[list(frontier)]
                    # Find all points within sqrt(2) distance of any frontier point
                    neighbors_list = tree.query_ball_point(frontier_pts, _sqrt2_tol)

                    for neighbors in neighbors_list:
                        for nb in neighbors:
                            if nb not in component:
                                component.add(nb)
                                new_frontier.add(nb)

                    frontier = new_frontier

                # Store this connected component
                component_idx = np.array(sorted(component))
                assigned[component_idx] = True
                grain_count += 1
                new_grain_data.append(grain_data[i][component_idx, :])

        num_grains = len(new_grain_data)
        grain_data = new_grain_data
        if verbose:
            print(f"  Split into {num_grains} grains")

    # Find grain boundaries and order them
    ms.Grains = [None] * num_grains
    ms.AllGrainNodes = np.zeros((0, 2))
    if verbose:
        print(f"  Ordering grain boundary points ({num_grains} grains)...")

    import time as _time_mod_order
    for i in range(num_grains):
        _n_grain_pts = grain_data[i].shape[0]
        if verbose:
            print(f"\r    Grain {i+1}/{num_grains} ({_n_grain_pts} pts)...                              ", end='', flush=True)
            _t_grain = _time_mod_order.time()
        xg = grain_data[i][:, 0]
        yg = grain_data[i][:, 1]

        # Build corner coordinates of each pixel
        xcor = np.column_stack([xg - 0.5, xg + 0.5, xg + 0.5, xg - 0.5]).T
        ycor = np.column_stack([yg - 0.5, yg - 0.5, yg + 0.5, yg + 0.5]).T
        xcor = 0.1 * np.round(10.0 * xcor.ravel())
        ycor = 0.1 * np.round(10.0 * ycor.ravel())
        pcor = np.column_stack([xcor, ycor])

        # Keep only boundary corners: points that appear 1, 2, or 3 times
        # (interior corners appear exactly 4 times — shared by 4 pixels)
        # Use return_counts for O(N log N) instead of 3× unique + mask
        _ucor, _inv_cor, _cnt_cor = np.unique(pcor, axis=0, return_inverse=True, return_counts=True)
        _boundary_mask = _cnt_cor[_inv_cor] < 4
        pcor = np.unique(pcor[_boundary_mask], axis=0)

        # Mid-edge points (centers of pixel edges)
        xmid = np.column_stack([xg, xg + 0.5, xg, xg - 0.5]).T
        ymid = np.column_stack([yg - 0.5, yg, yg + 0.5, yg]).T
        xmid = 0.1 * np.round(10.0 * xmid.ravel())
        ymid = 0.1 * np.round(10.0 * ymid.ravel())
        pmid = np.column_stack([xmid, ymid])

        # Keep only boundary mid-edge points: points appearing once
        # (interior mid-edge points appear twice — shared by 2 pixels)
        _umid, _inv_mid, _cnt_mid = np.unique(pmid, axis=0, return_inverse=True, return_counts=True)
        pmid = np.unique(pmid[_cnt_mid[_inv_mid] == 1], axis=0)

        pall = np.unique(1e-6 * np.round(1e6 * np.vstack([pcor, pmid])), axis=0)

        # For grains with holes, recompute boundary from the filled (hole-free)
        # pixel mask. This removes all inner boundary points (adjacent to hole
        # grains) that would create bridge artifacts in the boundary polygon.
        if pall.shape[0] > 6 and _grain_has_holes(grain_data[i][:, 0:2]):
            _pall_before = pall.shape[0]
            pall = _recompute_outer_boundary(grain_data[i][:, 0:2])
            if verbose:
                print(f"\n    Grain {i+1}: has holes — recomputed {_pall_before} → {pall.shape[0]} outer boundary points")

        # Order boundary points using hash-based neighbor lookup (O(N) instead of O(N²))
        # Boundary points are on a 0.5-spaced grid; neighbors are at distance exactly 0.5
        _half = 0.5
        _neighbors_dx = [0, _half, 0, -_half]
        _neighbors_dy = [-_half, 0, _half, 0]

        # Build a set of available points for O(1) lookup
        _avail = {}
        for _k in range(pall.shape[0]):
            key = (round(pall[_k, 0] * 10) / 10, round(pall[_k, 1] * 10) / 10)
            _avail[key] = _k

        pord = -1.0 * np.ones_like(pall)
        pord[0, :] = pall[0, :]
        _start_key = (round(pall[0, 0] * 10) / 10, round(pall[0, 1] * 10) / 10)
        del _avail[_start_key]

        # Find first neighbor
        _found_first = False
        for _di in range(4):
            _nkey = (round((pord[0, 0] + _neighbors_dx[_di]) * 10) / 10,
                     round((pord[0, 1] + _neighbors_dy[_di]) * 10) / 10)
            if _nkey in _avail:
                _idx = _avail.pop(_nkey)
                pord[1, :] = pall[_idx, :]
                _found_first = True
                break

        if not _found_first:
            if verbose:
                print(f"  Warning: Erroneous grain {i + 1}")
            ms.Grains[i] = np.zeros((0, 2))
            continue

        # Chain through neighbors using hash lookup
        for j in range(1, pord.shape[0] - 1):
            _found = False
            for _di in range(4):
                _nkey = (round((pord[j, 0] + _neighbors_dx[_di]) * 10) / 10,
                         round((pord[j, 1] + _neighbors_dy[_di]) * 10) / 10)
                if _nkey in _avail:
                    _idx = _avail.pop(_nkey)
                    pord[j + 1, :] = pall[_idx, :]
                    _found = True
                    break
            if not _found:
                break

        # Remove unused entries
        pord = pord[np.abs(pord[:, 0] + 1.0) >= eps_val, :]

        # Remove halfway points (keep only integer-coordinate points)
        mask_int = (np.abs(np.mod(pord[:, 0], 1)) < eps_val) & \
                   (np.abs(np.mod(pord[:, 1], 1)) < eps_val)
        pord = pord[mask_int, :]

        ms.Grains[i] = pord
        if verbose:
            _elapsed_grain = _time_mod_order.time() - _t_grain
            print(f"\r    Grain {i+1}/{num_grains} ({_n_grain_pts} pts) → {pord.shape[0]} boundary pts ({_elapsed_grain:.1f}s)        ", end='', flush=True)
            if i == num_grains - 1:
                print()  # final newline after last grain

    # Build AllGrainNodes in one shot (avoid incremental vstack in loop)
    _all_grain_parts = [g for g in ms.Grains if g is not None and g.shape[0] > 0]
    ms.AllGrainNodes = np.unique(np.vstack(_all_grain_parts), axis=0) if _all_grain_parts else np.zeros((0, 2))

    if verbose:
        print("  Sorting grains by size...")

    # Arrange grains smallest to largest
    gsize = np.zeros((num_grains, 2))
    for i in range(num_grains):
        gsize[i, :] = [ms.Grains[i].shape[0] if ms.Grains[i] is not None else 0, i]
    sort_idx = np.argsort(gsize[:, 0])
    gsize = gsize[sort_idx, :]
    G = [None] * num_grains
    minGrainPoints = np.inf
    maxGrainPoints = -np.inf
    # Count points per grain from grain_data (already known — no need for inpoly)
    for i in range(num_grains):
        orig_idx = int(gsize[i, 1])
        G[i] = ms.Grains[orig_idx]
        if G[i] is not None and G[i].shape[0] > 0:
            npts = grain_data[orig_idx].shape[0]
            minGrainPoints = min(npts, minGrainPoints)
            maxGrainPoints = max(npts, maxGrainPoints)
    ms.Grains = G

    # Remove small grains
    if remove_small_grains:
        if verbose:
            print(f"  Min. number of points in a grain = {minGrainPoints}")
            print(f"  Max. number of points in a grain = {maxGrainPoints}")
        if allowed_phases is None:
            allowed_phases = list(range(1, int(np.max(ms.DataPhase)) + 1))
        allowed_phases = np.array(allowed_phases)

        if verbose:
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

            pass  # per-grain detail suppressed

        if verbose:
            print(f"  {rcount} small grains removed.")

    # Rebuild grain list (remove None entries)
    G_new = []
    ms.AllGrainNodes = np.zeros((0, 2))
    for i in range(len(ms.Grains)):
        if ms.Grains[i] is not None and ms.Grains[i].shape[0] > 0:
            _, idx_u = np.unique(ms.Grains[i], axis=0, return_index=True)
            idx_u = np.sort(idx_u)
            G_new.append(ms.Grains[i][idx_u, :])

    # Interpolate along straight edges to add intermediate boundary points.
    # Collect segments into a list first, then concatenate once (avoid repeated vstack).
    ms.Grains = [None] * len(G_new)
    for i in range(len(G_new)):
        g = G_new[i]
        _segments = []
        for j in range(g.shape[0] - 1):
            if g[j, 0] == g[j + 1, 0]:
                y_interp = np.linspace(g[j, 1], g[j + 1, 1],
                                        int(abs(g[j + 1, 1] - g[j, 1])) + 1)
                x_interp = g[j, 0] * np.ones_like(y_interp)
                _segments.append(np.column_stack([x_interp, y_interp]))
            elif g[j, 1] == g[j + 1, 1]:
                x_interp = np.linspace(g[j, 0], g[j + 1, 0],
                                        int(abs(g[j + 1, 0] - g[j, 0])) + 1)
                y_interp = g[j, 1] * np.ones_like(x_interp)
                _segments.append(np.column_stack([x_interp, y_interp]))
        if _segments:
            pts = np.vstack(_segments)
            _, idx_u = np.unique(pts, axis=0, return_index=True)
            idx_u = np.sort(idx_u)
            ms.Grains[i] = pts[idx_u, :]
        else:
            ms.Grains[i] = g

    # Build AllGrainNodes in one shot (avoid incremental vstack in loop)
    _all_grain_parts2 = [g for g in ms.Grains if g is not None and g.shape[0] > 0]
    ms.AllGrainNodes = np.unique(np.vstack(_all_grain_parts2), axis=0) if _all_grain_parts2 else np.zeros((0, 2))

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
    # Keep grains with non-empty boundaries (boundary ordering already removed erroneous grains)
    G_final = [g for g in ms.Grains if g is not None and g.shape[0] >= 3]
    if verbose:
        print(f"  Filtered: {len(G_final)} grains with valid boundaries (removed {len(ms.Grains) - len(G_final)})")
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

    # For each grain, find Euler angles and phase by looking up the nearest data point
    # to the grain centroid (fast — avoids inpoly on 442K points)
    from scipy.spatial import cKDTree as _cKDTree_gs
    _xy_tree = _cKDTree_gs(ms.DataCoordinateList)

    if verbose:
        print(f"  Assigning grain properties ({num_final} grains)...")
    for i in range(num_final):
        # Centroid of grain boundary
        gc = np.mean(ms.Grains[i], axis=0)
        _, nn_idx = _xy_tree.query(gc)
        grain_set[i, :] = np.concatenate([ms.DataEulerAngle[nn_idx, :],
                                           [ms.DataPhase[nn_idx]]])

    if verbose:
        print()
        print("  Computing grain colors...")
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

    if verbose:
        print("  Assigning grain color shades...")
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
            pass  # per-grain detail suppressed
            # Find which unique angle this row matches
            diffs = np.sum(np.abs(unique_angles - cgset[j, :]), axis=1)
            loc = np.argmin(diffs)
            cgset[j, :] = shades[loc, :]
        ms.GrainColors[id_ph, :] = cgset

    if verbose:
        print("  EBSD loading complete.")
    return ms


def load_ebsd(job, run_dir=None, log_path=None, settings=None):
    """
    Load an EBSD data file, generate Stage 1 figures, and return a
    populated Microstructure object.

    This is the public entry point called from run_tesa.py. It extracts
    settings from the job dictionary, creates a Microstructure, calls the
    core loader (_parse_ebsd), generates visualization figures, prints a
    summary, and writes to the log file. Grain coloring data is stored
    on ms for use by later pipeline stages.

    Parameters
    ----------
    job : dict
        Job dictionary with keys: ebsd_file, euler_col, xy_col, phase_col, etc.
    run_dir : str or None
        Results directory for saving figures. If None, figures are skipped.
    log_path : str or None
        Path to log.md file. If provided, Stage 1 results are appended.
    settings : dict or None
        Global settings (show_figures, verbose, etc.). If None, defaults are used.

    Returns
    -------
    ms : Microstructure
        Populated with: Grains, OriginalDataCoordinateList,
        OriginalDataPhase, OriginalDataEulerAngle, NumberPhases,
        grain_color (callable), grain_color_idx (dict), N_CYCLE (int), etc.
    """
    # Use defaults if settings not provided
    if settings is None:
        settings = {}
    if "verbose_console" not in settings:
        print("  WARNING: 'verbose_console' not in settings, using default 'medium'")
    vc = settings.get("verbose_console", "medium")
    if "verbose_log" not in settings:
        print("  WARNING: 'verbose_log' not in settings, using default 'medium'")
    vl = settings.get("verbose_log", "medium")
    console_on = vc in ("medium", "high")
    console_high = vc == "high"

    # Extract settings from job dictionary
    ebsd_file = job["ebsd_file"]
    euler_col = job["euler_col"]
    xy_col    = job["xy_col"]
    phase_col = job["phase_col"]

    # Convert from 1-based (user-facing) to 0-based (Python internal)
    euler_col_0 = euler_col - 1
    xy_col_0    = xy_col - 1
    phase_col_0 = phase_col - 1

    # Extract optional grain filtering parameters
    remove_small_grains = job.get("remove_small_grains", False)
    min_grain_pixels    = job.get("min_grain_pixels", 10)

    # ── Read phase property files first (quick validation) ──
    phase_properties = job.get("phase_properties", {})
    if phase_properties:
        from .read_properties import read_properties
        # Create a temporary ms just to hold properties; will transfer after EBSD load
        _tmp_ms = Microstructure()
        _tmp_ms.NumberPhases = len(phase_properties)
        _tmp_ms.PhaseName = [''] * _tmp_ms.NumberPhases
        _tmp_ms.PhasePropertyFilename = [''] * _tmp_ms.NumberPhases
        _tmp_ms.PhaseStiffnessMatrix = [None] * _tmp_ms.NumberPhases
        _tmp_ms.PhaseThermalExpansionMatrix = [None] * _tmp_ms.NumberPhases
        _tmp_ms.PhaseThermalConductivityMatrix = [None] * _tmp_ms.NumberPhases
        _tmp_ms.PhaseDensity = np.zeros(_tmp_ms.NumberPhases)
        _tmp_ms = read_properties(_tmp_ms, phase_properties,
                                   log_path=log_path, settings=settings)

    if console_on:
        print("\n── Stage 1: Load EBSD data ──")

    # Create Microstructure and run core loader with progress messages
    import time as _time_mod
    _t0 = _time_mod.time()

    ms = Microstructure()
    ms = _parse_ebsd(ms, ebsd_file,
                    angle_cols=[euler_col_0, euler_col_0 + 1, euler_col_0 + 2],
                    coord_cols=[xy_col_0, xy_col_0 + 1],
                    phase_col=phase_col_0,
                    remove_small_grains=remove_small_grains,
                    min_grain_pixels=min_grain_pixels,
                    verbose=console_on)

    _elapsed = _time_mod.time() - _t0
    n_grains = len([g for g in ms.Grains if g is not None and np.asarray(g).size > 0])
    n_pts = ms.OriginalDataCoordinateList.shape[0]
    if console_on:
        print(f"  Parsed {n_pts} data points, {n_grains} grains ({_elapsed:.1f}s)")

    # Store EBSD filename on ms for use in figure titles
    ms.ebsd_file = ebsd_file

    # Extract summary data
    coords = ms.OriginalDataCoordinateList
    phases = ms.OriginalDataPhase
    eulers = ms.OriginalDataEulerAngle
    n_pts = len(phases)
    unique_phases = np.unique(phases)
    n_grains = len(ms.Grains)
    nx = len(np.unique(coords[:, 0]))
    ny = len(np.unique(coords[:, 1]))

    # Per-phase statistics
    phase_counts = {}
    for ph in sorted(unique_phases.astype(int)):
        n_ph = int(np.sum(phases == ph))
        phase_counts[ph] = n_ph

    # Print summary to console
    if console_on:
        print(f"  EBSD grid   : {nx} x {ny}")
        print(f"  Data points : {n_pts}")
        print(f"  Phases      : {len(unique_phases)}")
        print(f"  Grains      : {n_grains}")
        print(f"  X range     : [{coords[:,0].min():.2f}, {coords[:,0].max():.2f}]")
        print(f"  Y range     : [{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]")

    # Print detailed info if console is "high"
    if console_high:
        for ph, n_ph in phase_counts.items():
            print(f"  Phase {ph}     : {n_ph} pts ({100*n_ph/n_pts:.1f}%)")
        print(f"  Euler range : phi1=[{eulers[:,0].min():.4f}, {eulers[:,0].max():.4f}]")
        print(f"                Phi =[{eulers[:,1].min():.4f}, {eulers[:,1].max():.4f}]")
        print(f"                phi2=[{eulers[:,2].min():.4f}, {eulers[:,2].max():.4f}]")
        # Per-grain summary (single updating line)
        print(f"  ---- Grain sizes ----")
        _n_valid_grains = len([g for g in ms.Grains if g is not None and len(g) > 0])
        for i, g in enumerate(ms.Grains):
            if g is not None and len(g) > 0:
                g_arr = np.asarray(g)
                n_verts = g_arr.shape[0]
                centroid = g_arr.mean(axis=0)
                print(f"\r  Grain {i:3d}/{_n_valid_grains}: {n_verts:4d} boundary pts, "
                      f"centroid=({centroid[0]:.2f}, {centroid[1]:.2f})        ", end='', flush=True)
        print()  # final newline

    # Append Stage 1 summary to log.md
    if log_path is not None and vl in ("medium", "high"):
        with open(log_path, "a") as lf:
            lf.write("## Stage 1 — Load EBSD Data\n\n")
            lf.write(f"| {'Property':<14s} | {'Value':<30s} |\n")
            lf.write(f"|{'-'*16}|{'-'*32}|\n")
            lf.write(f"| {'EBSD grid':<14s} | {f'{nx} x {ny}':<30s} |\n")
            lf.write(f"| {'EBSD points':<14s} | {n_pts:<30,d} |\n")
            lf.write(f"| {'Phases':<14s} | {len(unique_phases):<30d} |\n")
            lf.write(f"| {'Grains':<14s} | {n_grains:<30d} |\n")
            lf.write(f"| {'X range':<14s} | {f'[{coords[:,0].min():.2f}, {coords[:,0].max():.2f}]':<30s} |\n")
            lf.write(f"| {'Y range':<14s} | {f'[{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]':<30s} |\n")

            # Add Euler angle ranges for "high" log (phase stats moved to merged table below)
            if vl == "high":
                lf.write(f"\n**Euler angle ranges (radians):**\n\n")
                lf.write(f"| {'Angle':<8s} | {'Min':>10s} | {'Max':>10s} |\n")
                lf.write(f"|{'-'*10}|{'-'*12}|{'-'*12}|\n")
                lf.write(f"| {'phi1':<8s} | {eulers[:,0].min():>10.4f} | {eulers[:,0].max():>10.4f} |\n")
                lf.write(f"| {'Phi':<8s} | {eulers[:,1].min():>10.4f} | {eulers[:,1].max():>10.4f} |\n")
                lf.write(f"| {'phi2':<8s} | {eulers[:,2].min():>10.4f} | {eulers[:,2].max():>10.4f} |\n")

                lf.write(f"\n**Grain boundary vertices:**\n\n")
                lf.write(f"| {'Grain':>6s} | {'Vertices':>10s} | {'Centroid X':>11s} | {'Centroid Y':>11s} |\n")
                lf.write(f"|{'-'*8}|{'-'*12}|{'-'*13}|{'-'*13}|\n")
                for i, g in enumerate(ms.Grains):
                    if g is not None and len(g) > 0:
                        g_arr = np.asarray(g)
                        cx, cy = g_arr.mean(axis=0)
                        lf.write(f"| {i:>6d} | {g_arr.shape[0]:>10d} | {cx:>11.2f} | {cy:>11.2f} |\n")

            lf.write("\n")

    # Transfer phase properties from pre-loaded _tmp_ms to the real ms
    if phase_properties and '_tmp_ms' in dir():
        for i in range(min(len(phase_properties), ms.NumberPhases)):
            ms.PhaseName[i] = _tmp_ms.PhaseName[i]
            ms.PhasePropertyFilename[i] = _tmp_ms.PhasePropertyFilename[i]
            ms.PhaseStiffnessMatrix[i] = _tmp_ms.PhaseStiffnessMatrix[i]
            ms.PhaseThermalExpansionMatrix[i] = _tmp_ms.PhaseThermalExpansionMatrix[i]
            ms.PhaseThermalConductivityMatrix[i] = _tmp_ms.PhaseThermalConductivityMatrix[i]
            ms.PhaseDensity[i] = _tmp_ms.PhaseDensity[i]

    # Compute EBSD coordinate system correction matrices
    from .compute_ebsd_correction_matrix import compute_ebsd_correction_matrix
    ref_frame_angle = job.get("ref_frame_angle", 90)
    direction_cosines, bond_matrix, bond_matrix_inv_T, theta_rad = compute_ebsd_correction_matrix(ref_frame_angle)
    ms.EBSDCorrectionMatrix = bond_matrix          # 6x6 Bond matrix MStar (for stiffness rotation)
    ms.EBSDCorrectionMatrixInvT = bond_matrix_inv_T  # 6x6 inv(MStar') NStar (for thermal expansion rotation)
    ms.EBSDDirectionCosines = direction_cosines     # 3x3 rotation matrix (for thermal conductivity)
    ms.EBSDCorrectionAngle = theta_rad              # angle in radians
    if console_on:
        print(f"  EBSD correction matrix computed (ref_frame_angle={ref_frame_angle}°)")

    # Store homogenization method on ms
    ms.ElementLevelHomogenizationMethodValue = job.get("element_homogenization", 4)

    # Compute phase volume fractions from EBSD data point counts
    phases = ms.OriginalDataPhase
    n_pts = len(phases)
    ms.PhaseVolumeFraction = np.zeros(ms.NumberPhases)
    for i in range(ms.NumberPhases):
        ms.PhaseVolumeFraction[i] = np.sum(phases == (i + 1)) / n_pts

    # Compute homogenized (effective) density: weighted average by volume fraction
    ms.HomogenizedDensity = np.sum(ms.PhaseDensity * ms.PhaseVolumeFraction)

    if console_on:
        for i in range(ms.NumberPhases):
            print(f"  Phase {i+1} volume fraction: {ms.PhaseVolumeFraction[i]:.4f}")
        print(f"  Homogenized density: {ms.HomogenizedDensity:.1f} kg/m³")

    # Write merged phase statistics table to log
    if log_path is not None and vl in ("medium", "high"):
        with open(log_path, "a") as f:
            f.write("### Phase Statistics\n\n")
            f.write(f"| {'Phase':>5s} | {'Name':<20s} | {'Points':>6s} | {'Vol. Fraction':>13s} | {'Density (kg/m³)':>16s} |\n")
            f.write(f"|{'-'*7}|{'-'*22}|{'-'*8}|{'-'*15}|{'-'*18}|\n")
            for i in range(ms.NumberPhases):
                ph = i + 1
                name = ms.PhaseName[i] if ms.PhaseName[i] else f"Phase {ph}"
                n_ph = int(np.sum(ms.OriginalDataPhase == ph))
                vf = f"{ms.PhaseVolumeFraction[i]:.4f}"
                rho = f"{ms.PhaseDensity[i]:.1f}"
                f.write(f"| {ph:>5d} | {name:<20s} | {n_ph:>6d} | {vf:>13s} | {rho:>16s} |\n")
            f.write(f"\n**Homogenized density:** {ms.HomogenizedDensity:.1f} kg/m³\n\n")
            f.write("---\n\n")

    # Generate Stage 1 visualization figures and store grain coloring on ms
    if run_dir is not None:
        from .plot_ebsd import plot_ebsd
        grain_color, grain_color_idx, N_CYCLE, fig_count = plot_ebsd(
            ms, run_dir, ebsd_file=ebsd_file, log_path=log_path,
            settings=settings)

        # Store grain coloring data on ms for use by later stages
        ms.grain_color = grain_color
        ms.grain_color_idx = grain_color_idx
        ms.N_CYCLE = N_CYCLE
        ms.fig_count = fig_count

        # Copy input files to input_data for future reference
        import shutil
        input_dir = os.path.join(run_dir, "input_data")
        os.makedirs(input_dir, exist_ok=True)
        # Copy EBSD data file
        if os.path.isfile(ebsd_file):
            shutil.copy2(ebsd_file, input_dir)
            if vc in ("medium", "high"):
                print(f"  Copied: {os.path.basename(ebsd_file)} → input_data/")
        # Copy phase property files
        for ph_num, ph_file in sorted(phase_properties.items()):
            if os.path.isfile(ph_file):
                shutil.copy2(ph_file, input_dir)
                if vc in ("medium", "high"):
                    print(f"  Copied: {os.path.basename(ph_file)} → input_data/")

    return ms
