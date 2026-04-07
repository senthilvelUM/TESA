import numpy as np
from .inpoly import inpoly


def find_phase_polylines(ms):
    """
    Find polylines that separate different phases.

    Identifies grain boundary segments shared by grains of different phases
    and stores them as ms.PhasePolylines.

    Parameters
    ----------
    ms : Microstructure
        Microstructure object. Must have GrainJunctionPoints, GrainsMeshed,
        GrainPhases, and HoleGrains set. Modified in-place: PhasePolylines
        is set.

    Returns
    -------
    None
        Results are stored in ms.PhasePolylines (list of ndarray).
    """
    jpts = ms.GrainJunctionPoints
    Gm = [g.copy() for g in ms.GrainsMeshed]  # deep copy the list of arrays
    gph = np.round(ms.GrainPhases).astype(int)

    n_grains = len(Gm)
    # Gpl stored as dict of (i, j) -> array, to mimic MATLAB cell(numel(Gm), 100)
    Gpl = {}
    Ppl = []
    pc = 0
    mnpl = 0

    for i in range(n_grains):
        print(f'Sorting grain polylines {i + 1} {n_grains}')
        # Check if any points of Gm[i] are junction points
        has_jpt = False
        for row in Gm[i]:
            if _row_in_array(row, jpts):
                has_jpt = True
                break

        if not has_jpt:
            Gpl[(i, 0)] = Gm[i].copy()
            continue

        # Close the polygon temporarily
        Gm_closed = np.vstack([Gm[i], Gm[i][0:1, :]])

        # Find indices where points are junction points
        jd = []
        for k in range(len(Gm_closed)):
            if _row_in_array(Gm_closed[k], jpts):
                jd.append(k)

        npl = 0
        for j in range(len(jd) - 1):
            seg = Gm_closed[jd[j]:jd[j + 1] + 1, :]
            # unique rows, stable
            _, uidx = np.unique(seg, axis=0, return_index=True)
            seg = seg[np.sort(uidx)]
            Gpl[(i, j)] = seg
            npl += 1

        # Gm[i] stays without the appended closing point (already a copy)
        mnpl = max(mnpl, npl)

    # Find phase polylines
    if ms.HoleGrains is None or len(ms.HoleGrains) == 0:
        holes = []
        for i in range(n_grains):
            print(f'Finding grain holes {i + 1} {n_grains}')
            for j in range(n_grains):
                if i == j:
                    continue
                result = inpoly(Gm[j], Gm[i])
                if np.all(result):
                    holes.append(j)
    else:
        holes = list(ms.HoleGrains)

    for i in range(n_grains):
        print(f'Finding phase polylines {i + 1} {n_grains}')

        if len(holes) > 0 and i in holes:
            # This grain is a hole - find its parent
            for j in range(n_grains):
                if i == j:
                    continue
                if np.all(inpoly(Gm[i], Gm[j])):
                    if gph[i] != gph[j]:
                        closed = np.vstack([Gm[i], Gm[i][0:1, :]])
                        Ppl.append(closed)
                        pc += 1
                    break
            continue

        found = False
        for j in range(mnpl):
            if (i, j) not in Gpl:
                break
            seg_ij = Gpl[(i, j)]
            if seg_ij is None:
                continue
            # Check if it's a scalar nan marker
            if isinstance(seg_ij, float) and np.isnan(seg_ij):
                continue

            for ii in range(n_grains):
                if i == ii:
                    continue
                for jj in range(mnpl):
                    if (ii, jj) not in Gpl:
                        break
                    seg_ii_jj = Gpl[(ii, jj)]
                    if seg_ii_jj is None:
                        continue
                    if isinstance(seg_ii_jj, float) and np.isnan(seg_ii_jj):
                        continue

                    if _arrays_equal(seg_ij, np.flipud(seg_ii_jj)):
                        found = True
                        if gph[i] != gph[ii]:
                            Ppl.append(seg_ij.copy())
                            pc += 1
                            Gpl[(i, j)] = np.nan
                            Gpl[(ii, jj)] = np.nan
                            break
                if found:
                    found = False
                    break

    ms.PhasePolylines = Ppl


def _row_in_array(row, arr):
    """
    Check if a 1D row vector exists in a 2D array.

    Parameters
    ----------
    row : ndarray, shape (D,)
        Row vector to search for.
    arr : ndarray, shape (M, D) or None
        Array to search in.

    Returns
    -------
    found : bool
        True if any row of arr matches row exactly.
    """
    if arr is None or len(arr) == 0:
        return False
    return np.any(np.all(arr == row, axis=1))


def _arrays_equal(a, b):
    """
    Check if two 2D arrays are exactly equal (same shape and values).

    Parameters
    ----------
    a : ndarray, shape (M, D)
        First array.
    b : ndarray, shape (M, D)
        Second array.

    Returns
    -------
    equal : bool
        True if shapes match and all elements are equal.
    """
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=0)
