"""
Compute area-weighted average of a field quantity for each phase.

Averages quadrature point field data (e.g., stress, strain) over all
elements belonging to each phase, weighted by element area.

"""

import numpy as np
from .triarea import triarea


def compute_average_phase_field(ms, comp, data):
    """
    Compute area-weighted field average for each phase.

    Parameters
    ----------
    ms : Microstructure
        Must have GrainsMeshed, ThreeNodeCoordinateList,
        ThreeNodeElementIndexList, GrainsElements, GrainPhases,
        NumberPhases populated.
    comp : int
        Field component index (0-based). The data column used is comp+2
        (cols 0,1 are x,y coordinates in the Microfield layout).
    data : (nQuadTotal, nCols) array
        Quadrature point data (e.g., ms.Microfield arranged as columns).

    Returns
    -------
    field_avg : (nPhases,) array
        Area-weighted average of the field for each phase.
    """
    # Load grain and element info
    Gm = ms.GrainsMeshed
    p = ms.ThreeNodeCoordinateList
    t = ms.ThreeNodeElementIndexList

    # GrainsElements: dict mapping grain index to list of element indices
    Ge = ms.GrainsElements

    # Load field data at quadrature points
    # Normalize by minimum value, average over 4 quad points per element
    zf_raw = data[:, comp + 2]
    min_val = np.min(zf_raw)
    zf = zf_raw / min_val
    nElements = t.shape[0]
    zf = zf.reshape(4, nElements, order='F')  # (4, nElements) — Fortran (column-major) order
    zf = np.sum(zf, axis=0) / 4.0  # average over 4 quad points per element

    # Compute element areas
    tA = triarea(p, t)

    # Accumulate area-weighted field sums per phase
    nPhases = ms.NumberPhases
    field_sum = np.zeros(nPhases)
    area_sum = np.zeros(nPhases)

    for n in range(len(Gm)):
        if Gm[n] is None:
            continue

        phase_idx = int(ms.GrainPhases[n]) - 1  # 0-based
        if phase_idx < 0 or phase_idx >= nPhases:
            continue

        # Element indices for this grain
        if isinstance(Ge, dict):
            elem_ids = Ge.get(n, Ge.get(n + 1, []))
        else:
            elem_ids = Ge[n] if n < len(Ge) else []

        if len(elem_ids) == 0:
            continue

        elem_ids = np.asarray(elem_ids, dtype=int)

        # Accumulate
        area_sum[phase_idx] += np.sum(tA[elem_ids])
        field_sum[phase_idx] += np.sum(tA[elem_ids] * zf[elem_ids])

    # Compute averages (undo the normalization)
    field_avg = np.zeros(nPhases)
    for i in range(nPhases):
        if area_sum[i] > 0:
            field_avg[i] = field_sum[i] * min_val / area_sum[i]

    return field_avg
