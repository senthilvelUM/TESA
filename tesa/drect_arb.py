import numpy as np


def drect_arb(p, pfix):
    """
    Signed distance of points to an axis-aligned rectangle defined by pfix extents.

    Parameters
    ----------
    p : ndarray, shape (N, 2)
        Query points.
    pfix : ndarray, shape (M, 2)
        Fixed points whose bounding box defines the rectangle.

    Returns
    -------
    d : ndarray, shape (N,)
        Signed distance (negative inside, positive outside).
    """
    x1 = np.min(pfix[:, 0])
    x2 = np.max(pfix[:, 0])
    y1 = np.min(pfix[:, 1])
    y2 = np.max(pfix[:, 1])

    d1 = y1 - p[:, 1]
    d2 = -y2 + p[:, 1]
    d3 = x1 - p[:, 0]
    d4 = -x2 + p[:, 0]

    d5 = np.sqrt(d1**2 + d3**2)
    d6 = np.sqrt(d1**2 + d4**2)
    d7 = np.sqrt(d2**2 + d3**2)
    d8 = np.sqrt(d2**2 + d4**2)

    d = -np.minimum(np.minimum(np.minimum(-d1, -d2), -d3), -d4)

    ix = (d1 > 0) & (d3 > 0)
    d[ix] = d5[ix]
    ix = (d1 > 0) & (d4 > 0)
    d[ix] = d6[ix]
    ix = (d2 > 0) & (d3 > 0)
    d[ix] = d7[ix]
    ix = (d2 > 0) & (d4 > 0)
    d[ix] = d8[ix]

    return d
