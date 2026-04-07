"""
Order of magnitude of a number.
order(0.002) returns -3, order(1.3e6) returns 6.
Author: Ivar Smith
"""

import math


def order(val):
    """
    Compute the order of magnitude of a number.

    Parameters
    ----------
    val : float
        Input value (must be nonzero).

    Returns
    -------
    n : int
        Order of magnitude, e.g. order(0.002) = -3, order(1.3e6) = 6.
    """
    return math.floor(math.log10(abs(val)))
