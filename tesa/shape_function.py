"""
Shape functions for 6-node isoparametric triangular elements.

Evaluates the quadratic shape function value at master element
coordinates (r, s) for a given node number (1-6).

Copyright 2014-2015, Alden C. Cook.
"""


def shape_function(num, r, s):
    """
    Evaluate a quadratic shape function at master element coordinates (r, s).

    Returns the value of the specified nodal shape function for a 6-node
    isoparametric triangular element.

    Parameters
    ----------
    num : int
        Nodal shape function number (1 through 6). Nodes 1-3 are corner
        nodes; nodes 4-6 are mid-side nodes.
    r : float
        r-coordinate in the master element.
    s : float
        s-coordinate in the master element.

    Returns
    -------
    NVal : float
        Value of the shape function at (r, s).
    """
    # Return the nodal shape function value for the given node number
    if num == 1:
        NVal = (1.0 - r - s) * (1.0 - 2.0 * r - 2.0 * s)

    elif num == 2:
        NVal = r * (2.0 * r - 1.0)

    elif num == 3:
        NVal = s * (2.0 * s - 1.0)

    elif num == 4:
        NVal = 4.0 * r * (1.0 - r - s)

    elif num == 5:
        NVal = 4.0 * r * s

    elif num == 6:
        NVal = 4.0 * s * (1.0 - r - s)

    else:
        raise ValueError(f"Unknown shape function number: {num}")

    return NVal
