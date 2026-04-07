"""
Create a rotated rectangle defined by its center, dimensions, and angle.

Returns the 5 vertices (closed polygon) of a rectangle centered at
(xc, yc) with width w, height h, rotated by angle phi about the center.

"""

import numpy as np


def create_rectangle(xc, yc, w, h, phi):
    """
    Create vertices of a rotated rectangle.

    Parameters
    ----------
    xc, yc : float
        Center coordinates.
    w : float
        Width of the rectangle.
    h : float
        Height of the rectangle.
    phi : float
        Rotation angle (radians, counterclockwise).

    Returns
    -------
    x, y : (5,) arrays
        Vertex coordinates (closed polygon, first = last).
    """
    # Define vertices of unrotated rectangle
    x = np.array([xc - w / 2, xc - w / 2, xc + w / 2, xc + w / 2, xc - w / 2])
    y = np.array([yc - h / 2, yc + h / 2, yc + h / 2, yc - h / 2, yc - h / 2])

    # Rotation matrix
    RM = np.array([[np.cos(phi), -np.sin(phi)],
                    [np.sin(phi),  np.cos(phi)]])

    # Rotate about center
    rxy = RM @ np.vstack([x - xc, y - yc])
    x = xc + rxy[0, :]
    y = yc + rxy[1, :]

    return x, y
