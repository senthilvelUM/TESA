"""
Compute derivatives of global coordinates (x, y) with respect to
master element coordinates (r, s) for 6-node isoparametric triangles.

The shape function derivatives dN/dr and dN/ds are evaluated at each
quadrature point, then combined with element nodal coordinates to
produce dx/ds, dx/dr, dy/ds, dy/dr for every element.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from . import fem_definitions as FEM


def compute_coordinate_derivatives(node_coordinates, element_indices):
    """
    Compute derivatives of global coordinates w.r.t. master element coordinates.

    Evaluates dx/dr, dx/ds, dy/dr, dy/ds at each quadrature point for every
    element using the 6-node isoparametric shape function derivatives.

    Parameters
    ----------
    node_coordinates : ndarray, shape (n_nodes, 2)
        Global (x, y) coordinates of all nodes.
    element_indices : ndarray, shape (n_elements, 6)
        Node indices for each element (0-based).

    Returns
    -------
    coord_derivs : dict
        Dictionary with keys ``'dxds'``, ``'dxdr'``, ``'dyds'``, ``'dydr'``.
        Each value is an ndarray of shape (n_elements, n_quadrature_points).
    """
    # Local declarations
    nqp = FEM.N_QUADRATURE_POINTS
    n_elements = element_indices.shape[0]
    r = FEM.r   # quadrature point coordinates in master element
    s = FEM.s

    # Extract element nodal coordinates (nElements, 6)
    element_x = node_coordinates[element_indices, 0]
    element_y = node_coordinates[element_indices, 1]

    # Initialize output arrays
    dxds = np.zeros((n_elements, nqp))
    dxdr = np.zeros((n_elements, nqp))
    dyds = np.zeros((n_elements, nqp))
    dydr = np.zeros((n_elements, nqp))

    # Shape function derivatives in master coordinate system
    # dNdr: (6, nQuadraturePoints)
    dNdr = np.array([
        4.0 * r + 4.0 * s - 3.0,
        4.0 * r - 1.0,
        np.zeros(nqp),
        -8.0 * r - 4.0 * (s - 1.0),
        4.0 * s,
        -4.0 * s
    ])

    # dNds: (6, nQuadraturePoints)
    dNds = np.array([
        4.0 * s + 4.0 * r - 3.0,
        np.zeros(nqp),
        4.0 * s - 1.0,
        -4.0 * r,
        4.0 * r,
        -8.0 * s - 4.0 * (r - 1.0)
    ])

    # Compute derivatives of global coordinates w.r.t. master element coordinates
    for iqp in range(nqp):
        # dx/ds = sum_i (x_i * dN_i/ds)
        dxds[:, iqp] = (element_x[:, 0] * dNds[0, iqp] +
                         element_x[:, 1] * dNds[1, iqp] +
                         element_x[:, 2] * dNds[2, iqp] +
                         element_x[:, 3] * dNds[3, iqp] +
                         element_x[:, 4] * dNds[4, iqp] +
                         element_x[:, 5] * dNds[5, iqp])

        # dx/dr = sum_i (x_i * dN_i/dr)
        dxdr[:, iqp] = (element_x[:, 0] * dNdr[0, iqp] +
                         element_x[:, 1] * dNdr[1, iqp] +
                         element_x[:, 2] * dNdr[2, iqp] +
                         element_x[:, 3] * dNdr[3, iqp] +
                         element_x[:, 4] * dNdr[4, iqp] +
                         element_x[:, 5] * dNdr[5, iqp])

        # dy/ds = sum_i (y_i * dN_i/ds)
        dyds[:, iqp] = (element_y[:, 0] * dNds[0, iqp] +
                         element_y[:, 1] * dNds[1, iqp] +
                         element_y[:, 2] * dNds[2, iqp] +
                         element_y[:, 3] * dNds[3, iqp] +
                         element_y[:, 4] * dNds[4, iqp] +
                         element_y[:, 5] * dNds[5, iqp])

        # dy/dr = sum_i (y_i * dN_i/dr)
        dydr[:, iqp] = (element_y[:, 0] * dNdr[0, iqp] +
                         element_y[:, 1] * dNdr[1, iqp] +
                         element_y[:, 2] * dNdr[2, iqp] +
                         element_y[:, 3] * dNdr[3, iqp] +
                         element_y[:, 4] * dNdr[4, iqp] +
                         element_y[:, 5] * dNdr[5, iqp])

    return {'dxds': dxds, 'dxdr': dxdr, 'dyds': dyds, 'dydr': dydr}
