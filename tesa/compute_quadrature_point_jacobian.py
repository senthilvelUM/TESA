"""
Compute the Jacobian at each quadrature point in each element.

The Jacobian is computed from the coordinate derivatives of the
isoparametric mapping:
    J = dxdr * dyds - dydr * dxds

Copyright 2014-2015, Alden C. Cook.
"""

from .compute_coordinate_derivatives import compute_coordinate_derivatives


def compute_quadrature_point_jacobian(node_coordinates, element_indices):
    """
    Compute the Jacobian determinant at each quadrature point in each element.

    The Jacobian of the isoparametric mapping is computed as
    ``J = dx/dr * dy/ds - dy/dr * dx/ds``.

    Parameters
    ----------
    node_coordinates : ndarray, shape (n_nodes, 2)
        Global (x, y) coordinates of all nodes.
    element_indices : ndarray, shape (n_elements, 6)
        Node indices for each element (0-based).

    Returns
    -------
    quadrature_point_jacobian : ndarray, shape (n_elements, n_qp)
        Jacobian determinant at each quadrature point.
    coordinate_derivatives : dict
        Coordinate derivatives with keys ``'dxds'``, ``'dxdr'``, ``'dyds'``,
        ``'dydr'``, returned for reuse by downstream functions.
    """
    # Compute the derivatives of the global coordinates with respect to the
    # master element coordinates
    coordinate_derivatives = compute_coordinate_derivatives(
        node_coordinates, element_indices)

    # Compute the Jacobian at each quadrature point
    # J = dxdr * dyds - dydr * dxds
    quadrature_point_jacobian = (coordinate_derivatives['dxdr'] *
                                  coordinate_derivatives['dyds'] -
                                  coordinate_derivatives['dydr'] *
                                  coordinate_derivatives['dxds'])

    return quadrature_point_jacobian, coordinate_derivatives
