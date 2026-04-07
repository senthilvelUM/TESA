"""
Assign rotated thermal conductivity at quadrature points.

For each element, rotates the phase thermal conductivity (kappa) by the
Euler angles, then assigns at each quadrature point using the selected
element-level homogenization method.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from . import fem_definitions as FEMDef
from .compute_data_point_adjusted_thermal_conductivity_matrix import (
    compute_data_point_adjusted_thermal_conductivity_matrix
)
from .quadrature_point_coordinates import quadrature_point_coordinates
from .knnsearch2 import knnsearch2
from .assign_quadrature_point_properties import _find_data_points_in_elements


def assign_quadrature_point_thermal_conductivity(
        node_coordinates,
        element_indices,
        data_point_coordinates,
        data_point_euler_angles,
        data_point_phase,
        phase_thermal_conductivity_matrix,
        data_coordinate_system_correction_angle,
        element_homogenization_method):
    """
    Assign rotated thermal conductivity at each quadrature point.

    Rotates per-phase thermal conductivity tensors to the crystal orientation
    at each EBSD data point, then maps them onto quadrature points using the
    selected element-level homogenization method.

    Parameters
    ----------
    node_coordinates : ndarray, shape (n_nodes, 2)
        6-node mesh node coordinates.
    element_indices : ndarray, shape (n_elements, 6), dtype int
        6-node element connectivity (1-based node IDs).
    data_point_coordinates : ndarray, shape (n_data_points, 2)
        Coordinates of EBSD data points.
    data_point_euler_angles : ndarray, shape (n_data_points, 3)
        Euler angles (phi1, Phi, phi2) in radians at each data point.
    data_point_phase : ndarray, shape (n_data_points,), dtype int
        Phase ID for each data point (1-based).
    phase_thermal_conductivity_matrix : list of ndarray
        Thermal conductivity for each phase; each entry has shape (3, 3).
    data_coordinate_system_correction_angle : float
        EBSD correction angle in radians.
    element_homogenization_method : str
        One of ``'Nearest Neighbor'``, ``'Voigt'``, ``'Reuss'``, ``'Hill'``,
        or ``'Geometric Mean'``.

    Returns
    -------
    kappaQuad : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (3, 3, n_elements) holding the rotated thermal conductivity at
        that quadrature point.
    """
    # Local declarations
    nElements = element_indices.shape[0]
    nQuadraturePoints = FEMDef.N_QUADRATURE_POINTS
    nDataPoints = data_point_coordinates.shape[0]

    # Initialize output
    kappaQuad = [np.zeros((3, 3, nElements)) for _ in range(nQuadraturePoints)]

    # Compute rotated thermal conductivity at each data point
    kappaData = compute_data_point_adjusted_thermal_conductivity_matrix(
        data_point_euler_angles, data_point_phase,
        phase_thermal_conductivity_matrix,
        data_coordinate_system_correction_angle)

    # Compute inverse of kappa at each data point (for Reuss)
    invkappaData = np.zeros_like(kappaData)
    for iPoint in range(kappaData.shape[2]):
        invkappaData[:, :, iPoint] = np.linalg.solve(
            kappaData[:, :, iPoint], np.eye(3))

    # Find data points in elements if not Nearest Neighbor
    if element_homogenization_method != 'Nearest Neighbor':
        dxy = data_point_coordinates
        elem_3node = element_indices[:, :3].copy()
        elem_3node_0 = elem_3node - 1  # 0-based
        max_node = int(np.max(elem_3node))
        node_coords_corner = node_coordinates[:max_node, :]

        # Find which element each data point belongs to
        data_point_element = _find_data_points_in_elements(
            dxy, node_coords_corner, elem_3node_0)

        # Build list of data points per element
        dataPointsInElements = [[] for _ in range(nElements)]
        for idp in range(nDataPoints):
            ie = data_point_element[idp]
            if ie >= 0:
                dataPointsInElements[ie].append(idp)

        # For elements with no data points, assign nearest
        noDataPoint = [ie for ie in range(nElements)
                       if len(dataPointsInElements[ie]) == 0]
        if len(noDataPoint) > 0:
            elementCentroid = np.zeros((len(noDataPoint), 2))
            for idx, ie in enumerate(noDataPoint):
                n0, n1, n2 = elem_3node_0[ie]
                elementCentroid[idx, 0] = (node_coords_corner[n0, 0] +
                                            node_coords_corner[n1, 0] +
                                            node_coords_corner[n2, 0]) / 3.0
                elementCentroid[idx, 1] = (node_coords_corner[n0, 1] +
                                            node_coords_corner[n1, 1] +
                                            node_coords_corner[n2, 1]) / 3.0
            iClosest, _ = knnsearch2(elementCentroid, dxy)
            for idx, ie in enumerate(noDataPoint):
                dataPointsInElements[ie] = [int(iClosest[idx, 0])]

    # Assign based on homogenization method
    if element_homogenization_method == 'Nearest Neighbor':
        # Convert to 0-based for quadrature_point_coordinates
        quadXY = quadrature_point_coordinates(node_coordinates, element_indices - 1)
        for iQP in range(nQuadraturePoints):
            closestID, _ = knnsearch2(quadXY[iQP], data_point_coordinates)
            closestID = closestID.ravel()
            kappaQuad[iQP] = kappaData[:, :, closestID]

    elif element_homogenization_method == 'Voigt':
        for iElement in range(nElements):
            dp = dataPointsInElements[iElement]
            nDP = len(dp)
            if nDP == 0:
                continue
            # Voigt: arithmetic mean of kappa
            kappaGQuad_e = np.sum(kappaData[:, :, dp], axis=2) / nDP
            kappaGQuad_e = 0.5 * (kappaGQuad_e + kappaGQuad_e.T)
            for iQP in range(nQuadraturePoints):
                kappaQuad[iQP][:, :, iElement] = kappaGQuad_e

    elif element_homogenization_method == 'Reuss':
        for iElement in range(nElements):
            dp = dataPointsInElements[iElement]
            nDP = len(dp)
            if nDP == 0:
                continue
            # Reuss: arithmetic mean of inv(kappa), then invert
            invkappaGQuad_e = np.linalg.solve(
                np.sum(invkappaData[:, :, dp], axis=2) / nDP, np.eye(3))
            invkappaGQuad_e = 0.5 * (invkappaGQuad_e + invkappaGQuad_e.T)
            for iQP in range(nQuadraturePoints):
                kappaQuad[iQP][:, :, iElement] = invkappaGQuad_e

    elif element_homogenization_method == 'Hill':
        for iElement in range(nElements):
            dp = dataPointsInElements[iElement]
            nDP = len(dp)
            if nDP == 0:
                continue
            # Voigt average
            kappaV = np.sum(kappaData[:, :, dp], axis=2) / nDP
            # Reuss average
            kappaR = np.linalg.solve(
                np.sum(invkappaData[:, :, dp], axis=2) / nDP, np.eye(3))
            # Hill = 0.5 * (Voigt + Reuss)
            kappaH = 0.5 * (kappaV + kappaR)
            for iQP in range(nQuadraturePoints):
                kappaQuad[iQP][:, :, iElement] = kappaH

    elif element_homogenization_method == 'Geometric Mean':
        # Transform kappaData to log space
        for iDataPoint in range(nDataPoints):
            eigvals, V = np.linalg.eigh(kappaData[:, :, iDataPoint])
            log_eigvals = np.log(np.maximum(eigvals, 1e-30))
            kappaData[:, :, iDataPoint] = V @ np.diag(log_eigvals) @ V.T
            kappaData[:, :, iDataPoint] = 0.5 * (
                kappaData[:, :, iDataPoint] + kappaData[:, :, iDataPoint].T)

        # Average in log space per element, then exponentiate
        for iElement in range(nElements):
            dp = dataPointsInElements[iElement]
            nDP = len(dp)
            if nDP == 0:
                continue
            A = np.sum(kappaData[:, :, dp], axis=2) / nDP
            eigvals, V = np.linalg.eigh(A)
            kappaGQuad_e = V @ np.diag(np.exp(eigvals)) @ V.T
            kappaGQuad_e = 0.5 * (kappaGQuad_e + kappaGQuad_e.T)
            for iQP in range(nQuadraturePoints):
                kappaQuad[iQP][:, :, iElement] = kappaGQuad_e

    else:
        raise ValueError(f"Unknown element homogenization method: "
                         f"'{element_homogenization_method}'")

    return kappaQuad
