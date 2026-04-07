"""
Assign rotated material properties at quadrature points.

For each element, rotates the phase stiffness (C), compliance (S), thermal
expansion (alpha), and stress-temperature moduli (beta) by the Euler angles,
then assigns them at each quadrature point using the selected element-level
homogenization method.

For conforming meshes (Type 1), each element has one data point (one grain),
so all methods give the same result.

For non-conforming meshes (Types 2, 3), elements may contain multiple EBSD
data points from different grains, and the homogenization method determines
how properties are averaged within each element.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np
from . import fem_definitions as FEMDef
from .compute_data_point_adjusted_stiffness_matrix import (
    compute_data_point_adjusted_stiffness_matrix
)
from .compute_data_point_adjusted_compliance_matrix import (
    compute_data_point_adjusted_compliance_matrix
)
from .compute_data_point_adjusted_thermal_properties import (
    compute_data_point_adjusted_thermal_properties
)
from .quadrature_point_coordinates import quadrature_point_coordinates
from .knnsearch2 import knnsearch2
from .shape_function import shape_function


def _find_data_points_in_elements(data_coords, node_coords, element_indices_3node):
    """
    Assign each data point to the triangular element that contains it.

    Uses barycentric coordinate test for point-in-triangle membership.
    Data points are processed in batches for memory efficiency.

    Parameters
    ----------
    data_coords : ndarray, shape (n_data_points, 2)
        Coordinates of data points.
    node_coords : ndarray, shape (n_nodes, 2)
        Coordinates of mesh nodes.
    element_indices_3node : ndarray, shape (n_elements, 3), dtype int
        Corner-node element connectivity (0-based node indices).

    Returns
    -------
    data_point_element : ndarray, shape (n_data_points,), dtype int
        Element index (0-based) for each data point. Set to -1 for data
        points not found inside any element.
    """
    nDataPoints = data_coords.shape[0]
    nElements = element_indices_3node.shape[0]
    data_point_element = -np.ones(nDataPoints, dtype=int)

    # Get triangle vertex coordinates for all elements
    v0 = node_coords[element_indices_3node[:, 0]]  # (nE, 2)
    v1 = node_coords[element_indices_3node[:, 1]]  # (nE, 2)
    v2 = node_coords[element_indices_3node[:, 2]]  # (nE, 2)

    # Precompute for barycentric coordinates
    d00 = np.sum((v1 - v0) * (v1 - v0), axis=1)
    d01 = np.sum((v1 - v0) * (v2 - v0), axis=1)
    d11 = np.sum((v2 - v0) * (v2 - v0), axis=1)
    inv_denom = 1.0 / (d00 * d11 - d01 * d01 + 1e-30)

    # Process data points in batches for memory efficiency
    batch_size = 1000
    tol = 1e-8
    for start in range(0, nDataPoints, batch_size):
        end = min(start + batch_size, nDataPoints)
        pts = data_coords[start:end]  # (batch, 2)

        # Compute barycentric coords for all pts vs all elements
        # v2p = pts - v0  for each (pt, element) pair
        for ip in range(pts.shape[0]):
            p = pts[ip]
            v2p0 = p[0] - v0[:, 0]
            v2p1 = p[1] - v0[:, 1]
            d20 = v2p0 * (v1[:, 0] - v0[:, 0]) + v2p1 * (v1[:, 1] - v0[:, 1])
            d21 = v2p0 * (v2[:, 0] - v0[:, 0]) + v2p1 * (v2[:, 1] - v0[:, 1])
            u = (d11 * d20 - d01 * d21) * inv_denom
            v = (d00 * d21 - d01 * d20) * inv_denom
            # Point is inside if u >= 0, v >= 0, u + v <= 1
            mask = (u >= -tol) & (v >= -tol) & (u + v <= 1.0 + tol)
            candidates = np.where(mask)[0]
            if len(candidates) > 0:
                data_point_element[start + ip] = candidates[0]

    return data_point_element


def assign_quadrature_point_properties(
        node_coordinates,
        element_indices,
        data_point_coordinates,
        data_point_euler_angles,
        data_point_phase,
        phase_elastic_stiffness_matrix,
        phase_thermal_expansion_matrix,
        data_coordinate_system_correction_matrix,
        data_coordinate_system_correction_angle,
        element_homogenization_method):
    """
    Assign rotated elastic and thermal properties at each quadrature point.

    Rotates per-phase stiffness, compliance, thermal expansion, and
    stress-temperature moduli to the crystal orientation at each EBSD data
    point, then maps properties onto quadrature points using the selected
    element-level homogenization method.

    Parameters
    ----------
    node_coordinates : ndarray, shape (n_nodes, 2)
        6-node mesh node coordinates.
    element_indices : ndarray, shape (n_elements, 6), dtype int
        6-node element connectivity (1-based node IDs).
    data_point_coordinates : ndarray, shape (n_data_points, 2)
        Coordinates of EBSD data points or element centroids.
    data_point_euler_angles : ndarray, shape (n_data_points, 3)
        Euler angles (phi1, Phi, phi2) in radians at each data point.
    data_point_phase : ndarray, shape (n_data_points,), dtype int
        Phase ID for each data point (1-based).
    phase_elastic_stiffness_matrix : list of ndarray
        Stiffness matrix for each phase; each entry has shape (6, 6).
    phase_thermal_expansion_matrix : list of ndarray
        Thermal expansion for each phase; each entry has shape (6, 1) or (6,).
    data_coordinate_system_correction_matrix : ndarray, shape (6, 6)
        EBSD coordinate system correction (Bond) matrix MStar.
    data_coordinate_system_correction_angle : float
        EBSD correction angle in radians (unused here; kept for API
        consistency with thermal conductivity assignment).
    element_homogenization_method : str
        One of ``'Nearest Neighbor'``, ``'Voigt'``, ``'Reuss'``, ``'Hill'``,
        or ``'Geometric Mean'``.

    Returns
    -------
    DQuad : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (6, 6, n_elements) holding the rotated stiffness at that
        quadrature point.
    betaQuad : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (6, 1, n_elements) holding the stress-temperature moduli at
        that quadrature point.
    alphaQuad : list of ndarray
        List of length ``N_QUADRATURE_POINTS``. Each entry is an ndarray of
        shape (6, 1, n_elements) holding the thermal expansion coefficients
        at that quadrature point.
    """
    # Local declarations
    nElements = element_indices.shape[0]
    nQuadraturePoints = FEMDef.N_QUADRATURE_POINTS
    nDataPoints = data_point_coordinates.shape[0]

    # Initialize output — list of arrays, one per quadrature point
    DQuad = [np.zeros((6, 6, nElements)) for _ in range(nQuadraturePoints)]
    betaQuad = [np.zeros((6, 1, nElements)) for _ in range(nQuadraturePoints)]
    alphaQuad = [np.zeros((6, 1, nElements)) for _ in range(nQuadraturePoints)]

    # Compute the adjusted properties at each data point
    DData, dataPointBondMatrix = compute_data_point_adjusted_stiffness_matrix(
        data_point_euler_angles, data_point_phase,
        phase_elastic_stiffness_matrix,
        data_coordinate_system_correction_matrix)

    SData = compute_data_point_adjusted_compliance_matrix(
        data_point_phase, phase_elastic_stiffness_matrix,
        data_coordinate_system_correction_matrix,
        dataPointBondMatrix)

    betaData, alphaData = compute_data_point_adjusted_thermal_properties(
        data_point_euler_angles, data_point_phase,
        phase_elastic_stiffness_matrix, phase_thermal_expansion_matrix,
        data_coordinate_system_correction_matrix)

    # Find data points in elements if not Nearest Neighbor
    if element_homogenization_method != 'Nearest Neighbor':
        dxy = data_point_coordinates
        # Use corner nodes only (first 3 columns of 6-node connectivity)
        elem_3node = element_indices[:, :3].copy()
        # Convert to 0-based for indexing
        elem_3node_0 = elem_3node - 1
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

        # For elements with no data points, assign nearest data point
        noDataPoint = [ie for ie in range(nElements) if len(dataPointsInElements[ie]) == 0]
        if len(noDataPoint) > 0:
            # Compute centroids of empty elements
            elementCentroid = np.zeros((len(noDataPoint), 2))
            for idx, ie in enumerate(noDataPoint):
                n0, n1, n2 = elem_3node_0[ie]
                elementCentroid[idx, 0] = (node_coords_corner[n0, 0] +
                                            node_coords_corner[n1, 0] +
                                            node_coords_corner[n2, 0]) / 3.0
                elementCentroid[idx, 1] = (node_coords_corner[n0, 1] +
                                            node_coords_corner[n1, 1] +
                                            node_coords_corner[n2, 1]) / 3.0
            # Find closest data point to each empty element's centroid
            iClosest, _ = knnsearch2(elementCentroid, dxy)
            for idx, ie in enumerate(noDataPoint):
                dataPointsInElements[ie] = [int(iClosest[idx, 0])]

    # Assign properties based on homogenization method
    if element_homogenization_method == 'Nearest Neighbor':
        # For each quadrature point, find nearest data point
        # Convert to 0-based for quadrature_point_coordinates
        quadXY = quadrature_point_coordinates(node_coordinates, element_indices - 1)
        for iQP in range(nQuadraturePoints):
            closestID, _ = knnsearch2(quadXY[iQP], data_point_coordinates)
            closestID = closestID.ravel()
            DQuad[iQP] = DData[:, :, closestID]
            betaQuad[iQP] = betaData[:, :, closestID]
            alphaQuad[iQP] = alphaData[:, :, closestID]

    elif element_homogenization_method == 'Voigt':
        for iElement in range(nElements):
            dp = dataPointsInElements[iElement]
            nDP = len(dp)
            if nDP == 0:
                continue
            # Voigt average: arithmetic mean of stiffness
            DGQuad_e = np.sum(DData[:, :, dp], axis=2) / nDP
            DGQuad_e = 0.5 * (DGQuad_e + DGQuad_e.T)  # symmetrize
            betaGQuad_e = np.sum(betaData[:, :, dp], axis=2) / nDP
            # alpha = S @ beta where S = inv(C_Voigt)
            alphaGQuad_e = np.linalg.solve(DGQuad_e, np.eye(6)) @ betaGQuad_e
            for iQP in range(nQuadraturePoints):
                DQuad[iQP][:, :, iElement] = DGQuad_e
                betaQuad[iQP][:, :, iElement] = betaGQuad_e
                alphaQuad[iQP][:, :, iElement] = alphaGQuad_e

    elif element_homogenization_method == 'Reuss':
        for iElement in range(nElements):
            dp = dataPointsInElements[iElement]
            nDP = len(dp)
            if nDP == 0:
                continue
            # Reuss average: arithmetic mean of compliance, then invert
            SGQuad_e = np.linalg.solve(
                np.sum(SData[:, :, dp], axis=2) / nDP, np.eye(6))
            SGQuad_e = 0.5 * (SGQuad_e + SGQuad_e.T)  # symmetrize
            alphaGQuad_e = np.sum(alphaData[:, :, dp], axis=2) / nDP
            betaGQuad_e = SGQuad_e @ alphaGQuad_e
            for iQP in range(nQuadraturePoints):
                DQuad[iQP][:, :, iElement] = SGQuad_e
                betaQuad[iQP][:, :, iElement] = betaGQuad_e
                alphaQuad[iQP][:, :, iElement] = alphaGQuad_e

    elif element_homogenization_method == 'Hill':
        for iElement in range(nElements):
            dp = dataPointsInElements[iElement]
            nDP = len(dp)
            if nDP == 0:
                continue
            # Voigt average of stiffness
            DGQuad_e = np.sum(DData[:, :, dp], axis=2) / nDP
            # Reuss average of compliance, then invert
            SGQuad_e = np.linalg.solve(
                np.sum(SData[:, :, dp], axis=2) / nDP, np.eye(6))
            # Hill = 0.5 * (Voigt + Reuss)
            DHill = 0.5 * (DGQuad_e + SGQuad_e)

            # Voigt beta and alpha
            betaGQuad_e = np.sum(betaData[:, :, dp], axis=2) / nDP
            betaQuadV = betaGQuad_e
            alphaQuadV = np.linalg.solve(DGQuad_e, np.eye(6)) @ betaGQuad_e

            # Reuss alpha and beta
            alphaGQuad_e = np.sum(alphaData[:, :, dp], axis=2) / nDP
            alphaQuadR = alphaGQuad_e
            betaQuadR = SGQuad_e @ alphaGQuad_e

            # Hill averages
            betaHill = 0.5 * (betaQuadV + betaQuadR)
            alphaHill = 0.5 * (alphaQuadV + alphaQuadR)

            for iQP in range(nQuadraturePoints):
                DQuad[iQP][:, :, iElement] = DHill
                betaQuad[iQP][:, :, iElement] = betaHill
                alphaQuad[iQP][:, :, iElement] = alphaHill

    elif element_homogenization_method == 'Geometric Mean':
        # Transform DData, betaData, alphaData to log space per data point
        idMatrix = np.eye(6)
        for iDataPoint in range(nDataPoints):
            # Matrix logarithm of DData
            V, _ = np.linalg.eig(DData[:, :, iDataPoint])
            V_mat = np.linalg.eig(DData[:, :, iDataPoint])[1]  # eigenvectors
            eigvals = np.linalg.eigvalsh(DData[:, :, iDataPoint])
            # Use scipy for proper matrix log
            D_dp = DData[:, :, iDataPoint]
            eigvals_d, V_d = np.linalg.eigh(D_dp)
            log_eigvals = np.log(np.maximum(eigvals_d, 1e-30))
            DData[:, :, iDataPoint] = V_d @ np.diag(log_eigvals) @ V_d.T
            DData[:, :, iDataPoint] = 0.5 * (DData[:, :, iDataPoint] +
                                               DData[:, :, iDataPoint].T)

            # Matrix logarithm of betaMat (3x3 tensor form)
            b = betaData[:, 0, iDataPoint]
            betaMat = np.array([
                [b[0], b[5], b[4]],
                [b[5], b[1], b[3]],
                [b[4], b[3], b[2]]
            ])
            eigvals_b, V_b = np.linalg.eigh(betaMat)
            log_eigvals_b = np.log(np.maximum(np.abs(eigvals_b), 1e-30))
            log_eigvals_b = np.sign(eigvals_b) * log_eigvals_b
            betaMat = V_b @ np.diag(log_eigvals_b) @ V_b.T
            betaMat = 0.5 * (betaMat + betaMat.T)
            betaData[0, 0, iDataPoint] = betaMat[0, 0]
            betaData[1, 0, iDataPoint] = betaMat[1, 1]
            betaData[2, 0, iDataPoint] = betaMat[2, 2]
            betaData[3, 0, iDataPoint] = betaMat[1, 2]
            betaData[4, 0, iDataPoint] = betaMat[0, 2]
            betaData[5, 0, iDataPoint] = betaMat[0, 1]

            # Matrix logarithm of alphaMat (3x3 tensor form, engineering shear)
            al = alphaData[:, 0, iDataPoint]
            alphaMat = np.array([
                [al[0],     al[5] / 2, al[4] / 2],
                [al[5] / 2, al[1],     al[3] / 2],
                [al[4] / 2, al[3] / 2, al[2]]
            ])
            eigvals_a, V_a = np.linalg.eigh(alphaMat)
            log_eigvals_a = np.log(np.maximum(np.abs(eigvals_a), 1e-30))
            log_eigvals_a = np.sign(eigvals_a) * log_eigvals_a
            alphaMat = V_a @ np.diag(log_eigvals_a) @ V_a.T
            alphaMat = 0.5 * (alphaMat + alphaMat.T)
            alphaData[0, 0, iDataPoint] = alphaMat[0, 0]
            alphaData[1, 0, iDataPoint] = alphaMat[1, 1]
            alphaData[2, 0, iDataPoint] = alphaMat[2, 2]
            alphaData[3, 0, iDataPoint] = 2 * alphaMat[1, 2]
            alphaData[4, 0, iDataPoint] = 2 * alphaMat[0, 2]
            alphaData[5, 0, iDataPoint] = 2 * alphaMat[0, 1]

        # Average in log space per element, then exponentiate
        for iElement in range(nElements):
            dp = dataPointsInElements[iElement]
            nDP = len(dp)
            if nDP == 0:
                continue

            # Matrix exponential of averaged log-stiffness
            A = np.sum(DData[:, :, dp], axis=2) / nDP
            eigvals_d, V_d = np.linalg.eigh(A)
            exp_eigvals = np.exp(eigvals_d)
            DGQuad_e = V_d @ np.diag(exp_eigvals) @ V_d.T
            DGQuad_e = 0.5 * (DGQuad_e + DGQuad_e.T)

            # Matrix exponential of averaged log-beta
            b_sliced = betaData[:, 0, dp]  # (6, nDP)
            betaMat_avg = np.zeros((3, 3))
            betaMat_avg[0, 0] = np.mean(b_sliced[0, :])
            betaMat_avg[1, 1] = np.mean(b_sliced[1, :])
            betaMat_avg[2, 2] = np.mean(b_sliced[2, :])
            betaMat_avg[1, 2] = np.mean(b_sliced[3, :])
            betaMat_avg[0, 2] = np.mean(b_sliced[4, :])
            betaMat_avg[0, 1] = np.mean(b_sliced[5, :])
            betaMat_avg[2, 1] = betaMat_avg[1, 2]
            betaMat_avg[2, 0] = betaMat_avg[0, 2]
            betaMat_avg[1, 0] = betaMat_avg[0, 1]
            eigvals_b, V_b = np.linalg.eigh(betaMat_avg)
            betaMat_exp = V_b @ np.diag(np.exp(eigvals_b)) @ V_b.T
            betaMat_exp = 0.5 * (betaMat_exp + betaMat_exp.T)
            betaGQuad_e = np.array([
                [betaMat_exp[0, 0]], [betaMat_exp[1, 1]], [betaMat_exp[2, 2]],
                [betaMat_exp[1, 2]], [betaMat_exp[0, 2]], [betaMat_exp[0, 1]]
            ])

            # Matrix exponential of averaged log-alpha
            al_sliced = alphaData[:, 0, dp]  # (6, nDP)
            alphaMat_avg = np.zeros((3, 3))
            alphaMat_avg[0, 0] = np.mean(al_sliced[0, :])
            alphaMat_avg[1, 1] = np.mean(al_sliced[1, :])
            alphaMat_avg[2, 2] = np.mean(al_sliced[2, :])
            alphaMat_avg[1, 2] = np.mean(al_sliced[3, :]) / 2
            alphaMat_avg[0, 2] = np.mean(al_sliced[4, :]) / 2
            alphaMat_avg[0, 1] = np.mean(al_sliced[5, :]) / 2
            alphaMat_avg[2, 1] = alphaMat_avg[1, 2]
            alphaMat_avg[2, 0] = alphaMat_avg[0, 2]
            alphaMat_avg[1, 0] = alphaMat_avg[0, 1]
            eigvals_a, V_a = np.linalg.eigh(alphaMat_avg)
            alphaMat_exp = V_a @ np.diag(np.exp(eigvals_a)) @ V_a.T
            alphaMat_exp = 0.5 * (alphaMat_exp + alphaMat_exp.T)
            alphaGQuad_e = np.array([
                [alphaMat_exp[0, 0]], [alphaMat_exp[1, 1]], [alphaMat_exp[2, 2]],
                [2 * alphaMat_exp[1, 2]], [2 * alphaMat_exp[0, 2]], [2 * alphaMat_exp[0, 1]]
            ])

            for iQP in range(nQuadraturePoints):
                DQuad[iQP][:, :, iElement] = DGQuad_e
                betaQuad[iQP][:, :, iElement] = betaGQuad_e
                alphaQuad[iQP][:, :, iElement] = alphaGQuad_e

    else:
        raise ValueError(f"Unknown element homogenization method: "
                         f"'{element_homogenization_method}'")

    return DQuad, betaQuad, alphaQuad
