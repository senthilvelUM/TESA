"""
Compute seismic wave speeds from an elastic stiffness matrix and density.

Computes the Christoffel matrix for each propagation direction (phi, gamma),
finds eigenvalues to get wave speeds, then classifies into P, SH, and SV
waves based on polarization direction relative to the propagation direction
and sagittal plane.

Output velocities are in m/s when C is in Pa and rho is in kg/m^3.

"""

import numpy as np
from .contract import contract


def wavespeeds(C, rho, phi, gamma):
    """
    Compute seismic wave speeds for given propagation directions.

    Parameters
    ----------
    C : (6, 6) array
        Elastic stiffness matrix (Voigt notation, Pa).
    rho : float
        Density (kg/m^3).
    phi : (nDirections,) array
        Azimuth angles (radians).
    gamma : (nDirections,) array
        Inclination angles (radians).

    Returns
    -------
    v : (1, 3, nDirections) array
        Raw eigenvalue-based wave speeds (m/s).
    ev : (3, 3, nDirections) array
        Eigenvectors (polarization directions) at each direction.
    V : (nDirections, 5) array
        Classified wave speeds: [VP, VS1, VS2, VSH, VSV] (m/s).
    """
    phi = np.asarray(phi, dtype=float).ravel()
    gamma = np.asarray(gamma, dtype=float).ravel()
    nDir = len(phi)

    # Calculate the normal vector from the azimuth and inclination angles
    # n = [sin(gamma)*cos(phi), sin(gamma)*sin(phi), cos(gamma)]
    n = np.zeros((1, 3, nDir))
    n[0, 0, :] = np.sin(gamma) * np.cos(phi)
    n[0, 1, :] = np.sin(gamma) * np.sin(phi)
    n[0, 2, :] = np.cos(gamma)

    # Calculate the normal to the sagittal plane from the azimuth angle
    # s = [-sin(phi), cos(phi), 0]
    s = np.zeros((1, 3, nDir))
    s[0, 0, :] = -np.sin(phi)
    s[0, 1, :] = np.cos(phi)
    s[0, 2, :] = 0.0

    # Compute the Christoffel matrix
    # g(i,k) = sum_j sum_l C(contract(i,j), contract(k,l)) * n_j * n_l
    g = np.zeros((3, 3, nDir))
    # Expand C to 3D for broadcasting
    Cm = np.zeros((6, 6, nDir))
    for i in range(6):
        for j in range(6):
            Cm[i, j, :] = C[i, j]

    for i in range(1, 4):        # 1-based for contract()
        for k in range(1, 4):
            for j in range(1, 4):
                for l in range(1, 4):
                    cij = contract(i, j) - 1  # 0-based
                    ckl = contract(k, l) - 1  # 0-based
                    g[i-1, k-1, :] += Cm[cij, ckl, :] * n[0, j-1, :] * n[0, l-1, :]

    # Find eigenvalues and eigenvectors for each direction
    A = np.zeros((3, 3, nDir))
    ev = np.zeros((3, 3, nDir))
    for i in range(nDir):
        eigvals, eigvecs = np.linalg.eig(g[:, :, i])
        A[:, :, i] = np.diag(eigvals)
        ev[:, :, i] = eigvecs

    # Determine wave speeds (m/s): v = sqrt(eigenvalue / rho)
    # v: (1, 3, nDir)
    v = np.zeros((1, 3, nDir))
    v[0, 0, :] = A[0, 0, :]
    v[0, 1, :] = A[1, 1, :]
    v[0, 2, :] = A[2, 2, :]
    v = np.sqrt(np.maximum(v / rho, 0.0))

    # Identify SH velocity: displacement predominantly in s-direction
    as_max = np.zeros((1, 1, nDir))
    VSH = np.zeros(nDir)
    SHIndex = np.zeros(nDir, dtype=int)
    for i in range(3):
        # Polarization vector for eigenmode i
        a = ev[:, i, :]  # (3, nDir)
        # Dot product with s-direction: |s . a|
        a_s = np.abs(np.sum(s[0, :, :] * a, axis=0))  # (nDir,)
        # Update where this mode has larger s-component
        k = a_s > as_max[0, 0, :]
        as_max[0, 0, k] = a_s[k]
        VSH[k] = v[0, i, k]
        SHIndex[k] = i

    # Identify P velocity: largest component in n-direction
    an_max = np.zeros((1, 1, nDir))
    VP = np.zeros(nDir)
    PIndex = np.zeros(nDir, dtype=int)
    for i in range(3):
        a = ev[:, i, :]  # (3, nDir)
        a_n = np.abs(np.sum(n[0, :, :] * a, axis=0))  # (nDir,)
        k = a_n > an_max[0, 0, :]
        an_max[0, 0, k] = a_n[k]
        VP[k] = v[0, i, k]
        PIndex[k] = i

    # Identify SV velocity: the remaining mode (not P, not SH)
    VSV = np.zeros(nDir)
    for i in range(3):
        k = (PIndex != i) & (SHIndex != i)
        VSV[k] = v[0, i, k]

    # VS1 is the larger shear wave speed, VS2 is the smaller
    VS1 = np.maximum(VSH, VSV)
    VS2 = np.minimum(VSH, VSV)

    # Array of seismic velocities: [VP, VS1, VS2, VSH, VSV]
    V = np.column_stack([VP, VS1, VS2, VSH, VSV])

    return v, ev, V
