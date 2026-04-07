"""
Compute the EBSD coordinate system correction matrices.

Builds a 3x3 direction cosines matrix and a 6x6 Bond/Voigt transformation
matrix from a rotation angle (degrees) about the z-axis. The 6x6 Bond matrix
rotates stiffness tensors (in Voigt notation) and the 3x3 matrix rotates
thermal conductivity tensors from the EBSD reference frame to the FE mesh
coordinate system.

"""

import numpy as np


def compute_ebsd_correction_matrix(theta_deg):
    """
    Compute the EBSD correction matrices for a z-axis rotation.

    Parameters
    ----------
    theta_deg : float
        Rotation angle in degrees (typically 90 for standard EBSD data).

    Returns
    -------
    direction_cosines : (3, 3) ndarray
        Direction cosines matrix (z-axis rotation). Used for rotating
        3x3 thermal conductivity tensors: kappa_rot = a @ kappa @ a.T
    bond_matrix : (6, 6) ndarray
        Bond/Voigt transformation matrix (MStar). Used for rotating
        6x6 stiffness tensors: C_rot = MStar @ M @ C @ M.T @ MStar.T
    bond_matrix_inv_T : (6, 6) ndarray
        Transpose-inverse of bond_matrix (NStar = inv(MStar.T)). Used for
        rotating 6x1 thermal expansion: alpha_rot = NStar @ N @ alpha
    theta_rad : float
        The angle in radians (stored for later use).
    """
    # Convert angle to radians
    theta_rad = np.radians(theta_deg)

    # 3x3 rotation matrix about z-axis
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    a = np.array([
        [ c, -s,  0],
        [ s,  c,  0],
        [ 0,  0,  1]
    ])

    # 6x6 Bond/Voigt transformation matrix
    # Transforms a symmetric 2nd-order tensor in Voigt notation
    # under the rotation defined by a
    corr_mat = np.array([
        [a[0,0]**2,         a[0,1]**2,         a[0,2]**2,
         2*a[0,1]*a[0,2],   2*a[0,2]*a[0,0],   2*a[0,0]*a[0,1]],

        [a[1,0]**2,         a[1,1]**2,         a[1,2]**2,
         2*a[1,1]*a[1,2],   2*a[1,2]*a[1,0],   2*a[1,0]*a[1,1]],

        [a[2,0]**2,         a[2,1]**2,         a[2,2]**2,
         2*a[2,1]*a[2,2],   2*a[2,2]*a[2,0],   2*a[2,0]*a[2,1]],

        [a[1,0]*a[2,0],     a[1,1]*a[2,1],     a[1,2]*a[2,2],
         a[1,1]*a[2,2]+a[1,2]*a[2,1],
         a[1,0]*a[2,2]+a[1,2]*a[2,0],
         a[1,1]*a[2,0]+a[1,0]*a[2,1]],

        [a[2,0]*a[0,0],     a[2,1]*a[0,1],     a[2,2]*a[0,2],
         a[0,1]*a[2,2]+a[0,2]*a[2,1],
         a[0,2]*a[2,0]+a[0,0]*a[2,2],
         a[0,0]*a[2,1]+a[0,1]*a[2,0]],

        [a[0,0]*a[1,0],     a[0,1]*a[1,1],     a[0,2]*a[1,2],
         a[0,1]*a[1,2]+a[0,2]*a[1,1],
         a[0,2]*a[1,0]+a[0,0]*a[1,2],
         a[0,0]*a[1,1]+a[0,1]*a[1,0]]
    ])

    # NStar = inv(MStar') — transpose-inverse of the Bond matrix
    # Used for rotating thermal expansion coefficients (strain-like quantities)
    corr_mat_inv_T = np.linalg.inv(corr_mat.T)

    return a, corr_mat, corr_mat_inv_T, theta_rad
