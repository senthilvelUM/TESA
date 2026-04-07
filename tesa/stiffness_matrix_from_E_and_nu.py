"""
Compute isotropic stiffness matrix from Young's modulus and Poisson's ratio.

Builds the 6x6 compliance matrix S, then inverts to get stiffness C.
Optionally saves the result as a TESA property file.

"""

import numpy as np


def stiffness_matrix_from_E_and_nu(E, nu, save_filename=None):
    """
    Compute 6x6 isotropic stiffness matrix from E and nu.

    Parameters
    ----------
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio.
    save_filename : str or None
        If provided, save the stiffness matrix as a TESA property file.

    Returns
    -------
    C : (6, 6) array
        Stiffness matrix (Pa).
    """
    # Build compliance matrix
    S = np.zeros((6, 6))
    S[0, 1] = -nu / E
    S[0, 2] = -nu / E
    S[1, 2] = -nu / E
    S = S + S.T
    S[0, 0] = 1.0 / E
    S[1, 1] = 1.0 / E
    S[2, 2] = 1.0 / E
    S[3, 3] = 2.0 * (1.0 + nu) / E
    S[4, 4] = S[3, 3]
    S[5, 5] = S[4, 4]

    # Invert to get stiffness matrix and enforce symmetry
    C = np.linalg.solve(S, np.eye(6))
    C = 0.5 * (C + C.T)

    # Optionally save as TESA property file
    if save_filename is not None:
        with open(save_filename, 'w') as f:
            f.write("# ──────────────────────────────────────────────────────────────\n")
            f.write("# TESA Property File\n")
            f.write("# ──────────────────────────────────────────────────────────────\n")
            f.write("#\n")
            f.write("# Generated from isotropic E and nu\n")
            f.write(f"# E  = {E:.6e} Pa\n")
            f.write(f"# nu = {nu}\n")
            f.write("#\n")
            f.write("# Keywords:\n")
            f.write("#   *phase                  Phase/material name\n")
            f.write("#   *density                Scalar (kg/m^3)\n")
            f.write("#   *stiffness_matrix       6x6 matrix (Pa)\n")
            f.write("#   *thermal_expansion      6x1 vector (1/K)\n")
            f.write("#   *thermal_conductivity   3x3 matrix (W/(m K))\n")
            f.write("#\n")
            f.write("# Notes:\n")
            f.write("#   - Comments (#) and blank lines are ignored by the parser\n")
            f.write("#   - Use \"# source:\" to document data provenance\n")
            f.write("#   - Missing keywords use defaults: *phase=filename,\n")
            f.write("#     *density=1.0, *stiffness_matrix=identity,\n")
            f.write("#     *thermal_expansion=zeros, *thermal_conductivity=identity\n")
            f.write("#\n")
            f.write("# ──────────────────────────────────────────────────────────────\n\n")
            f.write("*phase\n")
            f.write("Isotropic\n\n")
            f.write("*density\n")
            f.write("1.0\n\n")
            f.write("*stiffness_matrix\n")
            for row in range(6):
                f.write("   " + "  ".join(f"{C[row, col]:13.4e}" for col in range(6)) + "\n")
            f.write("\n*thermal_expansion\n")
            for _ in range(6):
                f.write("   0\n")
            f.write("\n*thermal_conductivity\n")
            f.write("   0 0 0\n")
            f.write("   0 0 0\n")
            f.write("   0 0 0\n")

    return C
