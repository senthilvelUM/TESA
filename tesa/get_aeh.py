"""
Post-processing utilities for AEH results.

Extracts engineering properties (E, G, nu) from an effective stiffness
matrix and optionally saves results to text files. Also provides a
utility to clean small matrix entries.

Contains the post-processing/utility portions. The solver is in
aehfe_thermoelastic_analysis.py.
"""

import os
import numpy as np


def clean_stiffness(C, tol_ratio=1e-6):
    """
    Zero out stiffness entries that are negligibly small.

    Sets entries whose absolute value is less than tol_ratio times
    the maximum absolute entry to zero. Also symmetrizes the matrix.

    Parameters
    ----------
    C : (6, 6) array
        Stiffness matrix.
    tol_ratio : float
        Tolerance ratio relative to max entry. Default 1e-6.

    Returns
    -------
    C_clean : (6, 6) array
        Cleaned stiffness matrix.
    """
    C_clean = 0.5 * (C + C.T)
    threshold = np.max(np.abs(C_clean)) * tol_ratio
    C_clean[np.abs(C_clean) < threshold] = 0.0
    return C_clean


def engineering_properties(C):
    """
    Extract engineering properties from a stiffness matrix.

    Parameters
    ----------
    C : (6, 6) array
        Stiffness matrix (Pa).

    Returns
    -------
    props : dict
        Dictionary with keys:
        - 'E1', 'E2', 'E3': Young's moduli (Pa)
        - 'G23', 'G13', 'G12': Shear moduli (Pa)
        - 'nu12', 'nu13', 'nu21', 'nu23', 'nu31', 'nu32': Poisson's ratios
    """
    S = np.linalg.inv(C)
    E1 = 1.0 / S[0, 0]
    E2 = 1.0 / S[1, 1]
    E3 = 1.0 / S[2, 2]
    G23 = 1.0 / S[3, 3]
    G13 = 1.0 / S[4, 4]
    G12 = 1.0 / S[5, 5]
    nu12 = -S[1, 0] * E1
    nu13 = -S[2, 0] * E1
    nu21 = -S[0, 1] * E2
    nu23 = -S[2, 1] * E2
    nu31 = -S[0, 2] * E3
    nu32 = -S[1, 2] * E3

    return {
        'E1': E1, 'E2': E2, 'E3': E3,
        'G23': G23, 'G13': G13, 'G12': G12,
        'nu12': nu12, 'nu13': nu13,
        'nu21': nu21, 'nu23': nu23,
        'nu31': nu31, 'nu32': nu32,
    }


def save_stiffness_file(C, filepath, title="Effective Elastic Stiffnesses (Pa):"):
    """
    Save a stiffness matrix to a formatted text file.

    Parameters
    ----------
    C : (6, 6) array
        Stiffness matrix.
    filepath : str
        Output file path.
    title : str
        Header title.
    """
    with open(filepath, 'w') as f:
        f.write(f"{title}\n\n")
        for row in range(6):
            f.write("  ".join(f"{C[row, col]:12.5e}" for col in range(6)) + "\n")


def save_engineering_properties_file(props, filepath,
                                      title="Effective Engineering Properties:"):
    """
    Save engineering properties to a formatted text file.

    Parameters
    ----------
    props : dict
        Dictionary from engineering_properties().
    filepath : str
        Output file path.
    title : str
        Header title.
    """
    with open(filepath, 'w') as f:
        f.write(f"{title}\n\n")
        f.write(f"E1 = {props['E1']/1e9:10.6g} GPa\n")
        f.write(f"E2 = {props['E2']/1e9:10.6g} GPa\n")
        f.write(f"E3 = {props['E3']/1e9:10.6g} GPa\n")
        f.write(f"G23 = {props['G23']/1e9:10.6g} GPa\n")
        f.write(f"G13 = {props['G13']/1e9:10.6g} GPa\n")
        f.write(f"G12 = {props['G12']/1e9:10.6g} GPa\n")
        f.write(f"nu12 = {props['nu12']:10.6g}\n")
        f.write(f"nu13 = {props['nu13']:10.6g}\n")
        f.write(f"nu21 = {props['nu21']:10.6g}\n")
        f.write(f"nu23 = {props['nu23']:10.6g}\n")
        f.write(f"nu31 = {props['nu31']:10.6g}\n")
        f.write(f"nu32 = {props['nu32']:10.6g}\n")


def clean_conductivity(kappa, tol_ratio=1e-6):
    """
    Zero out conductivity entries that are negligibly small.

    Parameters
    ----------
    kappa : (3, 3) array
        Thermal conductivity matrix.
    tol_ratio : float
        Tolerance ratio relative to max entry. Default 1e-6.

    Returns
    -------
    kappa_clean : (3, 3) array
        Cleaned conductivity matrix.
    """
    kappa_clean = 0.5 * (kappa + kappa.T)
    threshold = np.max(np.abs(kappa_clean)) * tol_ratio
    kappa_clean[np.abs(kappa_clean) < threshold] = 0.0
    return kappa_clean


def save_thermal_conductivity_file(kappa, filepath,
                                     title="Effective Thermal Conductivity (W/(m K)):"):
    """
    Save a thermal conductivity matrix to a formatted text file.

    Parameters
    ----------
    kappa : (3, 3) array
        Thermal conductivity matrix.
    filepath : str
        Output file path.
    title : str
        Header title.
    """
    with open(filepath, 'w') as f:
        f.write(f"{title}\n\n")
        for row in range(3):
            f.write("  ".join(f"{kappa[row, col]:12.5e}" for col in range(3)) + "\n")
