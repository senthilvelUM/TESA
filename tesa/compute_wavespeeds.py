"""
Compute seismic wave speeds over a full sphere and return anisotropy indices.

Similar to get_wave_speeds but uses the full sphere (not just hemisphere)
and converts velocities to km/s.

"""

import numpy as np
from .wavespeeds import wavespeeds as _wavespeeds


def compute_wavespeeds(C, rho, n=200):
    """
    Compute wave speeds over a full sphere and anisotropy indices.

    Parameters
    ----------
    C : (6, 6) array
        Elastic stiffness matrix (Pa).
    rho : float
        Density (kg/m^3).
    n : int
        Number of sampling points per dimension on the sphere.

    Returns
    -------
    VS : dict
        Dictionary containing:
        - 'XC', 'YC', 'ZC': full sphere coordinates, shape (n+1, n+1)
        - 'VP', 'VS1', 'VS2', 'VSH', 'VSV': wave speeds in km/s, shape (n+1, n+1)
        - 'AVS': shear wave splitting %, shape (n+1, n+1)
        - 'DTS': shear wave delay, shape (n+1, n+1)
        - 'DTP': P-wave delay, shape (n+1, n+1)
    AVP : float
        P-wave anisotropy (%).
    AVS1 : float
        Fast S-wave anisotropy (%).
    AVS2 : float
        Slow S-wave anisotropy (%).
    AVSH : float
        SH-wave anisotropy (%).
    AVSV : float
        SV-wave anisotropy (%).
    MaxAVS : float
        Maximum shear wave splitting (%).
    v : (1, 3, nDirections) array
        Raw eigenvalue-based wave speeds.
    ev : (3, 3, nDirections) array
        Eigenvectors (polarization directions).
    """
    # Generate full sphere coordinates
    u = np.linspace(0, 2 * np.pi, n + 1)
    v_angle = np.linspace(0, np.pi, n + 1)
    U, V_ang = np.meshgrid(u, v_angle)
    XC = np.sin(V_ang) * np.cos(U)
    YC = np.sin(V_ang) * np.sin(U)
    ZC = np.cos(V_ang)

    # Flatten for wavespeeds calculation
    XC_flat = XC.ravel()
    YC_flat = YC.ravel()
    ZC_flat = ZC.ravel()

    # Convert Cartesian to spherical (azimuth/elevation convention)
    Theta = np.arctan2(YC_flat, XC_flat)
    R = np.sqrt(XC_flat ** 2 + YC_flat ** 2 + ZC_flat ** 2)
    Phi = np.arcsin(np.clip(ZC_flat / np.maximum(R, 1e-30), -1, 1))

    phi = Theta
    gamma = np.pi / 2 - Phi

    # Compute wave speeds
    v_raw, ev, Vs = _wavespeeds(C, rho, phi, gamma)

    # Reshape and convert to km/s
    VP  = Vs[:, 0].reshape(n + 1, n + 1) / 1e3
    VS1 = Vs[:, 1].reshape(n + 1, n + 1) / 1e3
    VS2 = Vs[:, 2].reshape(n + 1, n + 1) / 1e3
    VSH = Vs[:, 3].reshape(n + 1, n + 1) / 1e3
    VSV = Vs[:, 4].reshape(n + 1, n + 1) / 1e3

    # Anisotropy percentages
    def _aniso(arr):
        vmax = np.max(arr)
        vmin = np.min(arr)
        vmean = (vmax + vmin) / 2.0
        if vmean > 0:
            return (vmax - vmin) / vmean * 100.0
        return 0.0

    AVP  = _aniso(VP)
    AVS1 = _aniso(VS1)
    AVS2 = _aniso(VS2)
    AVSH = _aniso(VSH)
    AVSV = _aniso(VSV)

    # Shear wave splitting
    AVS = (VS1 - VS2) / ((VS1 + VS2) / 2.0) * 100.0
    MaxAVS = np.max(AVS)

    # Delay times
    DTS = 1.0 / VS2 - 1.0 / VS1
    maxVP = np.max(VP)
    DTP = 1.0 / VP - 1.0 / maxVP

    VS_dict = {
        'XC': XC, 'YC': YC, 'ZC': ZC,
        'VP': VP, 'VS1': VS1, 'VS2': VS2, 'VSH': VSH, 'VSV': VSV,
        'AVS': AVS, 'DTS': DTS, 'DTP': DTP,
    }

    return VS_dict, AVP, AVS1, AVS2, AVSH, AVSV, MaxAVS, v_raw, ev
