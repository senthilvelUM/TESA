"""
Compute seismic wave speeds over a hemisphere and return anisotropy indices.

Generates propagation directions on a unit hemisphere using sphere sampling,
calls wavespeeds() for each direction, then computes anisotropy percentages
and shear wave splitting.

"""

import numpy as np
from .wavespeeds import wavespeeds as _wavespeeds


def get_wave_speeds(C, rho, n=200):
    """
    Compute wave speeds over a hemisphere and anisotropy indices.

    Parameters
    ----------
    C : (6, 6) array
        Elastic stiffness matrix (Pa).
    rho : float
        Density (kg/m^3).
    n : int
        Number of sampling points per dimension on the sphere (even number).
        Default is 200. Higher values give smoother results.

    Returns
    -------
    VS : dict
        Dictionary containing:
        - 'XC', 'YC', 'ZC': lower hemisphere coordinates, shape (n/2+1, n+1)
        - 'VP', 'VS1', 'VS2', 'VSH', 'VSV': wave speeds, shape (n/2+1, n+1)
        - 'AVS': shear wave splitting %, shape (n/2+1, n+1)
        - 'DTS': shear wave delay, shape (n/2+1, n+1)
        - 'DTP': P-wave delay, shape (n/2+1, n+1)
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
    # Ensure n is even
    if n % 2 == 1:
        n = n + 1

    # Generate sphere coordinates (lower hemisphere only, z from -1 to 0)
    # linspace(0, pi) gives z from +1 to -1, so we take rows n//2 to n
    # and flip to get z-ascending order.
    u = np.linspace(0, 2 * np.pi, n + 1)
    v_angle = np.linspace(0, np.pi, n + 1)
    U, V_ang = np.meshgrid(u, v_angle)
    XC_full = np.sin(V_ang) * np.cos(U)
    YC_full = np.sin(V_ang) * np.sin(U)
    ZC_full = np.cos(V_ang)

    # Take lower hemisphere and flip to z-ascending order (-1 to 0)
    half = n // 2 + 1
    XC = np.flip(XC_full[n // 2:, :], axis=0).copy()
    YC = np.flip(YC_full[n // 2:, :], axis=0).copy()
    ZC = np.flip(ZC_full[n // 2:, :], axis=0).copy()

    # Flatten for wavespeeds calculation
    XC_flat = XC.ravel()
    YC_flat = YC.ravel()
    ZC_flat = ZC.ravel()

    # Convert Cartesian to spherical (azimuth/elevation convention)
    # Theta = azimuth, Phi = elevation
    Theta = np.arctan2(YC_flat, XC_flat)
    R = np.sqrt(XC_flat ** 2 + YC_flat ** 2 + ZC_flat ** 2)
    Phi = np.arcsin(np.clip(ZC_flat / np.maximum(R, 1e-30), -1, 1))

    phi = Theta
    gamma = np.pi / 2 - Phi

    # Compute wave speeds
    v_raw, ev, Vs = _wavespeeds(C, rho, phi, gamma)

    # Reshape to hemisphere grid
    VP  = Vs[:, 0].reshape(half, n + 1)
    VS1 = Vs[:, 1].reshape(half, n + 1)
    VS2 = Vs[:, 2].reshape(half, n + 1)
    VSH = Vs[:, 3].reshape(half, n + 1)
    VSV = Vs[:, 4].reshape(half, n + 1)

    # Anisotropy percentages: (max - min) / mean * 100
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

    # Return values
    VS_dict = {
        'XC': XC, 'YC': YC, 'ZC': ZC,
        'VP': VP, 'VS1': VS1, 'VS2': VS2, 'VSH': VSH, 'VSV': VSV,
        'AVS': AVS, 'DTS': DTS, 'DTP': DTP,
    }

    return VS_dict, AVP, AVS1, AVS2, AVSH, AVSV, MaxAVS, v_raw, ev
