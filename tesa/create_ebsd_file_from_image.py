"""
Create a synthetic EBSD file from a color image.

Reads an image where each unique color represents a different phase,
assigns Euler angles (random or zero), and writes an .ang-style EBSD
text file. The output file has columns:
  [phi1, Phi, phi2, X, Y, phase]

Enforces periodicity by matching first and last rows/columns.

"""

import numpy as np
from PIL import Image


def create_ebsd_file_from_image(image_filename, save_filename,
                                 angle_method='zeros', seed=None):
    """
    Create a synthetic EBSD file from an image.

    Parameters
    ----------
    image_filename : str
        Path to the input image (PNG, JPG, etc.).
    save_filename : str
        Path for the output EBSD text file (.ang or .txt).
    angle_method : str
        Method for assigning Euler angles:
        - 'zeros': all angles set to zero
        - 'random': random angles per phase
    seed : int or None
        Random seed for reproducibility (used when angle_method='random').

    Returns
    -------
    data : (nPoints, 6) array
        The EBSD data: [phi1, Phi, phi2, X, Y, phase].
    """
    if seed is not None:
        np.random.seed(seed)

    # Read image
    img = Image.open(image_filename).convert('RGB')
    A = np.array(img)
    ny, nx = A.shape[0], A.shape[1]

    # Setup grid coordinates
    X_grid, Y_grid = np.meshgrid(np.linspace(0, nx - 1, nx),
                                  np.linspace(0, ny - 1, ny))
    # Transpose and flatten in column-major order
    X = X_grid.T.ravel()
    Y = Y_grid.T.ravel()

    # Process image channels (transpose and flip to match EBSD coordinate orientation)
    A1 = A[:, :, 0].T
    A2 = A[:, :, 1].T
    A3 = A[:, :, 2].T
    An = np.zeros((A1.shape[0], A1.shape[1], 3), dtype=A.dtype)
    An[:, :, 0] = np.fliplr(A1)
    An[:, :, 1] = np.fliplr(A2)
    An[:, :, 2] = np.fliplr(A3)
    A_flat = An.reshape(-1, 3)

    # Find unique colors (each color = one phase)
    _, first_idx = np.unique(A_flat, axis=0, return_index=True)
    first_idx = np.sort(first_idx)
    colors = A_flat[first_idx]

    # Assign phases and Euler angles
    phases = np.ones(len(X), dtype=int)
    angles = np.zeros((len(X), 3))

    for i in range(len(colors)):
        # Find pixels matching this color
        mask = ((A_flat[:, 0] == colors[i, 0]) &
                (A_flat[:, 1] == colors[i, 1]) &
                (A_flat[:, 2] == colors[i, 2]))
        phases[mask] = i + 1

        # Assign Euler angles
        if angle_method == 'random':
            angles[mask, 0] = 2 * np.pi * np.random.rand()
            angles[mask, 1] = np.pi * np.random.rand()
            angles[mask, 2] = 2 * np.pi * np.random.rand()
        # 'zeros': do nothing (already zeros)

    # Reshape to grid, enforce periodicity, reshape back
    ANG1 = angles[:, 0].reshape(nx, ny).T
    ANG2 = angles[:, 1].reshape(nx, ny).T
    ANG3 = angles[:, 2].reshape(nx, ny).T
    PH = phases.reshape(nx, ny).T

    # Match first and last rows for periodicity
    ANG1[-1, :] = ANG1[0, :]
    ANG2[-1, :] = ANG2[0, :]
    ANG3[-1, :] = ANG3[0, :]
    PH[-1, :] = PH[0, :]

    # Match first and last columns for periodicity
    ANG1[:, -1] = ANG1[:, 0]
    ANG2[:, -1] = ANG2[:, 0]
    ANG3[:, -1] = ANG3[:, 0]
    PH[:, -1] = PH[:, 0]

    # Reform to match XY ordering
    angles_out = np.column_stack([
        ANG1.T.ravel(),
        ANG2.T.ravel(),
        ANG3.T.ravel(),
    ])
    phases_out = PH.T.ravel()

    # Assemble output data: [phi1, Phi, phi2, X, Y_flipped, phase]
    data = np.column_stack([
        angles_out.astype(float),
        X.astype(float),
        np.flipud(Y).astype(float),
        phases_out.astype(float),
    ])

    # Save to file
    np.savetxt(save_filename, data, fmt='%15.6e')

    return data
