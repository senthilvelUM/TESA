"""
Load data from an ASCII file containing a text header.
"""

import numpy as np


def hdrload(file):
    """
    Load data from an ASCII file with a text header.

    Reads lines sequentially: non-numeric lines are collected as the header,
    and numeric lines are parsed into a 2D array. The number of columns is
    determined from the first data line.

    Parameters
    ----------
    file : str
        Path to the ASCII file.

    Returns
    -------
    header : list of str
        Header lines (non-numeric lines before and interspersed with data).
    data : ndarray, shape (n_rows, n_cols)
        Numeric data array parsed from the file. Empty array if no data found.
    """
    header_lines = []
    data_values = []
    ncols = 0

    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            # Try to parse the line as numbers
            parts = line.split()
            if not parts:
                header_lines.append(line)
                continue

            try:
                vals = [float(x) for x in parts]
            except ValueError:
                # Not numeric — it's a header line
                header_lines.append(line)
                continue

            # Check for the 'e' edge case: if the line starts with 'e'
            # sscanf in MATLAB would parse it as 0.00e+00 with nxtindex=1
            stripped = line.lstrip()
            if stripped and stripped[0].lower() == 'e' and not stripped[0].isdigit():
                # Could be a header line starting with 'e'
                # If the first token is just 'e' or starts with 'e' but isn't a valid float
                try:
                    float(parts[0])
                    # It parsed OK — this IS data
                except ValueError:
                    header_lines.append(line)
                    continue

            # First data line — record ncols
            if ncols == 0:
                ncols = len(vals)
            data_values.extend(vals)

    if not data_values:
        return header_lines, np.array([])

    data = np.array(data_values, dtype=float)

    # Reshape: MATLAB reads row-wise, reshapes to (ncols, nrows) then transposes
    if ncols > 0 and len(data) % ncols == 0:
        nrows = len(data) // ncols
        data = data.reshape(nrows, ncols)
    # else: return as column vector (irregular data)

    return header_lines, data
