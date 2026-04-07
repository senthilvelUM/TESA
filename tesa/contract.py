"""
Voigt notation index contraction.

Maps tensor indices (i, j) to Voigt notation index:
  (1,1)->1  (2,2)->2  (3,3)->3  (2,3)->4  (1,3)->5  (1,2)->6

"""


def contract(i, j):
    """
    Contract symmetric tensor indices (i, j) to Voigt notation index.

    Mapping: (1,1)->1, (2,2)->2, (3,3)->3, (2,3)->4, (1,3)->5, (1,2)->6.
    The mapping is symmetric, so ``contract(i, j) == contract(j, i)``.

    Parameters
    ----------
    i : int
        First tensor index (1-based, 1 to 3).
    j : int
        Second tensor index (1-based, 1 to 3).

    Returns
    -------
    index : int
        Voigt notation index (1-based, 1 to 6).
    """
    if i == 1 and j == 1:
        index = 1
    elif i == 2 and j == 2:
        index = 2
    elif i == 3 and j == 3:
        index = 3
    elif (i == 2 and j == 3) or (i == 3 and j == 2):
        index = 4
    elif (i == 1 and j == 3) or (i == 3 and j == 1):
        index = 5
    elif (i == 1 and j == 2) or (i == 2 and j == 1):
        index = 6
    else:
        raise ValueError(f"Invalid tensor indices: ({i}, {j})")
    return index
