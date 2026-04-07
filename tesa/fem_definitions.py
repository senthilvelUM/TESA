"""
Constants and variables for the Finite Element implementation of the
Asymptotic Expansion Homogenization (AEH) method.

Defines quadrature points, weights, and element properties for
6-node isoparametric triangular elements.

Copyright 2014-2015, Alden C. Cook.
"""

import numpy as np


# Number of quadrature points per element
N_QUADRATURE_POINTS = 4

# Number of nodes per element (6-node quadratic triangle)
N_ELEMENT_NODES = 6

# Master element quadrature point coordinates
r = np.array([1/3, 6/10, 2/10, 2/10])
s = np.array([1/3, 2/10, 6/10, 2/10])

# Quadrature weights
w = np.array([-0.5625, 1.5625/3.0, 1.5625/3.0, 1.5625/3.0])
QUADRATURE_WEIGHTS = w
