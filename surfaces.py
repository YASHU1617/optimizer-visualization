"""
surfaces.py
------------
The elongated-bowl loss surfaces used in Part A:
    L(x, y) = x^2 + k*y^2,  grad = [2x, 2k*y]
k=1  -> circular bowl (not used in this lab, kept for reference)
k=10, 50 (default), 100, 1000 -> increasingly ill-conditioned bowls,
used to explore how curvature/conditioning causes zig-zagging.
"""

import numpy as np

SURFACES = {
    "L1: x^2 + 10y^2": 10,
    "L2: x^2 + 50y^2 (default)": 50,
    "L3: x^2 + 100y^2": 100,
    "L4: x^2 + 1000y^2": 1000,
}

DEFAULT_SURFACE = "L2: x^2 + 50y^2 (default)"


def loss(params, k):
    x, y = params
    return x ** 2 + k * y ** 2


def grad(params, k):
    x, y = params
    return np.array([2.0 * x, 2.0 * k * y])


def condition_number(k):
    """Ratio of the Hessian's largest to smallest eigenvalue: 2k / 2 = k.
    Larger k -> narrower bowl -> worse zig-zagging for plain SGD."""
    return float(k)
