"""
topic_util.py
-------------
Small helpers shared by the topic tabs: number formatting for formulas and
for generated Python code.
"""

import math

import linalg_core as la
from algebra_render import paren

F = la.fmt          # 3.0 -> "3", 0.333333 -> "0.3333"
P = paren           # like F, but negatives in brackets: "(-2)"


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def deg(x):
    return "undefined" if x is None else f"{x:.2f}°"


def np_array(v):
    """Source text for a NumPy array literal."""
    return "np.array([" + ", ".join(F(x) for x in v) + "], dtype=float)"


def sq_sum(v):
    """'3^2 + 4^2' style mathtext."""
    return " + ".join(f"{P(x)}^2" for x in v)


def sqrt_tex(v):
    return r"\sqrt{" + sq_sum(v) + "}"
