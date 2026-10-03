"""
linalg_core.py
--------------
Pure calculation functions for the Linear Algebra Calculator.
No GUI code here, so other modules (addition, dot product, matrices...)
can import and reuse these.
"""

import math

import numpy as np

# float64 is NumPy's name for C's `double`: 8 bytes, ~15-16 significant digits.
# Every vector in the app uses it, so arrays are contiguous blocks of doubles
# and NumPy runs its loops in compiled C instead of in Python.
DTYPE = np.float64


def to_vector(values):
    """Convert a list/tuple of numbers to a float64 (C double) NumPy array."""
    return np.asarray(values, dtype=DTYPE)


def scalar_multiply(k, v):
    """k * v: NumPy multiplies every component in one vectorised C loop."""
    return DTYPE(k) * to_vector(v)


def magnitude(v):
    """Euclidean length ||v|| = sqrt(v1^2 + v2^2 + ...).

    math.hypot avoids overflow/underflow for very large or small components
    and, for 2-3 numbers, is much faster than building temporary arrays."""
    return math.hypot(*v)


def unit_vector(v):
    """Unit vector in the direction of v (None for the zero vector)."""
    m = magnitude(v)
    return None if m == 0 else to_vector(v) / m


def angle_2d(v):
    """Angle of a 2D vector from the positive x-axis, in degrees [0, 360)."""
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360)


# ---------------------------------------------------------------------------
# Two vectors: dot product, angle, projections, cross product
# ---------------------------------------------------------------------------

TOL = 1e-9


def dot(u, v):
    """u · v = u1*v1 + u2*v2 + ... (one number)."""
    return float(np.dot(to_vector(u), to_vector(v)))


def cos_angle(u, v):
    """cos θ = u·v / (||u|| ||v||), clamped to [-1, 1]; None if a vector is zero."""
    mu, mv = magnitude(u), magnitude(v)
    if mu == 0 or mv == 0:
        return None
    return max(-1.0, min(1.0, dot(u, v) / (mu * mv)))


def angle_between(u, v):
    """Angle between u and v in degrees (0..180); None if a vector is zero."""
    c = cos_angle(u, v)
    return None if c is None else math.degrees(math.acos(c))


def angle_kind(deg):
    if deg is None:
        return "no angle (zero vector)"
    if deg < 1e-7:
        return "same direction"
    if abs(deg - 90) < 1e-7:
        return "perpendicular"
    if abs(deg - 180) < 1e-7:
        return "opposite directions"
    return "acute angle" if deg < 90 else "obtuse angle"


def scalar_projection(v, onto):
    """Signed length of v's shadow on `onto`: v·onto / ||onto||."""
    m = magnitude(onto)
    return None if m == 0 else dot(v, onto) / m


def vector_projection(v, onto):
    """proj_onto(v) = (v·onto / ||onto||^2) onto; None if `onto` is zero."""
    onto = to_vector(onto)
    m2 = dot(onto, onto)
    return None if m2 == 0 else dot(v, onto) / m2 * onto


def cross(u, v):
    """3D: the vector u × v.  2D: the number u_x v_y - u_y v_x (signed area)."""
    u, v = to_vector(u), to_vector(v)
    if len(u) == 2:
        return float(u[0] * v[1] - u[1] * v[0])
    return np.cross(u, v).astype(DTYPE)


def distance(u, v):
    return magnitude(to_vector(u) - to_vector(v))


def relation(u, v):
    """'orthogonal', 'parallel (same direction)', 'parallel (opposite)' or 'neither'."""
    c = cos_angle(u, v)
    if c is None:
        return "undefined (zero vector)"
    if abs(c) < TOL:
        return "orthogonal"
    if abs(c - 1) < TOL:
        return "parallel (same direction)"
    if abs(c + 1) < TOL:
        return "parallel (opposite)"
    return "neither parallel nor orthogonal"


def effect_text(k):
    """Plain-language description of what multiplying by k does."""
    if k == 0:
        return "Collapses to the zero vector"
    if k == 1:
        return "Unchanged (identity)"
    if k == -1:
        return "Same length, direction reversed"
    size = "Stretched" if abs(k) > 1 else "Shrunk"
    direction = "same direction" if k > 0 else "direction reversed"
    return f"{size} by factor {abs(k):g}, {direction}"


def fmt_vec(v, digits=4):
    return "(" + ", ".join(f"{round(float(x), digits):g}" for x in v) + ")"


def describe(k, v):
    """Step-by-step explanation of k * v as a multi-line string."""
    v = to_vector(v)
    kv = scalar_multiply(k, v)
    mv, mkv = magnitude(v), magnitude(kv)

    lines = [
        f"v = {fmt_vec(v)}",
        f"k = {k:g}",
        "",
        "1) Multiply each component by k",
        "   k·v = (" + ", ".join(f"{k:g}·{x:g}" for x in v) + ")",
        f"       = {fmt_vec(kv)}",
        "",
        "2) Lengths",
        f"   ||v||   = {mv:.4f}",
        f"   ||k·v|| = {mkv:.4f}",
        f"   check: |k|·||v||",
        f"        = {abs(k):g} · {mv:.4f} = {abs(k) * mv:.4f}",
    ]
    if mv > 0:
        lines += ["", "3) Direction"]
        if len(v) == 2:
            lines.append(f"   angle(v)   = {angle_2d(v):.2f}°")
            if k != 0:
                lines.append(f"   angle(k·v) = {angle_2d(kv):.2f}°")
        u = unit_vector(v)
        lines.append(f"   û(v) = {fmt_vec(u)}")
        if k != 0:
            lines.append(f"   û(k·v) = {fmt_vec(unit_vector(kv))}")
    lines += ["", "Result: " + effect_text(k)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Python explanation: generates real, runnable code for the current numbers
# ---------------------------------------------------------------------------

def fmt(x, digits=4):
    """Short number formatting without '-0'."""
    r = round(float(x), digits)
    if r == 0:
        r = 0.0
    return f"{r:g}"


def python_code(k, v):
    """Return a commented Python program that computes k·v step by step."""
    v = to_vector(v)
    kv = scalar_multiply(k, v)
    is_2d = len(v) == 2
    arr = ", ".join(fmt(x) for x in v)
    comps = ", ".join(f"k*v[{i}]" for i in range(len(v)))
    squares = " + ".join(f"v{i}²" for i in range(len(v)))
    zero_v = magnitude(v) == 0

    c = [
        "import numpy as np",
        "",
        "# ── Inputs ───────────────────────────────",
        f"v = np.array([{arr}], dtype=float)  # the vector v",
        f"k = {fmt(k)}                      # the scalar k",
        "",
        "# ── Step 1: scalar multiplication ────────",
        "# NumPy multiplies k into EVERY component",
        "# (this is called broadcasting):",
        f"#   k*v  ==  [{comps}]",
        "kv = k * v",
        'print("k*v =", kv)',
        "",
        "# The same thing with a plain Python loop:",
        "kv_loop = [float(k * x) for x in v]",
        'print("loop version:", kv_loop)',
        "",
        "# ── Step 2: lengths (magnitudes) ─────────",
        f"# ||v|| = sqrt({squares})",
        "len_v  = np.linalg.norm(v)",
        "len_kv = np.linalg.norm(kv)",
        'print("||v||     =", round(len_v, 4))',
        'print("||k*v||   =", round(len_kv, 4))',
        "# Rule: ||k*v|| = |k| * ||v||",
        'print("|k|*||v|| =", round(abs(k) * len_v, 4))',
        "",
        "# ── Step 3: direction ────────────────────",
    ]
    if zero_v:
        c += ["# v is the zero vector: it has no direction",
              'print("v = 0, so it has no direction")']
    else:
        if is_2d:
            c += [
                "# angle from the +x axis, in degrees",
                "theta_v = np.degrees(np.arctan2(v[1], v[0])) % 360",
                'print("angle(v)   =", round(theta_v, 2))',
            ]
            if k != 0:
                c += [
                    "theta_kv = np.degrees(np.arctan2(kv[1], kv[0])) % 360",
                    'print("angle(k*v) =", round(theta_kv, 2))',
                ]
        c += [
            "# unit vector = vector / its length",
            "u_v = v / len_v",
            'print("unit(v)   =", np.round(u_v, 4))',
        ]
        if k != 0:
            c += [
                "u_kv = kv / len_kv",
                'print("unit(k*v) =", np.round(u_kv, 4))',
                "",
                "# cosine of the angle between v and k*v:",
                "#   +1 -> same direction,  -1 -> opposite",
                "cos = np.dot(v, kv) / (len_v * len_kv)",
                'print("cos(angle between) =", round(cos, 4))',
            ]
        else:
            c += ["# k = 0 -> k*v is the zero vector (no direction)"]
    c += [
        "",
        "# ── Step 4: what happened? ───────────────",
        "if k == 0:",
        '    print("collapsed to the zero vector")',
        "else:",
        '    size = "stretched" if abs(k) > 1 else ("same length" if abs(k) == 1 else "shrunk")',
        '    way = "same direction" if k > 0 else "direction reversed"',
        '    print(f"{size} by |k| = {abs(k)}, {way}")',
    ]
    return "\n".join(c)


def run_python(code):
    """Execute generated code and return what it printed."""
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__name__": "__calc__"})
    except Exception as e:  # pragma: no cover - shown to the user
        buf.write(f"Error: {e}\n")
    return buf.getvalue()
