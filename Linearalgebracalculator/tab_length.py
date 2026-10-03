"""
tab_length.py
-------------
Vector length (magnitude / norm): Pythagoras in 2D and twice in 3D, the
unit vector, and the rules lengths follow.
"""

import math

import numpy as np

import linalg_core as la
from algebra_render import nums
from geo_scene import right_angle_points, unit
from tab_base import GeoTab
from topic_util import F, P, close, np_array, sqrt_tex


class VectorLengthTab(GeoTab):
    title = "  Vector Length  "

    def build_controls(self, side):
        self.add_dim_selector(side)
        self.v = self.add_vector_box(side, "Vector v", "U", ("3", "4", "12"))
        box = self.add_box(side, "Property check uses")
        self.k, _ = self.add_entry_row(box, "scalar k =", "-2")
        self.show_legs, self.show_unit, self.show_circle = self.add_checks(
            side, "Display", [("Components (Pythagoras)", True), ("Unit vector û", True),
                              ("Circle of radius ‖v‖ (2D)", True)])
        self.add_actions(side)

    def read_params(self):
        v, err = self.read_vector(self.v, "v")
        k, err2 = self.read_number(self.k, "k")
        if err or err2:
            return None, err or err2
        return dict(dim=self.ndim(), v=v, k=k, legs=self.show_legs.get(),
                    unit=self.show_unit.get(), circle=self.show_circle.get()), None

    def extent(self, p):
        return max(la.magnitude(p["v"]), 1.0)

    def legend(self, p):
        th = self.th
        return [("v", th["v"], "line"), ("û  (unit vector)", th["w"], "line"),
                ("components", th["proj"], "dash")]

    def geometry(self, s, p):
        v, c = p["v"], s.col
        L = la.magnitude(v)
        size = max(L, 1.0)
        if p["legs"]:
            if len(v) == 2:
                x_end = np.array([v[0], 0.0])
                s.segment("lx", [0, 0], x_end, c["p"], ls="--", lw=1.8)
                s.segment("ly", x_end, v, c["p"], ls="--", lw=1.8)
                if v[0] and v[1]:
                    s.path("sq", right_angle_points(x_end, -x_end, v - x_end, size * 0.045),
                           c["muted"], lw=1)
                s.label("tx", [v[0] / 2, -math.copysign(size * 0.07, v[1] or 1)],
                        f"x = {F(v[0], 2)}", c["p"], size=9)
                s.label("ty", [v[0] + math.copysign(size * 0.12, v[0] or 1), v[1] / 2],
                        f"y = {F(v[1], 2)}", c["p"], size=9)
            else:
                a, b2 = np.array([v[0], 0, 0]), np.array([v[0], v[1], 0])
                s.path("legs", [np.zeros(3), a, b2, v], c["p"], ls="--", lw=1.8)
                s.segment("diag", np.zeros(3), b2, c["muted"], ls=":", lw=1.4)
                s.label("td", b2 * 0.55, "d", c["muted"], size=10)
                s.label("tz", (b2 + v) / 2, f"  z = {F(v[2], 2)}", c["p"], size=9, ha="left")
        if p["circle"] and len(v) == 2 and L > 0:
            t = np.linspace(0, 2 * math.pi, 120)
            s.path("circle", np.c_[L * np.cos(t), L * np.sin(t)], c["muted"], ls=":", lw=1.2)
            s.path("ucircle", np.c_[np.cos(t), np.sin(t)], c["muted"], ls="--", lw=0.8,
                   alpha=0.6)
        s.arrow("v", v, c["u"], width=0.011, zorder=4)
        if p["unit"] and L > 0:
            s.arrow("u", v / L, c["w"], width=0.012, zorder=5)
            s.label("tu", v / L * 0.5 + _side(v) * size * 0.06, "û", c["w"], size=11)
        if L > 0:
            s.label("tv", v * 0.62 + _side(v) * size * 0.1, f"‖v‖ = {F(L)}", c["u"], size=10)
        s.title(f"‖v‖ = {F(L)}\nlength of v = √(sum of squared components)")

    def status_text(self, p):
        L = la.magnitude(p["v"])
        u = la.unit_vector(p["v"])
        return f"‖v‖ = {F(L)}   —   unit vector û = {la.fmt_vec(u) if u is not None else 'none'}"

    def algebra(self, p):
        return length_steps, (p["v"], p["k"])

    def python_code(self, p):
        return length_code(p["v"], p["k"])


def _side(v):
    """A unit direction perpendicular to v (on the 'upper' side), for labels."""
    if len(v) == 2:
        n = unit([-v[1], v[0]])
        if n is None:
            return np.array([0.0, 1.0])
        return n if n[1] >= 0 else -n
    return np.array([0.0, 0.0, 1.0])


def length_steps(b, v, k):
    pal = b.pal
    U, W, INK = pal["v"], pal["w"], pal["ink"]
    v = la.to_vector(v)
    n = len(v)
    L = la.magnitude(v)
    s2 = float(np.dot(v, v))
    names = "xyz"[:n]

    b.heading("Pythagoras: square, add, take the square root")
    b.row([r"$\Vert\mathbf{v}\Vert = \sqrt{" + " + ".join(f"{c}^2" for c in names) + "}$"])
    b.row([f"$= {sqrt_tex(v)}$"], indent=40)
    b.row([r"$= \sqrt{" + " + ".join(F(x * x) for x in v) + rf"}} = \sqrt{{{F(s2)}}}"
           rf" \approx {F(L)}$"], indent=40)
    if n == 2:
        b.note("v is the long side (hypotenuse) of a right triangle whose short sides "
               "are its components x and y.")
    else:
        d = math.hypot(v[0], v[1])
        b.heading("In 3D: Pythagoras twice")
        b.note("First the diagonal d across the floor (x, y), then up the height z:")
        b.row([rf"$d = \sqrt{{{P(v[0])}^2 + {P(v[1])}^2}} \approx {F(d)}$"])
        b.row([rf"$\Vert\mathbf{{v}}\Vert = \sqrt{{d^2 + {P(v[2])}^2}} = "
               rf"\sqrt{{{F(d * d)} + {F(v[2] ** 2)}}} \approx {F(L)}$"])

    b.heading("Unit vector: same direction, length 1")
    if L == 0:
        b.note("v is the zero vector: it has length 0 and no direction, so it has no "
               "unit vector.", color=INK)
    else:
        u = v / L
        b.row([r"$\hat{\mathbf{u}} = \frac{\mathbf{v}}{\Vert\mathbf{v}\Vert} = "
               rf"\frac{{1}}{{{F(L)}}}$", ("vec", nums(v), U), "$=$", ("vec", nums(u), W)])
        b.check(r"\Vert\hat{\mathbf{u}}\Vert", r"\sqrt{" + " + ".join(
            f"{P(x, 3)}^2" for x in u) + "} = 1", close(la.magnitude(u), 1.0))
        b.note("Dividing by the length is called normalising the vector.")

    b.heading("Rules that lengths follow")
    b.subhead("1. Never negative:  ‖v‖ ≥ 0, and ‖v‖ = 0 only for v = 0")
    b.subhead(f"2. Scaling:  ‖k v‖ = |k| ‖v‖   with k = {F(k)}")
    kv = k * v
    b.check(r"\Vert k\mathbf{v}\Vert = " + F(la.magnitude(kv)),
            f"{F(abs(k))}\\cdot{F(L)} = {F(abs(k) * L)}",
            close(la.magnitude(kv), abs(k) * L))
    b.subhead("3. Length from the dot product:  ‖v‖² = v·v")
    b.check(f"{F(L)}^2", f"{F(s2)}", close(L * L, s2))
    b.subhead("4. Opposite vector, same length:  ‖−v‖ = ‖v‖")
    b.check(F(la.magnitude(-v)), F(L), close(la.magnitude(-v), L))


def length_code(v, k):
    return "\n".join([
        "import math",
        "import numpy as np",
        "",
        "# ── Inputs ───────────────────────────────",
        f"v = {np_array(v)}",
        f"k = {F(k)}",
        "",
        "# ── Step 1: Pythagoras ───────────────────",
        "squares = v ** 2",
        'print("squares       =", squares)',
        "length = math.sqrt(squares.sum())",
        'print("||v||         =", round(length, 4))',
        'print("np.linalg.norm:", round(np.linalg.norm(v), 4))',
        "",
        "# ── Step 2: unit vector ──────────────────",
        "if length == 0:",
        '    print("zero vector: no unit vector")',
        "else:",
        "    u_hat = v / length",
        '    print("unit vector   =", np.round(u_hat, 4))',
        '    print("its length    =", round(np.linalg.norm(u_hat), 4))',
        "",
        "# ── Step 3: rules (all should be True) ───",
        'print("||k v|| = |k| ||v|| :", np.isclose(np.linalg.norm(k * v), abs(k) * length))',
        'print("||v||^2 = v·v       :", np.isclose(length ** 2, v @ v))',
        'print("||-v|| = ||v||      :", np.isclose(np.linalg.norm(-v), length))',
    ])
