"""
tab_cauchy.py
-------------
Cauchy–Schwarz inequality  |u·v| ≤ ‖u‖ ‖v‖: why it holds (|cos θ| ≤ 1, or
"a shadow is never longer than the stick"), when it becomes an equality
(parallel vectors), and the triangle inequality that follows from it.
"""

import math
from tkinter import ttk

import numpy as np

import linalg_core as la
from geo_scene import right_angle_points, unit
from tab_base import GeoTab
from topic_util import F, P, close, np_array


class CauchySchwarzTab(GeoTab):
    title = "  Cauchy–Schwarz  "

    def build_controls(self, side):
        self.add_dim_selector(side)
        self.u = self.add_vector_box(side, "Vector u", "U", ("4", "1", "1"))
        self.v = self.add_vector_box(side, "Vector v", "V", ("2", "3", "-1"))
        self.show_shadow, self.show_gauge = self.add_checks(
            side, "Display", [("Shadow of v on u (projection)", True),
                              ("Gauge |u·v| ÷ ‖u‖‖v‖ (2D)", True)])
        quick = self.add_box(side, "Try making v")
        row = ttk.Frame(quick)
        row.pack(fill="x")
        for text, kind in (("parallel to u", "parallel"), ("perpendicular to u", "perp")):
            ttk.Button(row, text=text, style="Small.TButton",
                       command=lambda k=kind: self._set_v(k)).pack(side="left", padx=(0, 6))
        self.add_actions(side)

    def _set_v(self, kind):
        """v = ½u (equality case) or a vector at right angles to u (|u·v| = 0)."""
        u, err = self.read_vector(self.u, "u")
        if err:
            return
        if kind == "parallel":
            new = 0.5 * u
        elif len(u) == 2:
            new = np.array([-u[1], u[0]])
        else:
            new = np.cross(u, np.eye(3)[int(np.argmin(np.abs(u)))])
        for var, x in zip(self.v, new):
            var.set(F(x, 3))

    def read_params(self):
        u, err = self.read_vector(self.u, "u")
        v, err2 = self.read_vector(self.v, "v")
        if err or err2:
            return None, err or err2
        return dict(dim=self.ndim(), u=u, v=v, shadow=self.show_shadow.get(),
                    gauge=self.show_gauge.get()), None

    def extent(self, p):
        return max(la.magnitude(p["u"]), la.magnitude(p["v"]), 1.0)

    def legend(self, p):
        th = self.th
        return [("u", th["v"], "line"), ("v", th["kv"], "line"),
                ("shadow of v on u", th["proj"], "line")]

    def geometry(self, s, p):
        u, v, c = p["u"], p["v"], s.col
        L = self.extent(p)
        d = la.dot(u, v)
        mu, mv = la.magnitude(u), la.magnitude(v)
        if p["shadow"] and mu > 0:
            pr = la.vector_projection(v, u)
            s.segment("drop", v, pr, c["muted"], ls=":", lw=1.4)
            s.arrow("proj", pr, c["p"], width=0.008, glow=False, zorder=3)
            if np.any(pr) and np.linalg.norm(v - pr) > 1e-9 * L:
                s.path("sq", right_angle_points(pr, -pr, v - pr, L * 0.045), c["muted"], lw=1)
            if len(u) == 2 and mv > 0:              # the stick v swung onto u's line
                t = np.linspace(0, 2 * math.pi, 120)
                s.path("circle", np.c_[mv * np.cos(t), mv * np.sin(t)], c["muted"],
                       ls=":", lw=1.0, alpha=0.7)
        if p["gauge"] and len(u) == 2:
            y = 1.05 * L                             # above the vectors, clear of the legend
            left, right = -L, L
            ratio = abs(d) / (mu * mv) if mu and mv else 0.0
            s.segment("g_track", [left, y], [right, y], c["muted"], lw=7, alpha=0.25)
            s.segment("g_fill", [left, y], [left + (right - left) * ratio, y], c["w"], lw=7)
            s.label("g_text", [0, y + 0.09 * L],
                    f"|u·v| = {F(abs(d), 3)}   ≤   ‖u‖‖v‖ = {F(mu * mv, 3)}", c["text"],
                    size=9)
        s.arrow("v", v, c["v"], width=0.009, zorder=4)
        s.arrow("u", u, c["u"], width=0.011, zorder=5)
        for key, vec, col in (("lu", u, c["u"]), ("lv", v, c["v"])):
            n = unit(vec)
            if n is not None:
                s.label(key, vec + n * L * 0.09, key[1], col, size=12)
        s.title(f"|u·v| = {F(abs(d))}  ≤  ‖u‖‖v‖ = {F(mu * mv)}\n"
                f"{'equal: u and v are parallel' if _equal(u, v) else 'strictly smaller'}")

    def status_text(self, p):
        u, v = p["u"], p["v"]
        mu, mv = la.magnitude(u), la.magnitude(v)
        d = la.dot(u, v)
        ratio = f"{abs(d) / (mu * mv):.4f}" if mu and mv else "—"
        return f"|u·v| = {F(abs(d))}   ‖u‖‖v‖ = {F(mu * mv)}   ratio = {ratio}"

    def algebra(self, p):
        return cauchy_steps, (p["u"], p["v"])

    def python_code(self, p):
        return cauchy_code(p["u"], p["v"])


def _equal(u, v):
    """Equality in Cauchy–Schwarz: parallel (or a zero vector)."""
    mu, mv = la.magnitude(u), la.magnitude(v)
    return mu * mv == 0 or close(abs(la.dot(u, v)), mu * mv)


def cauchy_steps(b, u, v):
    INK = b.pal["ink"]
    u, v = la.to_vector(u), la.to_vector(v)
    d = la.dot(u, v)
    mu, mv = la.magnitude(u), la.magnitude(v)
    UV = r"\mathbf{u}\cdot\mathbf{v}"
    NU, NV = r"\Vert\mathbf{u}\Vert", r"\Vert\mathbf{v}\Vert"

    b.heading("The inequality")
    b.row([rf"$|{UV}| \leq {NU}\,{NV}$"])
    b.note("The dot product is never bigger (in size) than the two lengths multiplied.")

    b.heading("Check it with your numbers")
    b.long(f"{UV} = ", [f"{P(x)}\\cdot{P(y)}" for x, y in zip(u, v)], last=f" = {F(d)}")
    b.row([rf"${NU}\,{NV} \approx {F(mu)}\cdot{F(mv)} \approx {F(mu * mv)}$"])
    b.compare(f"|{F(d)}| = {F(abs(d))}", r"\leq", F(mu * mv),
              abs(d) <= mu * mv * (1 + 1e-12) + 1e-12)

    b.heading("Why it is always true")
    b.row([rf"${UV} = {NU}\,{NV}\cos\theta$"])
    b.row([rf"$|{UV}| = {NU}\,{NV}\,|\cos\theta| \leq {NU}\,{NV}$"])
    b.note("because |cos θ| is never more than 1. In the picture: the shadow of v on u "
           "is never longer than v itself.")
    if mu and mv:
        ratio = abs(d) / (mu * mv)
        b.row([rf"$|\cos\theta| = \frac{{|{UV}|}}{{{NU}\,{NV}}} \approx {F(ratio)}"
               r" \leq 1$"])

    b.heading("When are the two sides equal?")
    b.note("Only when u and v are parallel (v = k·u, so θ = 0° or 180°), or one of them "
           "is the zero vector.", color=INK)
    if _equal(u, v):
        b.note("Your vectors are parallel, so here it is an equality.", color=INK)
    else:
        b.note(f"Your vectors are not parallel (θ ≈ {la.angle_between(u, v):.2f}°), so "
               "the left side is strictly smaller.", color=INK)

    b.heading("A consequence: the triangle inequality")
    s = la.magnitude(u + v)
    b.row([rf"$\Vert\mathbf{{u}}+\mathbf{{v}}\Vert \leq {NU} + {NV}$"])
    b.compare(F(s), r"\leq", f"{F(mu)} + {F(mv)} = {F(mu + mv)}",
              s <= (mu + mv) * (1 + 1e-12) + 1e-12)
    b.note("One side of a triangle is never longer than the other two together.")


def cauchy_code(u, v):
    return "\n".join([
        "import numpy as np",
        "",
        f"u = {np_array(u)}",
        f"v = {np_array(v)}",
        "",
        "# Cauchy–Schwarz:  |u·v| <= |u| |v|",
        "left = abs(np.dot(u, v))",
        "right = np.linalg.norm(u) * np.linalg.norm(v)",
        'print("|u·v| =", left, "  |u||v| =", round(right, 4))',
        'print("inequality holds:", left <= right + 1e-12)',
        "",
        "# Equal only when u and v are parallel",
        'print("parallel (equality):", np.isclose(left, right))',
        "",
        "# Consequence: triangle inequality  |u + v| <= |u| + |v|",
        'print(np.linalg.norm(u + v) <= np.linalg.norm(u) + np.linalg.norm(v))',
    ])
