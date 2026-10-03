"""
tab_dot.py
----------
Dot product u·v: the component formula, the angle hidden inside it, the
projection of one vector on the other, and the properties of the dot
product checked with the user's own numbers.
"""

import numpy as np

import linalg_core as la
from algebra_render import nums
from geo_scene import arc_points, right_angle_points, unit
from tab_base import GeoTab
from topic_util import F, P, close, deg, np_array


class DotProductTab(GeoTab):
    title = "  Dot Product  "

    def build_controls(self, side):
        self.add_dim_selector(side)
        self.u = self.add_vector_box(side, "Vector u", "U", ("3", "1", "2"))
        self.v = self.add_vector_box(side, "Vector v", "V", ("1", "2", "-1"))
        box = self.add_box(side, "Property checks use", "W")
        self.w = self.add_xyz_row(box, ("-1", "1", "0"), label="w:")
        self.k, _ = self.add_entry_row(box, "scalar k =", "2", pady=(6, 0))
        self.show_proj, self.show_angle = self.add_checks(
            side, "Display", [("Projection of v onto u", True), ("Angle θ between u and v", True)])
        self.add_actions(side)

    # ---- values
    def read_params(self):
        u, err = self.read_vector(self.u, "u")
        v, err2 = self.read_vector(self.v, "v")
        w, err3 = self.read_vector(self.w, "w")
        k, err4 = self.read_number(self.k, "k")
        err = err or err2 or err3 or err4
        if err:
            return None, err
        return dict(dim=self.ndim(), u=u, v=v, w=w, k=k, proj=self.show_proj.get(),
                    angle=self.show_angle.get()), None

    def extent(self, p):
        return max(la.magnitude(p["u"]), la.magnitude(p["v"]), 1.0)

    def legend(self, p):
        th = self.th
        return [("u", th["v"], "line"), ("v", th["kv"], "line"),
                ("proj of v on u", th["proj"], "line")]

    # ---- picture
    def geometry(self, s, p):
        u, v, c = p["u"], p["v"], s.col
        L = self.extent(p)
        d = la.dot(u, v)
        theta = la.angle_between(u, v)

        if p["proj"]:
            pr = la.vector_projection(v, u)
            if pr is not None:
                s.segment("drop", v, pr, c["muted"], ls=":", lw=1.4)
                s.arrow("proj", pr, c["p"], width=0.007, glow=False, zorder=3)
                if np.any(pr) and np.linalg.norm(v - pr) > 1e-9 * L:
                    s.path("drop_sq", right_angle_points(pr, -pr, v - pr, L * 0.045),
                           c["muted"], lw=1)
        if p["angle"] and theta is not None:
            r = 0.3 * min(la.magnitude(u), la.magnitude(v))
            if close(theta, 90):
                pts = right_angle_points(np.zeros_like(u), u, v, r * 0.6)
                s.path("arc", pts, c["text"], lw=1.3)
                mid = (unit(u) + unit(v)) * r * 1.1
            else:
                s.path("arc", arc_points(u, v, r), c["text"], lw=1.3)
                mid_arc = arc_points(u, v, r * 1.75)
                mid = mid_arc[len(mid_arc) // 2] if mid_arc is not None else u * 0
            s.label("theta", mid, f"θ = {theta:.1f}°", c["text"], size=9)

        s.arrow("v", v, c["v"], width=0.009, zorder=4)
        s.arrow("u", u, c["u"], width=0.011, zorder=5)
        for key, vec, col in (("lu", u, c["u"]), ("lv", v, c["v"])):
            n = unit(vec)
            if n is not None:
                s.label(key, vec + n * L * 0.09, key[1], col, size=12)
        kind = la.angle_kind(theta)
        s.title(f"u·v = {F(d)}\nθ = {deg(theta)}  ·  {kind}")

    def status_text(self, p):
        theta = la.angle_between(p["u"], p["v"])
        return (f"u·v = {F(la.dot(p['u'], p['v']))}   —   θ = {deg(theta)}"
                f"  ({la.angle_kind(theta)})")

    # ---- explanation
    def algebra(self, p):
        return dot_steps, (p["u"], p["v"], p["w"], p["k"])

    def python_code(self, p):
        return dot_code(p["u"], p["v"], p["w"], p["k"])


def dot_steps(b, u, v, w, k):
    pal = b.pal
    U, V, INK = pal["v"], pal["kv"], pal["ink"]
    u, v, w = la.to_vector(u), la.to_vector(v), la.to_vector(w)
    n = len(u)
    d = la.dot(u, v)
    mu, mv = la.magnitude(u), la.magnitude(v)
    UV = r"\mathbf{u}\cdot\mathbf{v}"

    b.heading("The definition")
    b.row([rf"${UV} = \sum_{{i=1}}^{{{n}}} u_i\,v_i$"])
    b.row(["$= " + " + ".join(f"u_{i}v_{i}" for i in range(1, n + 1)) + "$"], indent=40)
    b.note("Multiply each pair of matching components, then add the results. "
           "The answer is a single number (a scalar), not a vector.")

    b.heading("With your numbers")
    b.row([f"${UV} =$", ("vec", nums(u), U), r"$\cdot$", ("vec", nums(v), V)])
    b.long("= ", [f"{P(a)}\\cdot{P(c)}" for a, c in zip(u, v)])
    b.long("= ", [P(a * c) for a, c in zip(u, v)], last=f" = {F(d)}")

    b.heading("What the number means: the angle")
    b.row([rf"${UV} = \Vert\mathbf{{u}}\Vert\,\Vert\mathbf{{v}}\Vert\,\cos\theta$"])
    if mu == 0 or mv == 0:
        b.note("One of the vectors is zero, so there is no angle.", color=INK)
    else:
        cos_t, theta = la.cos_angle(u, v), la.angle_between(u, v)
        b.row([rf"$\cos\theta = \frac{{{UV}}}{{\Vert\mathbf{{u}}\Vert\,\Vert\mathbf{{v}}\Vert}}"
               rf" = \frac{{{F(d)}}}{{{F(mu)}\cdot{F(mv)}}} \approx {F(cos_t)}$"])
        b.row([rf"$\theta \approx {theta:.2f}^\circ$"], indent=40)
        tol = 1e-9 * mu * mv
        if d > tol:
            meaning = "u·v > 0, so the angle is less than 90° (they point roughly the same way)."
        elif d < -tol:
            meaning = "u·v < 0, so the angle is more than 90° (they point roughly opposite ways)."
        else:
            meaning = "u·v = 0, so u and v are perpendicular (90°)."
        b.note(meaning, color=INK)

    b.heading("Projection: the shadow of v on u")
    if mu == 0:
        b.note("u is the zero vector, so there is nothing to project onto.", color=INK)
    else:
        b.row([rf"$\mathrm{{proj}}_{{\mathbf{{u}}}}\,\mathbf{{v}} = "
               rf"\frac{{{UV}}}{{\Vert\mathbf{{u}}\Vert^2}}\,\mathbf{{u}} =$",
               ("vec", nums(la.vector_projection(v, u)), pal["proj"])])
        b.note("The orange arrow in the plot.")

    b.heading("Properties (checked)")
    lhs = la.dot(u, v + w)
    b.check(r"\mathbf{u}\cdot\mathbf{v} = \mathbf{v}\cdot\mathbf{u}", F(d),
            close(d, la.dot(v, u)))
    b.check(r"\mathbf{u}\cdot(\mathbf{v}+\mathbf{w}) = \mathbf{u}\cdot\mathbf{v}"
            r" + \mathbf{u}\cdot\mathbf{w}", F(lhs), close(lhs, d + la.dot(u, w)))
    b.check(r"(k\mathbf{u})\cdot\mathbf{v} = k\,(\mathbf{u}\cdot\mathbf{v})",
            F(la.dot(k * u, v)), close(la.dot(k * u, v), k * d))
    b.check(r"\mathbf{v}\cdot\mathbf{v} = \Vert\mathbf{v}\Vert^2", F(la.dot(v, v)),
            close(la.dot(v, v), mv ** 2))
    b.note(f"Using w = {la.fmt_vec(w)} and k = {F(k)} (change them on the left).")


def dot_code(u, v, w, k):
    return "\n".join([
        "import numpy as np",
        "",
        f"u = {np_array(u)}",
        f"v = {np_array(v)}",
        "",
        "# Dot product: multiply matching components, then add",
        "# (the Σ formula). NumPy does it in one call:",
        "dot = np.dot(u, v)          # or: u @ v",
        'print("u·v =", dot)',
        "",
        "# The same sum, written out step by step:",
        'print("check:", sum(ui * vi for ui, vi in zip(u, v)))',
        "",
        "# The angle:  cos θ = u·v / (|u| |v|)",
        "len_u = np.linalg.norm(u)",
        "len_v = np.linalg.norm(v)",
        "if len_u > 0 and len_v > 0:",
        "    cos_t = np.clip(dot / (len_u * len_v), -1, 1)  # guard rounding",
        "    theta = np.degrees(np.arccos(cos_t))",
        '    print("θ =", round(theta, 2), "degrees")',
        "",
        "    # projection of v onto u (its shadow on u)",
        '    print("proj_u v =", dot / len_u**2 * u)',
        "",
        "# Properties (each prints True)",
        f"w = {np_array(w)}",
        f"k = {F(k)}",
        'print(np.isclose(u @ v, v @ u))                  # u·v = v·u',
        'print(np.isclose(u @ (v + w), u @ v + u @ w))    # u·(v+w) = u·v + u·w',
        'print(np.isclose((k * u) @ v, k * (u @ v)))      # (ku)·v = k(u·v)',
        'print(np.isclose(v @ v, len_v**2))               # v·v = |v|²',
    ])
