"""
tab_two.py
----------
Everything you can compute between two vectors u and v: sum, difference,
distance, angle, projections, cross product (area) and how they relate.
"""

import numpy as np

import linalg_core as la
from algebra_render import nums
from geo_scene import arc_points, right_angle_points, unit
from tab_base import GeoTab
from themes import mix
from topic_util import F, P, close, deg, np_array, sqrt_tex


class TwoVectorsTab(GeoTab):
    title = "  Between Two Vectors  "

    def build_controls(self, side):
        self.add_dim_selector(side)
        self.u = self.add_vector_box(side, "Vector u", "U", ("4", "1", "1"))
        self.v = self.add_vector_box(side, "Vector v", "V", ("1", "3", "2"))
        self.show_sum, self.show_diff, self.show_angle, self.show_cross = self.add_checks(
            side, "Display", [("Sum u + v (parallelogram)", True), ("Difference u − v", True),
                              ("Angle θ", True), ("Cross product u × v (3D)", True)])
        self.add_actions(side)

    def read_params(self):
        u, err = self.read_vector(self.u, "u")
        v, err2 = self.read_vector(self.v, "v")
        if err or err2:
            return None, err or err2
        return dict(dim=self.ndim(), u=u, v=v, sum=self.show_sum.get(),
                    diff=self.show_diff.get(), angle=self.show_angle.get(),
                    cross=self.show_cross.get()), None

    def extent(self, p):
        u, v = p["u"], p["v"]
        parts = [la.magnitude(u), la.magnitude(v), 1.0]
        if p["sum"]:
            parts.append(la.magnitude(u + v))
        if p["cross"] and len(u) == 3:
            parts.append(la.magnitude(la.cross(u, v)))
        return max(parts)

    def legend(self, p):
        th = self.th
        items = [("u", th["v"], "line"), ("v", th["kv"], "line"),
                 ("u + v", th["w"], "line"), ("u − v", th["proj"], "line")]
        if p["dim"] == 3:
            items.append(("u × v", mix(th["v"], th["kv"], 0.5), "line"))
        return items

    def geometry(self, s, p):
        u, v, c = p["u"], p["v"], s.col
        L = self.extent(p)
        theta = la.angle_between(u, v)
        if p["sum"]:
            s.polygon("para", [u * 0, u, u + v, v], c["w"], alpha=0.10)
            s.segment("vcopy", u, u + v, c["v"], ls="--", lw=1.1, alpha=0.7)
            s.segment("ucopy", v, u + v, c["u"], ls="--", lw=1.1, alpha=0.7)
            s.arrow("sum", u + v, c["w"], width=0.008, zorder=3)
            n = unit(u + v)
            if n is not None:
                s.label("lsum", (u + v) + n * L * 0.08, "u + v", c["w"], size=10)
        if p["diff"]:
            s.arrow("diff", u - v, c["p"], origin=v, width=0.007, zorder=3)
            if np.any(u - v):
                s.label("ldiff", (u + v) / 2 + _away(u, v) * L * 0.07, "u − v", c["p"], size=10)
        if p["angle"] and theta is not None:
            r = 0.28 * min(la.magnitude(u), la.magnitude(v))
            if close(theta, 90):
                s.path("arc", right_angle_points(u * 0, u, v, r * 0.6), c["text"], lw=1.3)
                mid = (unit(u) + unit(v)) * r
            else:
                s.path("arc", arc_points(u, v, r), c["text"], lw=1.3)
                pts = arc_points(u, v, r * 1.7)
                mid = pts[len(pts) // 2] if pts is not None else u * 0
            s.label("theta", mid, f"θ = {theta:.1f}°", c["text"], size=9)
        if p["cross"] and len(u) == 3:
            x = la.cross(u, v)
            s.arrow("cross", x, c["x"], width=0.009, zorder=4)
            if np.any(x):
                s.label("lx", x * 1.08, "u × v", c["x"], size=10)
        s.arrow("v", v, c["v"], width=0.009, zorder=5)
        s.arrow("u", u, c["u"], width=0.011, zorder=6)
        for key, vec, col in (("lu", u, c["u"]), ("lv", v, c["v"])):
            n = unit(vec)
            if n is not None:
                s.label(key, vec + n * L * 0.08, key[1], col, size=12)
        s.title(f"distance ‖u − v‖ = {F(la.distance(u, v))}   ·   θ = {deg(theta)}\n"
                f"{la.relation(u, v)}")

    def status_text(self, p):
        u, v = p["u"], p["v"]
        return (f"u + v = {la.fmt_vec(u + v)}   u − v = {la.fmt_vec(u - v)}   "
                f"‖u − v‖ = {F(la.distance(u, v))}   θ = {deg(la.angle_between(u, v))}")

    def algebra(self, p):
        return two_steps, (p["u"], p["v"])

    def python_code(self, p):
        return two_code(p["u"], p["v"])


def _away(u, v):
    """Direction pointing away from the origin, off the segment v→u (for its label)."""
    mid = (u + v) / 2
    d = unit(u - v)
    if d is None:
        return u * 0
    if len(u) == 2:
        n = np.array([-d[1], d[0]])
        return n if np.dot(n, mid) >= 0 else -n
    m = unit(mid)
    return m if m is not None else u * 0


def two_steps(b, u, v):
    pal = b.pal
    U, V, W, PR, INK = pal["v"], pal["kv"], pal["w"], pal["proj"], pal["ink"]
    u, v = la.to_vector(u), la.to_vector(v)
    n = len(u)
    d = la.dot(u, v)
    mu, mv = la.magnitude(u), la.magnitude(v)

    b.heading("Add and subtract, component by component")
    b.row([r"$\mathbf{u}+\mathbf{v} =$", ("vec", nums(u), U), "$+$", ("vec", nums(v), V),
           "$=$", ("vec", nums(u + v), W)])
    b.row([r"$\mathbf{u}-\mathbf{v} =$", ("vec", nums(u), U), "$-$", ("vec", nums(v), V),
           "$=$", ("vec", nums(u - v), PR)])
    b.note("u + v is the diagonal of the parallelogram; u − v runs from the tip of v "
           "to the tip of u.")

    b.heading("Distance between the tips")
    dist = la.distance(u, v)
    b.row([rf"$\Vert\mathbf{{u}}-\mathbf{{v}}\Vert = {sqrt_tex(u - v)} \approx {F(dist)}$"])

    b.heading("Dot product and the angle")
    b.long(r"\mathbf{u}\cdot\mathbf{v} = ", [f"{P(a)}\\cdot{P(c)}" for a, c in zip(u, v)],
           last=f" = {F(d)}")
    if mu == 0 or mv == 0:
        b.note("One vector is zero, so the angle is undefined.", color=INK)
    else:
        cos_t, theta = la.cos_angle(u, v), la.angle_between(u, v)
        b.row([rf"$\cos\theta = \frac{{{F(d)}}}{{{F(mu)}\cdot{F(mv)}}} \approx {F(cos_t)}"
               rf"\quad\Rightarrow\quad \theta \approx {theta:.2f}^\circ$"])

    b.heading("Projections (shadows on each other)")
    if mu and mv:
        b.row([r"$\mathrm{proj}_{\mathbf{u}}\,\mathbf{v} = \frac{\mathbf{u}\cdot\mathbf{v}}"
               r"{\Vert\mathbf{u}\Vert^2}\mathbf{u} =$",
               ("vec", nums(la.vector_projection(v, u)), PR)])
        b.row([r"$\mathrm{proj}_{\mathbf{v}}\,\mathbf{u} = \frac{\mathbf{u}\cdot\mathbf{v}}"
               r"{\Vert\mathbf{v}\Vert^2}\mathbf{v} =$",
               ("vec", nums(la.vector_projection(u, v)), PR)])
        b.note(f"Signed shadow lengths: comp_u v = {F(d / mu)},  comp_v u = {F(d / mv)}")
    else:
        b.note("Projections need two non-zero vectors.", color=INK)

    if n == 3:
        x = la.cross(u, v)
        b.heading("Cross product u × v (3D only)")
        b.row([r"$\mathbf{u}\times\mathbf{v} =$", ("vec", [
            r"$u_y v_z - u_z v_y$", r"$u_z v_x - u_x v_z$", r"$u_x v_y - u_y v_x$"], INK),
               "$=$", ("vec", [
                   rf"${P(u[1])}\cdot{P(v[2])} - {P(u[2])}\cdot{P(v[1])}$",
                   rf"${P(u[2])}\cdot{P(v[0])} - {P(u[0])}\cdot{P(v[2])}$",
                   rf"${P(u[0])}\cdot{P(v[1])} - {P(u[1])}\cdot{P(v[0])}$"], INK)])
        b.row(["$=$", ("vec", nums(x), pal["accent"])], indent=40)
        b.row([rf"$\mathrm{{area}} = \Vert\mathbf{{u}}\times\mathbf{{v}}\Vert \approx "
               rf"{F(la.magnitude(x))}$"])
        b.note("u × v is perpendicular to both u and v; its length is the area of the "
               "parallelogram they span.")
        b.check(r"(\mathbf{u}\times\mathbf{v})\cdot\mathbf{u}", F(la.dot(x, u)),
                close(la.dot(x, u), 0))
        b.check(r"(\mathbf{u}\times\mathbf{v})\cdot\mathbf{v}", F(la.dot(x, v)),
                close(la.dot(x, v), 0))
    else:
        a = la.cross(u, v)
        b.heading("Signed area (the 2D cross product)")
        b.row([rf"$u_x v_y - u_y v_x = {P(u[0])}\cdot{P(v[1])} - {P(u[1])}\cdot{P(v[0])}"
               rf" = {F(a)}$"])
        turn = ("v is counter-clockwise from u" if a > 0 else
                "v is clockwise from u" if a < 0 else "u and v lie on one line")
        b.note(f"Area of the parallelogram = |{F(a)}| = {F(abs(a))};  {turn}.")

    b.heading("How are u and v related?")
    rel = la.relation(u, v)
    lines = [rel[0].upper() + rel[1:],
             "u·v = 0  →  orthogonal (perpendicular)",
             "u = k·v  →  parallel (θ = 0° or 180°)"]
    b.result_box(lines, bold=1, color=pal["accent"])


def two_code(u, v):
    lines = [
        "import numpy as np",
        "",
        "# ── Inputs ───────────────────────────────",
        f"u = {np_array(u)}",
        f"v = {np_array(v)}",
        "",
        "# ── Sum, difference, distance ────────────",
        'print("u + v =", u + v)',
        'print("u - v =", u - v)',
        'print("distance ||u - v|| =", round(np.linalg.norm(u - v), 4))',
        "",
        "# ── Dot product and angle ────────────────",
        "dot = u @ v",
        "len_u, len_v = np.linalg.norm(u), np.linalg.norm(v)",
        'print("u·v =", dot)',
        "if len_u and len_v:",
        "    theta = np.degrees(np.arccos(np.clip(dot / (len_u * len_v), -1, 1)))",
        '    print("angle θ =", round(theta, 2), "degrees")',
        '    print("proj_u v =", np.round(dot / len_u**2 * u, 4))',
        '    print("proj_v u =", np.round(dot / len_v**2 * v, 4))',
        "",
    ]
    if len(u) == 3:
        lines += [
            "# ── Cross product (3D) ───────────────────",
            "x = np.cross(u, v)",
            'print("u × v =", x)',
            'print("area of parallelogram =", round(np.linalg.norm(x), 4))',
            'print("perpendicular to u and v:", np.isclose(x @ u, 0), np.isclose(x @ v, 0))',
        ]
    else:
        lines += [
            "# ── Signed area (2D cross product) ───────",
            "area = u[0] * v[1] - u[1] * v[0]",
            'print("signed area =", area)',
        ]
    lines += [
        "",
        "# ── Relationship ─────────────────────────",
        "if len_u == 0 or len_v == 0:",
        '    print("a zero vector is involved")',
        "elif np.isclose(dot, 0):",
        '    print("orthogonal")',
        "elif np.isclose(abs(dot), len_u * len_v):",
        '    print("parallel")',
        "else:",
        '    print("neither parallel nor orthogonal")',
    ]
    return "\n".join(lines)
