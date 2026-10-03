"""
tab_cosines.py
--------------
Law of cosines  c² = a² + b² − 2ab·cos C  for the triangle made by two
vectors u and v (or by two sides and the angle between them), with its
proof from the dot product and the triangle's other two angles.

Triangle labels: the angle C sits at the origin between the sides
a = ‖u‖ and b = ‖v‖; vertex B is the tip of u, vertex A the tip of v, and
the third side c = ‖u − v‖ runs from A to B.
"""

import math
import tkinter as tk
from tkinter import ttk

import linalg_core as la
from algebra_render import nums
from geo_scene import arc_points, right_angle_points, unit
from tab_base import GeoTab
from topic_util import F, P, close, np_array


def triangle(u, v):
    """Sides a, b, c and angles A, B, C (degrees) of the triangle O, u, v."""
    a, b, c = la.magnitude(u), la.magnitude(v), la.distance(u, v)
    C = la.angle_between(u, v)
    A = la.angle_between(-v, u - v)          # at the tip of v
    B = la.angle_between(-u, v - u)          # at the tip of u
    return a, b, c, A, B, C


class LawOfCosinesTab(GeoTab):
    title = "  Law of Cosines  "

    def build_controls(self, side):
        self.mode = tk.StringVar(value="vectors")
        box = self.add_box(side, "Build the triangle from", pady=(0, 0))
        for text, value in (("two vectors u and v", "vectors"),
                            ("two sides and the angle between", "sides")):
            ttk.Radiobutton(box, text=text, value=value, variable=self.mode,
                            command=self._on_mode).pack(anchor="w")
        self.add_dim_selector(side, pady=(10, 0))
        self.u = self.add_vector_box(side, "Vector u  (side a)", "U", ("4", "0", "1"))
        self.v = self.add_vector_box(side, "Vector v  (side b)", "V", ("1", "3", "2"))
        box = self.add_box(side, "Sides and angle")
        self.a, ea = self.add_entry_row(box, "a =", "4", width=6)
        self.b, eb = self.add_entry_row(box, "b =", "3", width=6, pady=(4, 0))
        self.C, ec = self.add_entry_row(box, "angle C (°) =", "60", width=6, pady=(4, 0))
        self._side_entries = [ea, eb, ec]
        self.show_angles, = self.add_checks(side, "Display", [("All three angles", True)])
        self.add_actions(side)
        self._on_mode()

    def _on_mode(self):
        sides = self.mode.get() == "sides"
        for e in self._side_entries:
            e.configure(state="normal" if sides else "disabled")
        for vars_ in (self.u, self.v):
            for i, e in enumerate(self.entries[id(vars_)]):
                on = not sides and (i < 2 or self.ndim() == 3)
                e.configure(state="normal" if on else "disabled")
        self.refresh()

    def _on_dim(self):
        super()._on_dim()
        self._on_mode()

    def read_params(self):
        mode = self.mode.get()
        if mode == "sides":
            a, e1 = self.read_number(self.a, "a")
            b, e2 = self.read_number(self.b, "b")
            C, e3 = self.read_number(self.C, "angle C")
            err = e1 or e2 or e3
            if err:
                return None, err
            if a <= 0 or b <= 0:
                return None, "Sides a and b must be greater than 0."
            if not 0 < C < 180:
                return None, "Angle C must be between 0° and 180°."
            r = math.radians(C)
            u = la.to_vector([a, 0.0])
            v = la.to_vector([b * math.cos(r), b * math.sin(r)])
            dim = 2
        else:
            u, e1 = self.read_vector(self.u, "u")
            v, e2 = self.read_vector(self.v, "v")
            if e1 or e2:
                return None, e1 or e2
            dim = self.ndim()
        return dict(mode=mode, dim=dim, u=u, v=v, angles=self.show_angles.get()), None

    def extent(self, p):
        return max(la.magnitude(p["u"]), la.magnitude(p["v"]), 1.0)

    def legend(self, p):
        th = self.th
        return [("a = ‖u‖", th["v"], "line"), ("b = ‖v‖", th["kv"], "line"),
                ("c = ‖u − v‖", th["w"], "line")]

    def geometry(self, s, p):
        u, v, c = p["u"], p["v"], s.col
        L = self.extent(p)
        a, b, cc, A, B, C = triangle(u, v)
        O = u * 0
        centroid = (u + v) / 3
        s.polygon("tri", [O, u, v], c["w"], alpha=0.10)
        s.arrow("c", u - v, c["w"], origin=v, width=0.008, zorder=3)
        s.arrow("b", v, c["v"], width=0.009, zorder=4)
        s.arrow("a", u, c["u"], width=0.009, zorder=4)

        def out(mid):                      # push a label away from the triangle
            d = unit(mid - centroid)
            return mid + (d if d is not None else O) * L * 0.13

        if a > 0:
            s.label("la", out(u / 2), f"a = {F(a, 2)}", c["u"], size=10)
        if b > 0:
            s.label("lb", out(v / 2), f"b = {F(b, 2)}", c["v"], size=10)
        if cc > 0:
            s.label("lc", out((u + v) / 2), f"c = {F(cc, 2)}", c["w"], size=10)

        r = 0.22 * min(x for x in (a, b, cc, L) if x > 0) if (a and b) else 0
        corners = [("C", O, u, v, C, c["text"]), ("B", u, -u, v - u, B, c["muted"]),
                   ("A", v, -v, u - v, A, c["muted"])]
        for name, at, d1, d2, ang, col in corners:
            if ang is None or r == 0:
                continue
            if name != "C" and not p["angles"]:
                continue
            if close(ang, 90):
                s.path("arc" + name, right_angle_points(at, d1, d2, r * 0.55), col, lw=1.2)
            else:
                pts = arc_points(d1, d2, r)
                s.path("arc" + name, None if pts is None else pts + at, col, lw=1.2)
            dirn = unit((unit(d1) if unit(d1) is not None else O) +
                        (unit(d2) if unit(d2) is not None else O))
            # just inside the arc, along the bisector of the corner
            spot = at + (dirn if dirn is not None else O) * (r + L * 0.09)
            s.label("t" + name, spot, f"{ang:.1f}°", col, size=8.5)
        for name, at in (("C", O), ("B", u), ("A", v)):
            d = unit(at - centroid)
            s.label("v" + name, at + (d if d is not None else O) * L * 0.07, name,
                    c["text"], size=11)
        s.title("c² = a² + b² − 2ab·cos C\n"
                f"{F(cc ** 2, 3)} = {F(a * a, 3)} + {F(b * b, 3)} − "
                f"{F(2 * a * b * math.cos(math.radians(C or 0)), 3)}")

    def status_text(self, p):
        a, b, cc, A, B, C = triangle(p["u"], p["v"])
        angles = "  ".join(f"{n} = {x:.2f}°" for n, x in (("A", A), ("B", B), ("C", C))
                           if x is not None)
        return f"a = {F(a)}   b = {F(b)}   c = {F(cc)}   —   {angles}"

    def algebra(self, p):
        return cosine_steps, (p["u"], p["v"], p["mode"])

    def python_code(self, p):
        return cosine_code(p["u"], p["v"], p["mode"])


def cosine_steps(b, u, v, mode):
    pal = b.pal
    W, INK = pal["w"], pal["ink"]
    u, v = la.to_vector(u), la.to_vector(v)
    a, bb, c, A, B, C = triangle(u, v)

    b.heading("The triangle")
    if mode == "sides":
        b.note(f"Given two sides and the angle between them: a = {F(a)}, b = {F(bb)}, "
               f"C = {F(C, 2)}°.", color=INK)
    else:
        b.note("Sides a and b are the vectors u and v; C is the angle between them.",
               color=INK)
        b.row([rf"$a = \Vert\mathbf{{u}}\Vert \approx {F(a)},\quad b = \Vert\mathbf{{v}}\Vert"
               rf" \approx {F(bb)}$"])
        if C is not None:
            b.row([rf"$\cos C = \frac{{\mathbf{{u}}\cdot\mathbf{{v}}}}{{ab}} = "
                   rf"\frac{{{F(la.dot(u, v))}}}{{{F(a)}\cdot{F(bb)}}}"
                   rf" \Rightarrow C \approx {C:.2f}^\circ$"])
    if C is None or a == 0 or bb == 0:
        b.note("With a zero vector there is no triangle (and no angle C).", color=INK)
        return
    cosC = math.cos(math.radians(C))

    b.heading("Law of cosines: find the third side c")
    b.row([r"$c^2 = a^2 + b^2 - 2ab\cos C$"])
    b.row([rf"$= {P(a)}^2 + {P(bb)}^2 - 2\cdot{F(a)}\cdot{F(bb)}\cdot\cos({C:.2f}^\circ)$"],
          indent=30)
    b.row([rf"$= {F(a * a)} + {F(bb * bb)} - {P(2 * a * bb * cosC)} = {F(c * c)}$"], indent=30)
    b.row([rf"$c = \sqrt{{{F(c * c)}}} \approx {F(c)}$"])

    b.heading("Check: c is the length of u − v")
    b.row([r"$\mathbf{u}-\mathbf{v} =$", ("vec", nums(u - v), W)])
    b.check(r"\Vert\mathbf{u}-\mathbf{v}\Vert",
            r"\sqrt{" + " + ".join(f"{P(x)}^2" for x in u - v) + rf"}} \approx {F(c)}",
            close(la.distance(u, v), c))

    b.heading("Why it works: the dot product")
    b.row([r"$\Vert\mathbf{u}-\mathbf{v}\Vert^2 = (\mathbf{u}-\mathbf{v})\cdot"
           r"(\mathbf{u}-\mathbf{v})$"])
    b.row([r"$= \mathbf{u}\cdot\mathbf{u} - 2\,\mathbf{u}\cdot\mathbf{v} + "
           r"\mathbf{v}\cdot\mathbf{v}$"], indent=40)
    b.row([r"$= a^2 - 2ab\cos C + b^2$"], indent=40)
    b.note("using u·u = a², v·v = b² and u·v = ab·cos C.")

    b.heading("The other two angles")
    if c == 0:
        b.note("c = 0: the triangle has collapsed, so A and B are undefined.", color=INK)
    else:
        b.row([r"$\cos A = \frac{b^2 + c^2 - a^2}{2bc}"
               rf" = {F((bb * bb + c * c - a * a) / (2 * bb * c))}"
               rf" \Rightarrow A \approx {A:.2f}^\circ$"])
        b.row([r"$\cos B = \frac{a^2 + c^2 - b^2}{2ac}"
               rf" = {F((a * a + c * c - bb * bb) / (2 * a * c))}"
               rf" \Rightarrow B \approx {B:.2f}^\circ$"])
        b.check(r"A + B + C", f"{A:.2f} + {B:.2f} + {C:.2f} = {A + B + C:.2f}^\\circ",
                close(round(A + B + C, 6), 180))

    b.heading("Special cases")
    if close(C, 90):
        first = "C = 90°  →  c² = a² + b²  (Pythagoras!)"
    elif C < 90:
        first = "C < 90°  →  c² < a² + b²  (acute at C)"
    else:
        first = "C > 90°  →  c² > a² + b²  (obtuse at C)"
    lines = [first, "C = 90°  →  cos C = 0: Pythagoras",
             "C → 0° or 180°  →  a flat triangle"]
    b.result_box(lines, bold=1, color=pal["accent"])


def cosine_code(u, v, mode):
    return "\n".join([
        "import math",
        "import numpy as np",
        "",
        "# ── Inputs ───────────────────────────────",
        f"u = {np_array(u)}   # side a" + ("  (a along the x axis)" if mode == "sides" else ""),
        f"v = {np_array(v)}   # side b",
        "",
        "# ── The triangle: two sides and the angle ─",
        "a = np.linalg.norm(u)",
        "b = np.linalg.norm(v)",
        "C = math.degrees(math.acos(np.clip(u @ v / (a * b), -1, 1)))",
        'print("a =", round(a, 4), " b =", round(b, 4), " C =", round(C, 2), "degrees")',
        "",
        "# ── Law of cosines ───────────────────────",
        "c2 = a**2 + b**2 - 2 * a * b * math.cos(math.radians(C))",
        "c = math.sqrt(c2)",
        'print("c^2 =", round(c2, 4), "  c =", round(c, 4))',
        "",
        "# ── Check against the vectors ────────────",
        'print("||u - v|| =", round(np.linalg.norm(u - v), 4),',
        '      " same:", math.isclose(c, np.linalg.norm(u - v), rel_tol=1e-9))',
        "",
        "# ── The other angles ─────────────────────",
        "# (np.clip guards against tiny rounding errors outside -1..1)",
        "if c > 0:",
        "    A = math.degrees(math.acos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1, 1)))",
        "    B = math.degrees(math.acos(np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1, 1)))",
        '    print("A =", round(A, 2), " B =", round(B, 2), " A+B+C =", round(A + B + C, 2))',
    ])
