"""
tab_complex.py
--------------
Complex numbers as vectors in the complex plane: z = a + bi is the arrow
(a, b). Adding is vector addition, multiplying rotates and scales, the
conjugate mirrors, and z̄w packs the dot product and the 2D cross product
into one complex number. Ends with vectors whose entries are complex.
"""

import math
import tkinter as tk
from tkinter import ttk

import numpy as np

import linalg_core as la
from algebra_render import nums
from geo_scene import arc_points, unit
from tab_base import GeoTab
from topic_util import F, P, close


def cfmt(z, digits=4):
    """(3, -2) -> '3 − 2i'"""
    re, im = float(z[0]), float(z[1])
    mag = F(abs(im), digits)
    imag = "i" if mag == "1" else f"{mag}i"          # 1i is written just i
    if mag == "0":
        return F(re, digits)
    if F(re, digits) == "0":
        return imag if im > 0 else f"−{imag}"
    sign = "−" if im < 0 else "+"
    return f"{F(re, digits)} {sign} {imag}"


def ctex(z, digits=4):
    """The same for mathtext."""
    return cfmt(z, digits).replace("−", "-")


def cmul(z, w):
    return np.array([z[0] * w[0] - z[1] * w[1], z[0] * w[1] + z[1] * w[0]])


def arg(z):
    return math.degrees(math.atan2(z[1], z[0]))


class ComplexTab(GeoTab):
    title = "  Complex Numbers  "

    def build_controls(self, side):
        self.z = self._complex_box(side, "z = a + bi", "U", ("3", "1"), pady=(0, 0))
        self.w = self._complex_box(side, "w = c + di", "V", ("1", "2"))
        self.show_sum, self.show_prod, self.show_conj, self.show_arg = self.add_checks(
            side, "Display", [("Sum z + w", True), ("Product z · w", True),
                              ("Conjugate z̄", False), ("Angles (arguments)", True)])
        self.add_actions(side)

    def _complex_box(self, side, title, role, defaults, pady=(12, 0)):
        box = self.add_box(side, title, role, pady=pady)
        row = ttk.Frame(box)
        row.pack(fill="x")
        variables = []
        for label, default in zip(("real =", "imaginary ="), defaults):
            var = tk.StringVar(value=default)
            ttk.Label(row, text=label).pack(side="left", padx=(0, 3))
            ttk.Entry(row, textvariable=var, width=6, justify="center").pack(
                side="left", padx=(0, 10))
            variables.append(var)
        self.watch(*variables)
        return variables

    def read_params(self):
        z, err = self.read_vector(self.z, "z", 2)
        w, err2 = self.read_vector(self.w, "w", 2)
        if err or err2:
            return None, err or err2
        return dict(dim=2, z=z, w=w, sum=self.show_sum.get(), prod=self.show_prod.get(),
                    conj=self.show_conj.get(), arg=self.show_arg.get()), None

    def extent(self, p):
        z, w = p["z"], p["w"]
        parts = [la.magnitude(z), la.magnitude(w), 1.0]
        if p["sum"]:
            parts.append(la.magnitude(z + w))
        if p["prod"]:
            parts.append(la.magnitude(cmul(z, w)))
        return max(parts)

    def legend(self, p):
        th = self.th
        return [("z", th["v"], "line"), ("w", th["kv"], "line"),
                ("z + w", th["w"], "line"), ("z · w", th["proj"], "line")]

    def on_scene(self, scene):
        scene.ax.set_xlabel("Re  (real part)")
        scene.ax.set_ylabel("Im  (imaginary part)")

    def geometry(self, s, p):
        z, w, c = p["z"], p["w"], s.col
        L = self.extent(p)
        t = np.linspace(0, 2 * math.pi, 120)
        s.path("unit", np.c_[np.cos(t), np.sin(t)], c["muted"], ls=":", lw=1, alpha=0.6)
        if p["sum"]:
            s.segment("wcopy", z, z + w, c["v"], ls="--", lw=1.1, alpha=0.6)
            s.segment("zcopy", w, z + w, c["u"], ls="--", lw=1.1, alpha=0.6)
            s.arrow("sum", z + w, c["w"], width=0.008, zorder=3)
            self._tip(s, "lsum", z + w, "z + w", c["w"], L)
        if p["prod"]:
            zw = cmul(z, w)
            s.arrow("prod", zw, c["p"], width=0.009, zorder=4)
            self._tip(s, "lprod", zw, "z·w", c["p"], L)
        if p["conj"]:
            zc = np.array([z[0], -z[1]])
            s.segment("mirror", z, zc, c["muted"], ls=":", lw=1.2)
            s.arrow("conj", zc, c["muted"], width=0.007, glow=False, zorder=3)
            self._tip(s, "lconj", zc, "z̄", c["muted"], L)
        if p["arg"]:
            items = [("z", z, c["u"], 0.22), ("w", w, c["v"], 0.32)]
            if p["prod"]:
                items.append(("zw", cmul(z, w), c["p"], 0.42))
            for name, vec, col, r in items:
                pts = arc_points(np.array([1.0, 0.0]), vec, r * L)
                s.path("arc" + name, pts, col, lw=1.3)
                if pts is not None:
                    s.label("targ" + name, pts[len(pts) // 2] * 1.18,
                            f"{arg(vec):.0f}°", col, size=8.5)
        s.arrow("w", w, c["v"], width=0.009, zorder=5)
        s.arrow("z", z, c["u"], width=0.011, zorder=6)
        self._tip(s, "lz", z, "z", c["u"], L)
        self._tip(s, "lw", w, "w", c["v"], L)
        s.title(f"z = {cfmt(z)},   w = {cfmt(w)}\n"
                f"z + w = {cfmt(z + w)}   ·   z·w = {cfmt(cmul(z, w))}")

    @staticmethod
    def _tip(s, key, vec, text, col, L):
        n = unit(vec)
        if n is not None:
            s.label(key, vec + n * L * 0.08, text, col, size=11)

    def status_text(self, p):
        z, w = p["z"], p["w"]
        return (f"|z| = {F(la.magnitude(z))}   arg z = {arg(z):.2f}°   —   "
                f"z·w = {cfmt(cmul(z, w))}   |z·w| = {F(la.magnitude(cmul(z, w)))}")

    def algebra(self, p):
        return complex_steps, (p["z"], p["w"])

    def python_code(self, p):
        return complex_code(p["z"], p["w"])


def complex_steps(b, z, w):
    pal = b.pal
    U, V, INK = pal["v"], pal["kv"], pal["ink"]
    z, w = la.to_vector(z), la.to_vector(w)
    a, bb = z
    c, d = w
    zw = cmul(z, w)
    rz, rw = la.magnitude(z), la.magnitude(w)

    b.heading("A complex number is a 2D vector")
    b.row([rf"$z = {ctex(z)} \;\leftrightarrow\;$", ("vec", nums(z), U),
           rf"$\qquad w = {ctex(w)} \;\leftrightarrow\;$", ("vec", nums(w), V)])
    b.note("Real part → x, imaginary part → y. This picture is called the complex plane; "
           "i is the number with i² = −1.")

    b.heading("Length and angle (polar form)")
    b.row([rf"$|z| = \sqrt{{{P(a)}^2 + {P(bb)}^2}} \approx {F(rz)}$"])
    if rz:
        b.row([rf"$\arg z = \mathrm{{atan2}}({F(bb)},\ {F(a)}) \approx {arg(z):.2f}^\circ$"])
        b.row([r"$z = |z|\,(\cos\theta + i\sin\theta) = |z|\,e^{i\theta}$"])
    b.note("|z| is just the length of the vector; the argument is its angle from the "
           "real axis.")

    b.heading("Adding = adding vectors")
    b.row([rf"$z + w = ({F(a)} + {P(c)}) + ({F(bb)} + {P(d)})\,i = {ctex(z + w)}$"])
    b.note("Exactly like u + v: add the real parts, add the imaginary parts.")

    b.heading("Multiplying = rotate and stretch")
    b.row([r"$zw = (ac - bd) + (ad + bc)\,i$"])
    b.row([rf"$= ({P(a)}\cdot{P(c)} - {P(bb)}\cdot{P(d)}) + "
           rf"({P(a)}\cdot{P(d)} + {P(bb)}\cdot{P(c)})\,i = {ctex(zw)}$"], indent=30)
    b.check(r"|zw| = |z|\,|w|", f"{F(rz)}\\cdot{F(rw)} \\approx {F(rz * rw)}",
            close(la.magnitude(zw), rz * rw))
    if rz and rw:
        s = (arg(z) + arg(w) + 180) % 360 - 180
        b.check(r"\arg(zw) = \arg z + \arg w", f"{arg(z):.2f} + {arg(w):.2f} \\approx "
                f"{s:.2f}^\\circ", close(round(arg(zw), 6) % 360, round(s, 6) % 360))
    b.note("Multiplying by w turns z by w's angle and stretches it by |w|. "
           "(This uses i·i = −1.)")

    b.heading("Conjugate: mirror in the real axis")
    b.row([rf"$\bar z = {ctex([a, -bb])}$"])
    b.check(r"z\,\bar z = |z|^2", f"{F(a * a + bb * bb)}", True)

    b.heading("Link to the dot and cross product")
    zcw = cmul([a, -bb], w)
    b.row([rf"$\bar z\,w = {ctex(zcw)}$"])
    b.check(r"\mathrm{Re}(\bar z\,w) = \mathbf{z}\cdot\mathbf{w} = ac + bd", F(zcw[0]),
            close(zcw[0], la.dot(z, w)))
    b.check(r"\mathrm{Im}(\bar z\,w) = \mathbf{z}\times\mathbf{w} = ad - bc", F(zcw[1]),
            close(zcw[1], la.cross(z, w)))
    b.note("One complex product holds both vector products at once.")

    b.heading("Vectors with complex entries")
    b.row([r"$\mathbf{x} = (z,\ w),\qquad \Vert\mathbf{x}\Vert^2 = \bar z z + \bar w w"
           rf" = {F(rz ** 2)} + {F(rw ** 2)} = {F(rz ** 2 + rw ** 2)}$"])
    b.note("For complex vectors the dot product conjugates the first vector "
           "(⟨x, y⟩ = Σ x̄ᵢ yᵢ). Without it, z·z = z² can be negative or complex "
           "and would not be a length.", color=INK)


def complex_code(z, w):
    return "\n".join([
        "import math",
        "import cmath",
        "import numpy as np",
        "",
        "# Python writes the imaginary unit as j",
        f"z = complex({F(z[0])}, {F(z[1])})",
        f"w = complex({F(w[0])}, {F(w[1])})",
        "",
        'print("z + w =", z + w)',
        'print("z * w =", z * w)',
        'print("|z| =", round(abs(z), 4), "  arg z =", round(math.degrees(cmath.phase(z)), 2))',
        'print("|zw| == |z||w| :", math.isclose(abs(z * w), abs(z) * abs(w)))',
        'print("conjugate      :", z.conjugate(), "  z * conj(z) =", z * z.conjugate())',
        "",
        "# The same numbers as 2D vectors",
        "zv = np.array([z.real, z.imag])",
        "wv = np.array([w.real, w.imag])",
        "zcw = z.conjugate() * w",
        'print("Re(conj(z) w) =", zcw.real, "= z·w =", np.dot(zv, wv))',
        'print("Im(conj(z) w) =", zcw.imag, "= z×w =", zv[0] * wv[1] - zv[1] * wv[0])',
        "",
        "# A vector with complex entries: its length needs the conjugate",
        "x = np.array([z, w])",
        'print("|x|^2 =", np.vdot(x, x).real)   # np.vdot conjugates the first vector',
    ])
