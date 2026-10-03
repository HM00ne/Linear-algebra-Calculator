"""
tab_hadamard.py
---------------
Hadamard (element-wise) product u ⊙ v: multiply matching components and keep
them as a vector. Geometrically it stretches each axis separately, which is
the same as multiplying by the diagonal matrix diag(u).
"""

import numpy as np

import linalg_core as la
from algebra_render import nums
from geo_scene import unit
from tab_base import GeoTab
from topic_util import F, P, close, np_array


class HadamardTab(GeoTab):
    title = "  Hadamard Product  "

    def build_controls(self, side):
        self.add_dim_selector(side)
        self.u = self.add_vector_box(side, "Vector u  (the stretch factors)", "U",
                                     ("2", "0.5", "1.5"))
        self.v = self.add_vector_box(side, "Vector v", "V", ("1.5", "3", "2"))
        self.show_boxes, = self.add_checks(side, "Display",
                                           [("Component boxes (2D)", True)])
        self.add_actions(side)

    def read_params(self):
        u, err = self.read_vector(self.u, "u")
        v, err2 = self.read_vector(self.v, "v")
        if err or err2:
            return None, err or err2
        return dict(dim=self.ndim(), u=u, v=v, boxes=self.show_boxes.get()), None

    def extent(self, p):
        u, v = p["u"], p["v"]
        return max(la.magnitude(u), la.magnitude(v), la.magnitude(u * v), 1.0)

    def legend(self, p):
        th = self.th
        return [("u", th["v"], "line"), ("v", th["kv"], "line"), ("u ⊙ v", th["w"], "line")]

    def geometry(self, s, p):
        u, v, c = p["u"], p["v"], s.col
        h = u * v
        L = self.extent(p)
        if p["boxes"] and len(u) == 2:
            for key, vec, col in (("bv", v, c["v"]), ("bh", h, c["w"])):
                x, y = vec
                s.path(key, [[0, 0], [x, 0], [x, y], [0, y], [0, 0]], col, ls="--", lw=1.1,
                       alpha=0.6)
            s.label("tx", [h[0], -0.07 * L], f"{F(u[0], 2)} × {F(v[0], 2)}", c["w"], size=8.5)
            s.label("ty", [-0.12 * L, h[1]], f"{F(u[1], 2)} × {F(v[1], 2)}", c["w"], size=8.5)
        s.arrow("v", v, c["v"], width=0.009, zorder=4)
        s.arrow("u", u, c["u"], width=0.008, zorder=4, alpha=0.8)
        s.arrow("h", h, c["w"], width=0.011, zorder=5)
        for key, vec, text, col in (("lu", u, "u", c["u"]), ("lv", v, "v", c["v"]),
                                    ("lh", h, "u ⊙ v", c["w"])):
            n = unit(vec)
            if n is not None:
                s.label(key, vec + n * L * 0.09, text, col, size=11)
        s.title(f"u ⊙ v = {la.fmt_vec(h)}\nsum of its entries = u·v = {F(la.dot(u, v))}")

    def status_text(self, p):
        h = p["u"] * p["v"]
        return f"u ⊙ v = {la.fmt_vec(h)}   —   sum = {F(float(h.sum()))} = u·v"

    def algebra(self, p):
        return hadamard_steps, (p["u"], p["v"])

    def python_code(self, p):
        return hadamard_code(p["u"], p["v"])


def hadamard_steps(b, u, v):
    pal = b.pal
    U, V, W = pal["v"], pal["kv"], pal["w"]
    u, v = la.to_vector(u), la.to_vector(v)
    n = len(u)
    h = u * v

    b.heading("The definition")
    b.row([r"$(\mathbf{u}\odot\mathbf{v})_i = u_i\,v_i$"])
    b.note("Multiply matching components, but keep them separate: the answer is a "
           "vector of the same size (unlike the dot product, which adds them up).")

    b.heading("With your numbers")
    b.row([r"$\mathbf{u}\odot\mathbf{v} =$", ("vec", nums(u), U), r"$\odot$",
           ("vec", nums(v), V), "$=$",
           ("vec", [f"${P(x)}\\cdot{P(y)}$" for x, y in zip(u, v)], pal["ink"]),
           "$=$", ("vec", nums(h), W)])

    b.heading("Compared with the dot product")
    b.long(r"\mathbf{u}\cdot\mathbf{v} = \sum_i (\mathbf{u}\odot\mathbf{v})_i = ",
           [P(x) for x in h], last=f" = {F(float(h.sum()))}")
    b.note("Add up the entries of u ⊙ v and you get the dot product.")

    b.heading("What it does: stretches each axis")
    D = [[f"${F(u[i]) if i == j else '0'}$" for j in range(n)] for i in range(n)]
    b.row([r"$\mathbf{u}\odot\mathbf{v} = \mathrm{diag}(\mathbf{u})\,\mathbf{v} =$",
           ("mat", D, U), ("vec", nums(v), V)])
    b.note("Each component of v is scaled by its own factor: x by u₁, y by u₂"
           + (", z by u₃." if n == 3 else ".") + " In the plot, the dashed box of v "
           "is stretched into the box of u ⊙ v.")

    b.heading("Properties (checked)")
    b.check(r"\mathbf{u}\odot\mathbf{v} = \mathbf{v}\odot\mathbf{u}", la.fmt_vec(h),
            np.allclose(u * v, v * u))
    ones = np.ones(n)
    b.check(r"\mathbf{1}\odot\mathbf{v} = \mathbf{v}", la.fmt_vec(v),
            np.allclose(ones * v, v))
    b.check(r"\sum_i (\mathbf{u}\odot\mathbf{v})_i = \mathbf{u}\cdot\mathbf{v}",
            F(float(h.sum())), close(float(h.sum()), la.dot(u, v)))


def hadamard_code(u, v):
    return "\n".join([
        "import numpy as np",
        "",
        f"u = {np_array(u)}",
        f"v = {np_array(v)}",
        "",
        "# Hadamard (element-wise) product: in NumPy that is just *",
        "h = u * v                    # same as np.multiply(u, v)",
        'print("u ⊙ v =", h)',
        "",
        "# Careful: * is NOT the dot product. The dot product adds the entries up:",
        'print("sum of u ⊙ v =", h.sum(), "  np.dot(u, v) =", np.dot(u, v))',
        "",
        "# Same as multiplying by the diagonal matrix diag(u)",
        'print("diag(u) @ v =", np.diag(u) @ v)',
    ])
