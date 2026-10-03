"""
tab_notation.py
---------------
Vector product notations side by side: the different ways each product is
written, what kind of thing it gives (number, vector, matrix), its value for
the user's vectors, and the NumPy call. The plot area is a reference table.
"""

import numpy as np
from matplotlib.patches import Rectangle

import linalg_core as la
from algebra_render import nums
from tab_base import BaseTab
from themes import mix
from topic_util import F, np_array


def products(u, v, k):
    """(name, notations, gives, value text, NumPy) for every product."""
    n = len(u)
    outer = np.outer(u, v)
    if n == 3:
        cross = ("Cross product", "u × v", "a vector (3D only)",
                 la.fmt_vec(la.cross(u, v), 3), "np.cross(u, v)")
    else:
        cross = ("Cross product", "u × v", "a number in 2D", F(la.cross(u, v), 3),
                 "u[0]*v[1] - u[1]*v[0]")
    return [
        ("Scalar multiple", "k v   ·   kv", "a vector", la.fmt_vec(k * v, 3), "k * v"),
        ("Dot (inner) product", "u·v   ⟨u, v⟩   uᵀv   Σ uᵢvᵢ", "a number",
         F(la.dot(u, v), 3), "np.dot(u, v)   or   u @ v"),
        ("Hadamard product", "u ⊙ v   ·   u ∘ v", "a vector (same size)",
         la.fmt_vec(u * v, 3), "u * v"),
        ("Outer product", "u ⊗ v   ·   u vᵀ", f"a {n}×{n} matrix",
         "\n".join(la.fmt_vec(r, 2) for r in outer), "np.outer(u, v)"),
        cross,
        ("Length (norm)", "‖v‖ = √(v·v)", "a number", F(la.magnitude(v), 3),
         "np.linalg.norm(v)"),
    ]


class NotationTable:
    """The reference table drawn with Matplotlib text (full redraw on change)."""
    blit = False

    def __init__(self, fig, th):
        self.fig, self.th = fig, th
        fig.clf()
        fig.set_facecolor(th["plot_bg"])
        self.ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])

    def view(self):
        return None

    def draw_frame(self):
        pass

    def render(self, p):
        th, ax = self.th, self.ax
        ax.clear()
        ax.axis("off")
        rows = products(p["u"], p["v"], p["k"])
        heights = [max(2.4, 1.5 + 0.75 * r[3].count("\n") + 0.75) for r in rows]
        total = 1.6 + sum(heights)
        ax.set_xlim(0, 1)
        ax.set_ylim(total, 0)
        cols = (0.01, 0.43, 0.70)
        for x, head in zip(cols, ("Product and its notations", "Gives / your result",
                                  "NumPy")):
            ax.text(x, 0.7, head, fontsize=10, fontweight="bold", color=th["accent"],
                    va="center")
        ax.plot([0, 1], [1.3, 1.3], color=th["border"], lw=1)
        y = 1.6
        for i, ((name, notation, gives, value, code), h) in enumerate(zip(rows, heights)):
            if i % 2 == 0:
                ax.add_patch(Rectangle((0, y - 0.15), 1, h, fc=mix(th["plot_bg"], th["text"],
                                                                     0.04), ec="none"))
            ax.text(cols[0], y + 0.3, name, fontsize=10, fontweight="bold",
                    color=th["text"], va="top")
            ax.text(cols[0], y + 1.1, notation, fontsize=11, color=th["v"], va="top")
            ax.text(cols[1], y + 0.3, gives, fontsize=9, color=th["muted"], va="top")
            ax.text(cols[1], y + 1.1, value, fontsize=10, color=th["w"], va="top",
                    fontweight="bold", linespacing=1.3)
            ax.text(cols[2], y + 0.3, code, fontsize=9, color=th["text"], va="top",
                    family="monospace")
            y += h


class NotationTab(BaseTab):
    title = "  Product Notations  "

    def build_controls(self, side):
        self.add_dim_selector(side)
        self.u = self.add_vector_box(side, "Vector u", "U", ("2", "1", "3"))
        self.v = self.add_vector_box(side, "Vector v", "V", ("3", "-2", "4"))
        box = self.add_box(side, "Scalar")
        self.k, _ = self.add_entry_row(box, "k =", "3")
        self.add_actions(side, label="▶  Replay steps")

    def params_for_plot(self):
        u, err = self.read_vector(self.u, "u")
        v, err2 = self.read_vector(self.v, "v")
        k, err3 = self.read_number(self.k, "k")
        if err or err2 or err3:
            return None, err or err2 or err3
        return dict(u=u, v=v, k=k), None

    def draw(self, p):
        if self.scene is None or self.scene.th["name"] != self.th["name"]:
            self.scene = NotationTable(self.fig, self.th)
        self.scene.render(p)
        return True

    def status_text(self, p):
        return "number: u·v, ‖v‖   —   vector: kv, u ⊙ v, u × v   —   matrix: u ⊗ v"

    def on_first_show(self):
        self.replay_steps()

    def algebra(self, p):
        return notation_steps, (p["u"], p["v"], p["k"])

    def python_code(self, p):
        return notation_code(p["u"], p["v"], p["k"])


def notation_steps(b, u, v, k):
    pal = b.pal
    V, W, INK = pal["kv"], pal["w"], pal["ink"]
    u, v = la.to_vector(u), la.to_vector(v)
    n = len(u)
    U_, V_ = r"\mathbf{u}", r"\mathbf{v}"

    b.heading("Scalar multiplication")
    b.row([rf"$k{V_} = {F(k)}\,$", ("vec", nums(v), V), "$=$", ("vec", nums(k * v), W)])
    b.note("Number × vector → a vector.")

    b.heading("Dot product (inner product)")
    b.row([rf"${U_}\cdot{V_} = \langle {U_}, {V_}\rangle = {U_}^T{V_} = "
           rf"\sum_i u_i v_i = {F(la.dot(u, v))}$"])
    b.note("Four ways to write the same thing. Vector · vector → a number.")

    b.heading("Hadamard (element-wise) product")
    b.row([rf"${U_}\odot{V_} =$", ("vec", nums(u * v), W)])
    b.note("Also written u ∘ v. Multiply matching entries → a vector.")

    b.heading("Outer product")
    M = np.outer(u, v)
    b.row([rf"${U_}\otimes{V_} = {U_}{V_}^T =$",
           ("mat", [[f"${F(x, 3)}$" for x in r] for r in M], W)])
    b.note(f"Column ({n}×1) times row (1×{n}) → a {n}×{n} matrix.")

    b.heading("Cross product")
    if n == 3:
        b.row([rf"${U_}\times{V_} =$", ("vec", nums(la.cross(u, v)), W)])
        b.note("Only in 3D: a vector perpendicular to both u and v.")
    else:
        b.row([rf"${U_}\times{V_} = u_1v_2 - u_2v_1 = {F(la.cross(u, v))}$"])
        b.note("In 2D it is just a number: the signed area of the parallelogram.")

    b.heading("Which gives what?")
    b.result_box(["a number:   u·v = ⟨u, v⟩ = uᵀv,   ‖v‖",
                  "a vector:   k v,   u ⊙ v,   u × v (3D)",
                  "a matrix:   u ⊗ v = u vᵀ"], bold=0, color=INK)


def notation_code(u, v, k):
    lines = [
        "import numpy as np",
        "",
        f"u = {np_array(u)}",
        f"v = {np_array(v)}",
        f"k = {F(k)}",
        "",
        'print("k v   =", k * v)                  # scalar multiple',
        'print("u · v =", np.dot(u, v), u @ v)    # dot product (two ways)',
        'print("u ⊙ v =", u * v)                  # Hadamard: * is element-wise',
        'print("u ⊗ v =")',
        "print(np.outer(u, v))                   # outer product = u vᵀ",
    ]
    if len(u) == 3:
        lines.append('print("u × v =", np.cross(u, v))           # cross product (3D)')
    else:
        lines.append('print("u × v =", u[0]*v[1] - u[1]*v[0])   # 2D cross: a number')
    lines.append('print("‖v‖   =", np.linalg.norm(v))     # length')
    return "\n".join(lines)
