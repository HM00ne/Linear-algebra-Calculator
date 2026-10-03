"""
tab_rowcol.py
-------------
Column vectors (n×1) and row vectors (1×n): the transpose, why shapes decide
what a product gives, the inner product aᵀb (a number) and the outer product
abᵀ (an n×n matrix). The plot is a diagram of the actual boxes of numbers.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

import linalg_core as la
from algebra_render import nums
from tab_base import BaseTab
from themes import mix
from topic_util import F, P, close, np_array

MAX_N = 5
FONT_FOR_N = {2: 12, 3: 11, 4: 10, 5: 9}
VIEWS = (("Entries", "entries"), ("Columns", "columns"), ("Rows", "rows"))


def sub(k):
    """Index as subscript digits: 3 -> '₃'."""
    return "".join("₀₁₂₃₄₅₆₇₈₉"[int(c)] for c in str(k))


class MatrixDiagram:
    """Boxes-of-numbers picture. Redrawn in full when the values change
    (it does not animate frame by frame, so no blitting is needed)."""
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
        n, a, b = p["n"], p["a"], p["b"]
        fs = FONT_FOR_N[n]
        U, V, W = th["v"], th["kv"], th["w"]

        def cell(x, y, value, color, strong=False, fill=None):
            ax.add_patch(Rectangle((x, y), 1, 1, fc=fill or mix(th["plot_bg"], color, 0.16),
                                   ec=color, lw=1.6 if strong else 1.1))
            ax.text(x + 0.5, y + 0.5, F(value, 2), ha="center", va="center", fontsize=fs,
                    color=th["text"], fontweight="bold" if strong else "normal")

        def column(x, y, values, color):
            for i, val in enumerate(values):
                cell(x, y + i, val, color)

        def row(x, y, values, color):
            for j, val in enumerate(values):
                cell(x + j, y, val, color)

        def heading(y, text):
            ax.text(0, y, text, ha="left", va="center", fontsize=10.5, color=th["accent"],
                    fontweight="bold")

        def shape(x, y, text):
            ax.text(x, y, text, ha="center", va="center", fontsize=9, color=th["muted"])

        def arrow(x0, y0, x1, y1):
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                         mutation_scale=14, color=th["muted"], lw=1.4))

        y = 0.0
        heading(y, "Transpose: a column becomes a row")
        y += 0.8
        column(0, y, a, U)
        shape(0.5, y + n + 0.45, f"a   ({n}×1)")
        mid = y + n / 2
        arrow(1.4, mid, 3.0, mid)
        ax.text(2.2, mid - 0.45, "ᵀ", ha="center", va="center", fontsize=fs + 2, color=th["muted"])
        row(3.4, mid - 0.5, a, U)
        shape(3.4 + n / 2, mid + 0.95, f"aᵀ   (1×{n})")
        y += n + 1.5

        if p["inner"]:
            heading(y, f"Inner product aᵀb:  (1×{n})({n}×1) = 1×1, a number")
            y += 0.8
            mid = y + n / 2
            row(0, mid - 0.5, a, U)
            ax.text(n + 0.45, mid, "×", ha="center", va="center", fontsize=fs + 3,
                    color=th["text"])
            column(n + 0.9, y, b, V)
            ax.text(n + 2.4, mid, "=", ha="center", va="center", fontsize=fs + 3,
                    color=th["text"])
            cell(n + 2.9, mid - 0.5, float(np.dot(a, b)), W, strong=True)
            shape(n / 2, mid + 0.95, f"aᵀ (1×{n})")
            shape(n + 1.4, y + n + 0.45, f"b ({n}×1)")
            y += n + 1.5

        if p["outer"]:
            heading(y, f"Outer product abᵀ:  ({n}×1)(1×{n}) = {n}×{n}, a matrix")
            y += 0.8
            row(1.4, y, b, V)                                  # bᵀ across the top
            column(0, y + 1.3, a, U)                           # a down the side
            M = np.outer(a, b)
            top = float(np.max(np.abs(M))) or 1.0
            for i in range(n):
                for j in range(n):
                    val = M[i, j]
                    tint = mix(th["plot_bg"], th["kv"] if val >= 0 else th["v"],
                               0.12 + 0.5 * abs(val) / top)
                    cell(1.4 + j, y + 1.3 + i, val, th["border"], fill=tint)
            ax.text(0.5, y + 0.5, "×", ha="center", va="center", fontsize=fs + 3,
                    color=th["muted"])                    # a multiplication table
            g0 = y + 1.3                                       # top of the grid
            if p["view"] == "columns":                        # column j = b_j · a
                for j in range(n):
                    ax.add_patch(Rectangle((1.4 + j + 0.06, g0 + 0.06), 0.88, n - 0.12,
                                           fc="none", ec=W, lw=2.2))
                    ax.text(1.9 + j, g0 + n + 0.4, f"b{sub(j + 1)}a", ha="center",
                            va="center", fontsize=fs, color=W, fontweight="bold")
            elif p["view"] == "rows":                         # row i = a_i · bᵀ
                for i in range(n):
                    ax.add_patch(Rectangle((1.4 + 0.06, g0 + i + 0.06), n - 0.12, 0.88,
                                           fc="none", ec=W, lw=2.2))
                    ax.text(1.6 + n, g0 + i + 0.5, f"a{sub(i + 1)}bᵀ", ha="left",
                            va="center", fontsize=fs, color=W, fontweight="bold")
            y += n + 1.3 + (1.0 if p["view"] == "columns" else 0.6)

        # square cells: widen (or lengthen) the view to the shape of the axes,
        # keeping the content in the top-left corner
        x0, x1, y0, y1 = -0.3, max(3.4 + n, n + 4.0) + 0.3, -0.6, y + 0.2
        pos = ax.get_position()
        fw, fh = self.fig.get_size_inches()
        box = (pos.width * fw) / (pos.height * fh)
        if (x1 - x0) / (y1 - y0) < box:
            x1 = x0 + (y1 - y0) * box
        else:
            y1 = y0 + (x1 - x0) / box
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)                                     # y grows downwards


class RowColumnTab(BaseTab):
    title = "  Row & Column Vectors  "

    def build_controls(self, side):
        self.n = tk.StringVar(value="3")
        box = self.add_box(side, "Number of components n", pady=(0, 0))
        row = ttk.Frame(box)
        row.pack(fill="x")
        for k in range(2, MAX_N + 1):
            ttk.Radiobutton(row, text=str(k), value=str(k), variable=self.n,
                            command=self._on_n).pack(side="left", padx=(0, 10))
        self.a_vars, self.a_entries = self._entries(side, "Vector a", "U",
                                                    ("2", "-1", "3", "1", "0"))
        self.b_vars, self.b_entries = self._entries(side, "Vector b", "V",
                                                    ("1", "4", "-2", "2", "1"))
        self.show_inner, self.show_outer = self.add_checks(
            side, "Display", [("Inner product aᵀb", True), ("Outer product abᵀ", True)])
        self.view = tk.StringVar(value="entries")
        box = self.add_box(side, "Look at abᵀ by")
        row = ttk.Frame(box)
        row.pack(fill="x")
        for text, value in VIEWS:
            ttk.Radiobutton(row, text=text, value=value, variable=self.view,
                            command=self.refresh).pack(side="left", padx=(0, 10))
        self.add_actions(side, label="▶  Replay steps")
        self._on_n()

    def _entries(self, side, title, role, defaults):
        box = self.add_box(side, title, role)
        variables = [tk.StringVar(value=s) for s in defaults]
        grid = ttk.Frame(box)
        grid.pack(fill="x")
        entries = []
        for i, var in enumerate(variables):
            ttk.Label(grid, text=f"{i + 1}", style="Muted.TLabel").grid(row=0, column=i)
            e = ttk.Entry(grid, textvariable=var, width=5, justify="center")
            e.grid(row=1, column=i, padx=2)
            entries.append(e)
        self.watch(*variables)
        return variables, entries

    def _on_n(self):
        n = int(self.n.get())
        for entries in (self.a_entries, self.b_entries):
            for i, e in enumerate(entries):
                e.configure(state="normal" if i < n else "disabled")
        self.refresh()

    def params_for_plot(self):
        n = int(self.n.get())
        a, err = self.read_vector(self.a_vars, "a", n)
        b, err2 = self.read_vector(self.b_vars, "b", n)
        if err or err2:
            return None, err or err2
        return dict(n=n, a=a, b=b, inner=self.show_inner.get(),
                    outer=self.show_outer.get(), view=self.view.get()), None

    def read_vector(self, variables, name, n=None):
        values = []
        for i, var in enumerate(variables[:n]):
            x, err = self.read_number(var, f"{name}{i + 1}")
            if err:
                return None, err
            values.append(x)
        return la.to_vector(values), None

    def draw(self, p):
        if self.scene is None or self.scene.th["name"] != self.th["name"]:
            self.scene = MatrixDiagram(self.fig, self.th)
        self.scene.render(p)
        return True

    def status_text(self, p):
        return (f"aᵀb = {F(float(np.dot(p['a'], p['b'])))}   —   a is {p['n']}×1,  "
                f"aᵀ is 1×{p['n']},  abᵀ is {p['n']}×{p['n']}")

    def on_first_show(self):
        self.replay_steps()

    def on_resize(self):
        self.refresh()                   # keep the cells square at the new size

    def algebra(self, p):
        return rowcol_steps, (p["a"], p["b"])

    def python_code(self, p):
        return rowcol_code(p["a"], p["b"])


def rowcol_steps(b, a, bb):
    pal = b.pal
    U, V, W = pal["v"], pal["kv"], pal["w"]
    a, bb = la.to_vector(a), la.to_vector(bb)
    n = len(a)
    d = float(np.dot(a, bb))

    b.heading("Column vector and row vector")
    b.row([r"$\mathbf{a} =$", ("vec", nums(a), U), rf"$\quad ({n}\times 1)$"])
    b.row([r"$\mathbf{a}^T =$", ("row", nums(a), U), rf"$\quad (1\times {n})$"])
    b.note("The transpose ᵀ turns the column into a row (and back again): (aᵀ)ᵀ = a. "
           "Same numbers, different shape.")

    b.heading("Shapes decide what a product gives")
    b.row([r"$(m\times n)\,(n\times p) \;\rightarrow\; (m\times p)$"])
    b.note("The inner sizes must match; the outer sizes are the shape of the answer.")
    b.row([rf"$\mathrm{{row}}\times\mathrm{{column}}:\ (1\times {n})({n}\times 1)"
           rf" \rightarrow 1\times 1$"])
    b.row([rf"$\mathrm{{column}}\times\mathrm{{row}}:\ ({n}\times 1)(1\times {n})"
           rf" \rightarrow {n}\times {n}$"])

    b.heading("Inner product aᵀb  (row × column)")
    b.row([r"$\mathbf{a}^T\mathbf{b} =$", ("row", nums(a), U), ("vec", nums(bb), V)])
    b.long("= ", [f"{P(x)}\\cdot{P(y)}" for x, y in zip(a, bb)])
    b.long("= ", [P(x * y) for x, y in zip(a, bb)], last=f" = {F(d)}")
    b.note("A single number: this is exactly the dot product a·b.")

    b.heading("Outer product abᵀ  (column × row)")
    b.row([r"$\mathbf{a}\mathbf{b}^T =$", ("vec", nums(a), U), ("row", nums(bb), V)])
    M = np.outer(a, bb)
    b.row(["$=$", ("mat", [[f"${F(x, 3)}$" for x in r] for r in M], W)])
    b.note("Entry (i, j) is aᵢ·bⱼ.")

    b.heading("Column view: each column is a, scaled")
    b.row([r"$\mathbf{a}\mathbf{b}^T = \left[\ b_1\mathbf{a}\ \ b_2\mathbf{a}\ \cdots\ "
           rf"b_{{{n}}}\mathbf{{a}}\ \right]$"])
    b.row([rf"$\mathrm{{column\ 1}} = b_1\mathbf{{a}} = {P(bb[0])}\,$", ("vec", nums(a), U),
           "$=$", ("vec", nums(bb[0] * a), W)])
    b.note("Every column points along a (or is zero).")

    b.heading("Row view: each row is bᵀ, scaled")
    b.row([rf"$\mathrm{{row\ 1}} = a_1\mathbf{{b}}^T = {P(a[0])}\,$", ("row", nums(bb), V)])
    b.row(["$=$", ("row", nums(a[0] * bb), W)], indent=40)
    b.note("Every row is a multiple of bᵀ. Since all columns (and rows) are multiples "
           "of one vector, abᵀ has rank 1 (unless a or b is zero).")

    b.heading("Useful facts, checked")
    b.subhead("aᵀb = bᵀa  (order does not matter for the inner product)")
    b.check(F(d), F(float(np.dot(bb, a))), close(d, float(np.dot(bb, a))))
    b.subhead("trace(abᵀ) = aᵀb  (sum of the diagonal)")
    tr = float(np.trace(M))
    b.check(" + ".join(P(x) for x in np.diag(M)) + f" = {F(tr)}", F(d), close(tr, d))
    b.subhead("(a + b)ᵀ = aᵀ + bᵀ  and  (aᵀ)ᵀ = a")
    b.row([r"$(\mathbf{a}+\mathbf{b})^T =$", ("row", nums(a + bb), W)])


def rowcol_code(a, b):
    return "\n".join([
        "import numpy as np",
        "",
        "# ── Inputs ───────────────────────────────",
        f"a = {np_array(a)}",
        f"b = {np_array(b)}",
        "",
        "# ── Column and row versions ──────────────",
        "a_col = a.reshape(-1, 1)    # n x 1",
        "a_row = a.reshape(1, -1)    # 1 x n  (same as a_col.T)",
        'print("column shape:", a_col.shape, "  row shape:", a_row.shape)',
        'print("transpose of the column is the row:", np.array_equal(a_col.T, a_row))',
        "",
        "# ── Inner product: (1 x n) @ (n x 1) -> 1 x 1 ──",
        "inner = a_row @ b.reshape(-1, 1)",
        'print("a^T b =", inner, "-> as a number:", inner.item())',
        'print("same as the dot product:", np.isclose(inner.item(), a @ b))',
        "",
        "# ── Outer product: (n x 1) @ (1 x n) -> n x n ──",
        "outer = a_col @ b.reshape(1, -1)",
        'print("a b^T =")',
        "print(outer)",
        'print("np.outer gives the same:", np.array_equal(outer, np.outer(a, b)))',
        'print("rank:", np.linalg.matrix_rank(outer))',
        "",
        "# Column view: column j is b[j] * a",
        "for j in range(len(b)):",
        '    print("column", j, "=", outer[:, j], " = b[j]*a:", np.allclose(outer[:, j], b[j] * a))',
        "# Row view: row i is a[i] * b",
        "for i in range(len(a)):",
        '    print("row", i, "=", outer[i, :], " = a[i]*b:", np.allclose(outer[i, :], a[i] * b))',
        'print("trace(a b^T) == a^T b:", np.isclose(np.trace(outer), a @ b))',
    ])
