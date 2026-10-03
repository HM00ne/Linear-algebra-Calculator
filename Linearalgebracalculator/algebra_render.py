"""
algebra_render.py
-----------------
Draws algebraic working as a picture: real column/row vectors and matrices
with brackets, square roots, fractions and angles (Matplotlib mathtext, no
LaTeX install needed). Returns a PNG plus the vertical position of every
step, so the application can reveal the steps one by one like they are
being written on a board.

Each topic supplies a "builder": a function builder(board, *args) that adds
headings, rows and notes to the board. render_steps() runs it and caches
the result.
"""

import io
import base64
import textwrap
from functools import lru_cache

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyBboxPatch, Circle

import linalg_core as la

DPI = 100
# default colours; a theme can override any of them via the palette argument
PALETTE = dict(ink="#1f2937", muted="#6b7280", accent="#2563eb", on_accent="#ffffff",
               v="#2563eb", kv="#dc2626", w="#059669", proj="#d97706",
               paper="#ffffff", result_bg="#fef2f2", rule="#e5e7eb")
FONT = 12.5          # pt, size of the maths
LINE_H = 24          # px, height of one row inside a vector/matrix
BOARD_IN = 45        # inches of scratch height to lay out on (cropped at the end)


def _p(x, digits=4):
    """Number for use inside a formula; negatives get parentheses."""
    s = la.fmt(x, digits)
    return f"({s})" if s.startswith("-") else s


paren = _p


def nums(values, digits=4):
    """Entries for a ('vec'/'row') piece: each number as mathtext."""
    return [f"${la.fmt(x, digits)}$" for x in values]


class Board:
    """A tiny layout engine: rows of maths placed top to bottom (pixel units)."""

    def __init__(self, width_px, pal, scale=1.0):
        """Layout happens in 100 %-scaling pixels; `scale` (e.g. 1.25 on a
        screen at 125 %) only raises the resolution of the final picture."""
        self.W = width_px
        self.pal = pal
        self.scale = scale
        self.fig = Figure(figsize=(width_px / DPI, BOARD_IN), dpi=DPI * scale,
                          facecolor=pal["paper"])
        FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width_px)
        self.ax.set_ylim(BOARD_IN * DPI, 0)     # y grows downwards, in pixels
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        self.y = 14
        self.x0 = 18
        self.steps = []
        self._made = None          # artists created during a measuring pass
        self._step_no = 0

    # -- measuring -------------------------------------------------------
    def _size(self, artist):
        bb = artist.get_window_extent(self.renderer)
        return bb.width / self.scale, bb.height / self.scale

    def _text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FONT)
        kw.setdefault("color", self.pal["ink"])
        kw.setdefault("va", "center")
        kw.setdefault("ha", "left")
        t = self.ax.text(x, y, s, **kw)
        if self._made is not None:
            self._made.append(t)
        return t

    def _plot(self, xs, ys, **kw):
        line, = self.ax.plot(xs, ys, **kw)
        if self._made is not None:
            self._made.append(line)

    # -- building blocks -------------------------------------------------
    def heading(self, title, number=None):
        """Numbered heading; also starts a new step for the board replay."""
        if self._step_no:
            self.end_step()
            self.separator()
        self._step_no += 1
        self._start = self.y
        yc = self.y + 14
        accent = self.pal["accent"]
        self.ax.add_patch(Circle((self.x0 + 10, yc), 11, color=accent, zorder=2))
        self._text(self.x0 + 10, yc, str(number or self._step_no), color=self.pal["on_accent"],
                   fontsize=10.5, fontweight="bold", ha="center", zorder=3)
        self._text(self.x0 + 30, yc, title, fontsize=12, fontweight="bold", color=accent)
        self.y += 36

    def note(self, s, color=None, size=10.5):
        color = color or self.pal["muted"]
        chars = int((self.W - self.x0 - 14) / (size * 0.8))      # rough fit per line
        s = "\n".join(textwrap.fill(part, chars) for part in s.split("\n"))
        t = self._text(self.x0 + 6, self.y + 4, s, color=color, fontsize=size, style="italic",
                       va="top")
        self.y += self._size(t)[1] + 10

    def row(self, pieces, indent=12):
        """pieces: mathtext strings and/or ('vec', entries, colour),
        ('row', entries, colour) or ('mat', [[entries]], colour).
        A row that would not fit the width is shrunk to fit."""
        size = FONT
        avail = self.W - self.x0 - indent - 8
        width = self._measure(pieces, size)
        if width > avail:
            size = FONT * max(0.6, avail / width)
        rh = max(self._piece_h(p, size) for p in pieces)
        yc = self.y + rh / 2 + 4
        x = self.x0 + indent
        for p in pieces:
            x += self._piece(p, x, yc, size) + 6
        self.y += rh + 14

    def long(self, first, terms, sep=" + ", last="", indent=12, cont_indent=40):
        """A long sum written over as many lines as needed:
        first + t1 + t2 + ... + last, breaking before a `sep` when the line
        is full. All parts are mathtext without $."""
        avail = self.W - self.x0 - cont_indent - 8
        lines, cur = [], first + (terms[0] if terms else "")
        for t in terms[1:]:
            trial = cur + sep + t
            if self._measure([f"${trial}$"], FONT) > avail:
                lines.append(cur)
                cur = sep.strip() + " " + t
            else:
                cur = trial
        if last:
            if self._measure([f"${cur}{last}$"], FONT) > avail:
                lines.append(cur)
                cur = last.lstrip()
            else:
                cur += last
        lines.append(cur)
        for i, line in enumerate(lines):
            self.row([f"${line}$"], indent=indent if i == 0 else cont_indent)

    def subhead(self, s, color=None):
        """A bold line of plain text (e.g. the name of a property); wraps."""
        chars = int((self.W - self.x0 - 14) / (11 * 0.88))
        t = self._text(self.x0 + 6, self.y + 4, textwrap.fill(s, chars), fontsize=11,
                       fontweight="bold", color=color or self.pal["ink"], va="top")
        self.y += self._size(t)[1] + 8

    def compare(self, lhs, op, rhs, ok, indent=24):
        """'lhs op rhs  ✓' with any relation, e.g. op = r'\\leq'."""
        mark = r"\ \ \checkmark" if ok else r"\ \ \times"
        self.row([f"${lhs} {op} {rhs}{mark}$"], indent=indent)

    def check(self, lhs, rhs, ok=True, indent=24):
        """'lhs = rhs  ✓' (or ✗) as one row; lhs/rhs are mathtext without $."""
        mark = r"\ \ \checkmark" if ok else r"\ \ \times"
        self.row([f"${lhs} = {rhs}{mark}$"], indent=indent)

    def _measure(self, pieces, size):
        self._made = []
        x = 0
        for p in pieces:
            x += self._piece(p, x, -500, size) + 6
        for artist in self._made:
            artist.remove()
        self._made = None
        return x - 6

    def _piece(self, p, x, yc, size):
        if isinstance(p, str):
            return self._size(self._text(x, yc, p, fontsize=size))[0]
        kind, data, color = p
        rows = {"vec": lambda: [[e] for e in data], "row": lambda: [list(data)],
                "mat": lambda: data}[kind]()
        return self._matrix(x, yc, rows, color, size)

    def _piece_h(self, p, size):
        if isinstance(p, str):
            t = self._text(0, -500, p, fontsize=size)
            h = self._size(t)[1]
            t.remove()
            return h
        n = len(p[1]) if p[0] in ("vec", "mat") else 1
        return n * LINE_H * size / FONT + 8

    def _matrix(self, x, yc, rows, color, size=FONT):
        """Matrix (or column/row vector) in square brackets, centred on yc."""
        lh = LINE_H * size / FONT
        nr, nc = len(rows), len(rows[0])
        h = nr * lh
        top, bot = yc - h / 2 - 3, yc + h / 2 + 3
        cells = [[self._text(0, top + 3 + lh * (i + 0.5), e, color=color, ha="center",
                             fontsize=size) for e in r] for i, r in enumerate(rows)]
        col_w = [max(self._size(cells[i][j])[0] for i in range(nr)) for j in range(nc)]
        gap = 14 * size / FONT
        w = sum(col_w) + gap * (nc - 1) + 16
        cx = x + 8
        for j in range(nc):
            for i in range(nr):
                cells[i][j].set_x(cx + col_w[j] / 2)
            cx += col_w[j] + gap
        tick = 5
        kw = dict(color=self.pal["ink"], lw=1.4, solid_capstyle="butt")
        self._plot([x + tick, x, x, x + tick], [top, top, bot, bot], **kw)
        self._plot([x + w - tick, x + w, x + w, x + w - tick], [top, top, bot, bot], **kw)
        return w

    def result_box(self, lines, bold=1, color=None):
        color = color or self.pal["kv"]
        self._bold = bold
        pad = 10
        h = len(lines) * 22 + 2 * pad
        top = self.y + 2
        self.ax.add_patch(FancyBboxPatch(
            (self.x0, top), self.W - 2 * self.x0, h,
            boxstyle="round,pad=0,rounding_size=8",
            facecolor=self.pal["result_bg"], edgecolor=color, lw=1.2))
        texts = [self._text(self.x0 + 12, top + pad + 11 + 22 * i, s, fontsize=11.5,
                            color=color if i < self._bold else self.pal["ink"],
                            fontweight="bold" if i < self._bold else "normal")
                 for i, s in enumerate(lines)]
        widest = max(self._size(t)[0] for t in texts)
        avail = self.W - 2 * self.x0 - 24
        if widest > avail:                       # shrink every line to fit the box
            for t in texts:
                t.set_fontsize(11.5 * avail / widest)
        self.y += h + 14

    def separator(self):
        self.ax.plot([self.x0, self.W - self.x0], [self.y, self.y],
                     color=self.pal["rule"], lw=1)
        self.y += 12

    # -- steps -------------------------------------------------------------
    def begin_step(self):
        self._start = self.y

    def end_step(self):
        self.steps.append((self._start, self.y))

    # -- output ------------------------------------------------------------
    def to_png(self):
        H = int(self.y + 10)
        self.fig.set_size_inches(self.W / DPI, H / DPI)
        self.ax.set_ylim(H, 0)
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=DPI * self.scale, facecolor=self.pal["paper"])
        return buf.getvalue(), H


_Board = Board          # old name


def render_steps(builder, args, width_px=400, palette=None, scale=1.0):
    """Run builder(board, *args) and return (png_base64, height, steps), with
    height and steps in screen pixels (layout units × scale).

    Typesetting maths takes 0.2-0.4 s, so results are cached: going back to
    values (or a theme) already seen is instant. `args` must be hashable."""
    pal = dict(PALETTE, **(palette or {}))
    return _render_steps(builder, args, width_px, tuple(sorted(pal.items())), scale)


@lru_cache(maxsize=64)
def _render_steps(builder, args, width_px, pal, scale=1.0):
    b = Board(width_px, dict(pal), scale)
    builder(b, *args)
    if b._step_no:                       # close the last step opened by heading()
        b.end_step()
    png, H = b.to_png()
    steps = tuple((y0 * scale, y1 * scale) for y0, y1 in b.steps)
    return base64.b64encode(png).decode("ascii"), round(H * scale), steps


def render(k, v, width_px=400, palette=None):
    """The scalar multiplication working for k·v (see scalar_steps)."""
    return render_steps(scalar_steps, (float(k), tuple(float(x) for x in v)),
                        width_px, palette)


def scalar_steps(b, k, v):
    pal = b.pal
    INK, V_COLOR, KV_COLOR = pal["ink"], pal["v"], pal["kv"]
    v = la.to_vector(v)
    kv = la.scalar_multiply(k, v)
    n = len(v)
    names = ["x", "y", "z"][:n]
    mv, mkv = la.magnitude(v), la.magnitude(kv)
    K = la.fmt(k)

    # ---------- Step 1: component-wise multiplication
    b.heading("Multiply every component by k")
    b.row([r"$k\,\mathbf{v} = k$",
           ("vec", [f"${c}$" for c in names], INK),
           "$=$",
           ("vec", [f"$k\\cdot {c}$" for c in names], INK)])
    b.row([f"${_p(k)}$", r"$\cdot$",
           ("vec", [f"${la.fmt(x)}$" for x in v], V_COLOR),
           "$=$",
           ("vec", [f"${_p(k)}\\cdot{_p(x)}$" for x in v], INK),
           "$=$",
           ("vec", [f"${la.fmt(x)}$" for x in kv], KV_COLOR)])

    # ---------- Step 2: lengths
    b.heading("Length (magnitude)")
    sq_v = " + ".join(f"{_p(x)}^2" for x in v)
    sq_kv = " + ".join(f"{_p(x)}^2" for x in kv)
    sum_v = float(np.sum(v ** 2))
    sum_kv = float(np.sum(kv ** 2))
    b.row([rf"$\Vert\mathbf{{v}}\Vert = \sqrt{{{sq_v}}}$"])
    b.row([rf"$= \sqrt{{{la.fmt(sum_v)}}} \approx {la.fmt(mv)}$"], indent=40)
    b.row([rf"$\Vert k\mathbf{{v}}\Vert = \sqrt{{{sq_kv}}}$"])
    b.row([rf"$= \sqrt{{{la.fmt(sum_kv)}}} \approx {la.fmt(mkv)}$"], indent=40)
    b.note("Shortcut rule:  ‖k·v‖ = |k| · ‖v‖")
    b.row([rf"$\left|{{{K}}}\right|\cdot {la.fmt(mv)} = {la.fmt(abs(k) * mv)}\ \ \checkmark$"])

    # ---------- Step 3: direction
    b.heading("Direction")
    if mv == 0:
        b.note("v is the zero vector, so it has no direction.", color=INK, size=11)
    else:
        if n == 2:
            b.row([rf"$\theta_{{\mathbf{{v}}}} = \mathrm{{atan2}}({la.fmt(v[1])},\ {la.fmt(v[0])})"
                   rf" = {la.angle_2d(v):.2f}^\circ$"])
            if k != 0:
                b.row([rf"$\theta_{{k\mathbf{{v}}}} = \mathrm{{atan2}}({la.fmt(kv[1])},\ {la.fmt(kv[0])})"
                       rf" = {la.angle_2d(kv):.2f}^\circ$"])
                b.note("k > 0  →  same angle" if k > 0 else "k < 0  →  angle turns by 180°")
        u = la.unit_vector(v)
        b.row([r"$\hat{\mathbf{u}} = \frac{\mathbf{v}}{\Vert\mathbf{v}\Vert} =$",
               ("vec", [f"${la.fmt(x)}$" for x in u], V_COLOR)])
        if k != 0:
            sign = "+1" if k > 0 else "-1"
            b.row([rf"$\hat{{\mathbf{{u}}}}_{{k\mathbf{{v}}}} = \mathrm{{sign}}(k)\,\hat{{\mathbf{{u}}}}"
                   rf" = {sign}\cdot\hat{{\mathbf{{u}}}} =$",
                   ("vec", [f"${la.fmt(x)}$" for x in la.unit_vector(kv)], KV_COLOR)])
        else:
            b.note("k = 0  →  k·v = 0 has no direction")

    # ---------- Step 4: geometric meaning
    b.heading("What it means geometrically")
    lines = [part.strip().capitalize() for part in la.effect_text(k).split(",")]
    if k == 0:
        lines.append("Every point shrinks onto the origin.")
    else:
        size = ("|k| > 1 → longer" if abs(k) > 1
                else "|k| < 1 → shorter" if abs(k) < 1 else "|k| = 1 → same length")
        way = "k > 0 → points the same way" if k > 0 else "k < 0 → flips to the other side"
        lines += [size, way, "k·v always stays on the line (span) of v"]
    b.result_box(lines, bold=len(lines) - (1 if k == 0 else 3))


if __name__ == "__main__":
    # quick check: python algebra_render.py  -> writes algebra_preview.png
    data, h, steps = render(-1.5, [3, 2])
    with open("algebra_preview.png", "wb") as f:
        f.write(base64.b64decode(data))
    print("height", h, "steps", steps)
