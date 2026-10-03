"""
splash.py
---------
Animated start-up screen for the Linear Algebra Calculator.

Sequence (about 2.5 s, click anywhere to skip):
  1. grid and axes fade in
  2. a blue vector v grows out of the origin
  3. its span line appears and a red k·v sweeps through
     k = 1 → 2.2 → -1.3 → 0.5 → 1.6  (stretch, flip, shrink)
  4. the title types itself, subtitle fades in, progress bar fills
  5. the splash fades out and the main window fades in
"""

import tkinter as tk
import math
import time

from animation import ease, FPS
import ui_scale

BG = "#0b1220"
GRID = "#1c2a44"
AXIS = "#3b4d6e"
BLUE = "#3b82f6"
BLUE_GLOW = "#1e3a6e"
RED = "#ef4444"
RED_GLOW = "#5c1f28"
SPAN = "#64748b"
WHITE = "#f8fafc"
MUTED = "#94a3b8"

FRAME_MS = 16
TOTAL = 160           # length of the main animation, in 60 fps frames
FADE = 14             # frames of the fade-out


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def keyframes(f, frames):
    """Interpolate value at frame f from [(frame, value), ...] with easing."""
    if f <= frames[0][0]:
        return frames[0][1]
    for (f0, v0), (f1, v1) in zip(frames, frames[1:]):
        if f <= f1:
            return v0 + (v1 - v0) * ease((f - f0) / (f1 - f0))
    return frames[-1][1]


class Splash(tk.Toplevel):
    W, H = 640, 390
    OX, OY = 320, 152           # origin on the canvas
    V = (92, -50)               # vector v in pixels
    K_FRAMES = [(62, 1.0), (82, 2.2), (104, -1.3), (122, 0.5), (142, 1.6)]
    TITLE = "Linear Algebra Calculator"
    MESSAGES = [(0, "Preparing canvas…"), (40, "Drawing vectors…"),
                (90, "Scaling by k…"), (140, "Ready")]

    def __init__(self, master, on_done):
        super().__init__(master)
        self.on_done = on_done
        self.frame = 0
        self._t0 = time.perf_counter()
        self._grid_done = False
        self._shown = False
        self._finished = False
        self.overrideredirect(True)
        self.configure(bg=BG)
        # everything is laid out at 100 % scaling and scaled to the screen
        self.s = s = ui_scale.S
        w, h = round(self.W * s), round(self.H * s)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self._set_alpha(0.0)

        c = self.c = tk.Canvas(self, width=w, height=h, bg=BG, highlightthickness=0)
        c.pack()
        self._build()
        if s != 1:
            c.scale("all", 0, 0, s, s)
        self.bind("<Button-1>", lambda e: self._skip())
        c.bind("<Button-1>", lambda e: self._skip())
        self.lift()
        self.after(FRAME_MS, self._tick)

    # ---------------------------------------------------------------------
    def _set_alpha(self, a):
        try:
            self.attributes("-alpha", a)
        except tk.TclError:
            pass

    def _coords(self, item, *xy):
        """Move an item using 100 %-scaling coordinates."""
        self.c.coords(item, *(v * self.s for v in xy))

    def _build(self):
        cv = self.c
        k = self.s                       # line widths / arrow heads are not scaled by
                                         # Canvas.scale, so size them here
        x0, x1, y0, y1 = 70, 570, 26, 270
        self.grid_lines = []
        for x in range(x0, x1 + 1, 25):
            self.grid_lines.append(cv.create_line(x, y0, x, y1, fill=BG))
        for y in range(self.OY - 125, y1 + 1, 25):
            if y >= y0:
                self.grid_lines.append(cv.create_line(x0, y, x1, y, fill=BG))
        self.axes = [cv.create_line(x0, self.OY, x1, self.OY, fill=BG, width=1.5 * k),
                     cv.create_line(self.OX, y0, self.OX, y1, fill=BG, width=1.5 * k)]

        L = math.hypot(*self.V)
        ux, uy = self.V[0] / L, self.V[1] / L
        r = 250
        self.span = cv.create_line(self.OX - ux * r, self.OY - uy * r,
                                   self.OX + ux * r, self.OY + uy * r,
                                   fill=BG, dash=(round(6 * k), round(5 * k)))

        arrow = dict(arrow=tk.LAST, capstyle=tk.ROUND)
        o = (self.OX, self.OY, self.OX, self.OY)

        def shape(a, b, c):
            return (a * k, b * k, c * k)

        self.kv_glow = cv.create_line(*o, fill=BG, width=10 * k, **arrow,
                                      arrowshape=shape(18, 20, 8))
        self.kv = cv.create_line(*o, fill=BG, width=3 * k, **arrow, arrowshape=shape(13, 15, 5))
        self.v_glow = cv.create_line(*o, fill=BG, width=12 * k, **arrow,
                                     arrowshape=shape(20, 22, 9))
        self.v = cv.create_line(*o, fill=BG, width=5 * k, **arrow, arrowshape=shape(15, 17, 6))
        self.dot = cv.create_oval(self.OX - 4, self.OY - 4, self.OX + 4, self.OY + 4,
                                  fill=BG, outline="")
        self.v_label = cv.create_text(0, 0, text="v", fill=BG, font=("Segoe UI", 13, "bold"))
        self.k_label = cv.create_text(0, 0, text="", fill=BG, font=("Consolas", 11, "bold"))

        self.title = cv.create_text(self.W // 2, 304, text="", fill=WHITE,
                                    font=("Segoe UI", 22, "bold"))
        self.subtitle = cv.create_text(self.W // 2, 334, text="vectors · scalars · geometry",
                                       fill=BG, font=("Segoe UI", 11))
        cv.create_rectangle(70, 362, 570, 366, fill=GRID, outline="")
        self.bar = cv.create_rectangle(70, 362, 70, 366, fill=BLUE, outline="")
        self.msg = cv.create_text(570, 378, text="", fill=MUTED, anchor="e",
                                  font=("Segoe UI", 8))

    # ---------------------------------------------------------------------
    def _tick(self):
        if self._finished:
            return
        # the frame number comes from the clock (and may be fractional), so a
        # slow frame never slows the animation down, it just skips ahead
        f = self.frame = min(TOTAL, (time.perf_counter() - self._t0) * FPS)
        cv = self.c

        # window fade-in
        if not self._shown:
            self._set_alpha(min(1.0, f / 12))
            self._shown = f >= 12

        # 1. grid + axes fade in
        if not self._grid_done:          # ~40 items: stop touching them once faded in
            g = ease(f / 30)
            grid, axis = lerp_color(BG, GRID, g), lerp_color(BG, AXIS, g)
            for line in self.grid_lines:
                cv.itemconfigure(line, fill=grid)
            for line in self.axes:
                cv.itemconfigure(line, fill=axis)
            cv.itemconfigure(self.dot, fill=lerp_color(BG, WHITE, g))
            self._grid_done = g >= 1

        # 2. v grows from the origin
        t = ease((f - 22) / 32)
        if t > 0:
            ex, ey = self.OX + self.V[0] * t, self.OY + self.V[1] * t
            for item, col in ((self.v_glow, BLUE_GLOW), (self.v, BLUE)):
                self._coords(item, self.OX, self.OY, ex, ey)
                cv.itemconfigure(item, fill=col)
            cv.itemconfigure(self.v_label, fill=lerp_color(BG, BLUE, t))
            self._coords(self.v_label, ex - 14, ey - 12)

        # 3. span line + k·v sweeping
        cv.itemconfigure(self.span, fill=lerp_color(BG, SPAN, ease((f - 52) / 14)))
        if f >= 62:
            k = keyframes(f, self.K_FRAMES)
            ex, ey = self.OX + self.V[0] * k, self.OY + self.V[1] * k
            fade = ease((f - 62) / 8)
            for item, col in ((self.kv_glow, RED_GLOW), (self.kv, RED)):
                self._coords(item, self.OX, self.OY, ex, ey)
                cv.itemconfigure(item, fill=lerp_color(BG, col, fade))
            cv.itemconfigure(self.k_label, text=f"k = {k:+.2f}",
                             fill=lerp_color(BG, RED, fade))
            side = 1 if k >= 0 else -1
            self._coords(self.k_label, ex + 46 * side, ey + 4)
            cv.tag_raise(self.v)

        # 4. title typing, subtitle, progress
        n = int(len(self.TITLE) * max(0.0, min(1.0, (f - 30) / 70)))
        cv.itemconfigure(self.title, text=self.TITLE[:n] + ("▍" if n < len(self.TITLE) and int(f) % 16 < 10 else ""))
        cv.itemconfigure(self.subtitle, fill=lerp_color(BG, MUTED, ease((f - 100) / 20)))
        p = ease(min(1.0, f / TOTAL))
        self._coords(self.bar, 70, 362, 70 + 500 * p, 366)
        for start, text in self.MESSAGES:
            if f >= start:
                cv.itemconfigure(self.msg, text=text)

        if f < TOTAL:
            self.after(FRAME_MS, self._tick)
        else:
            self._fade_start = time.perf_counter()
            self._fade_out()

    def _fade_out(self):
        i = (time.perf_counter() - self._fade_start) * FPS
        if i < FADE:
            self._set_alpha(1 - i / FADE)
            self.after(FRAME_MS, self._fade_out)
        else:
            self._finish()

    def _skip(self):
        if not self._finished and self.frame < TOTAL:
            self._t0 -= (TOTAL - self.frame) / FPS     # jump to the end, then fade

    def _finish(self):
        if self._finished:
            return
        self._finished = True
        self.destroy()
        self.on_done()
