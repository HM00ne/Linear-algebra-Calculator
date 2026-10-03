"""
tab_bar.py
----------
The row of topic tabs, drawn on one canvas.

It replaces ttk.Notebook, which repaints its whole page area through an
off-screen bitmap on every step of a window resize (~13 ms each time).
A canvas of text items costs almost nothing, and lets the highlight slide
smoothly to the chosen tab.
"""

import tkinter as tk
import tkinter.font as tkfont

from animation import ease_out
from settings_panel import Tweened, round_rect
from themes import mix
from ui_scale import px

FONT = ("Segoe UI", 10)


class TabBar(tk.Canvas, Tweened):
    H = 38
    PAD = 18          # space left and right of each title
    GAP = 4

    def __init__(self, master, anim, on_select, surface="bg", height=None):
        """surface: the colour role behind the bar ("bg" for the main tabs,
        "panel" for a bar inside a panel); the chosen tab takes the other."""
        self.H = px(height or self.H)
        self.PAD, self.GAP = px(self.PAD), px(self.GAP)
        super().__init__(master, height=self.H, highlightthickness=0, bd=0)
        self.anim, self.on_select = anim, on_select
        self.surface = surface
        self.th = None
        self.font = tkfont.Font(family=FONT[0], size=FONT[1])
        self.titles, self.texts, self.spans = [], [], []
        self.current = 0
        self._pos = 0.0               # where the highlight is, in tab indexes (animated)
        self._hover = None
        self.line = self.create_line(0, self.H - 1, 0, self.H - 1)
        self.pill = self.create_polygon(0, 0, 0, 0, smooth=True, outline="")
        self.bar = self.create_rectangle(0, 0, 0, 0, outline="")
        self.bind("<Button-1>", self._click)
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda e: self._set_hover(None))
        self._width = 0
        self._size = FONT[1]
        self._probe, self._widths = {}, {}
        self.bind("<Configure>", self._on_resize)

    def add(self, title):
        self.titles.append(title)
        self.texts.append(self.create_text(0, self.H / 2 + 1, text=title, font=self.font))
        self._layout()

    def _on_resize(self, e):
        self.coords(self.line, 0, self.H - 1, e.width, self.H - 1)
        if e.width != self._width:
            self._width = e.width
            self._layout()

    def _text_widths(self, size):
        """Title widths at a font size, measured once and remembered.
        (Measuring with separate fonts matters: re-configuring the shown font,
        even to the same size, makes Tk re-measure every user of it, which
        cost ~11 ms on each step of a window resize.)"""
        key = (size, len(self.titles))
        if key not in self._widths:
            probe = self._probe.get(size)
            if probe is None:
                probe = self._probe[size] = tkfont.Font(family=FONT[0], size=size)
            self._widths[key] = [probe.measure(t) for t in self.titles]
        return self._widths[key]

    def _layout(self):
        """Place the titles; when they do not all fit, first squeeze the space
        around them, then use a smaller font."""
        avail = self._width or 10 ** 6
        size = FONT[1]
        while True:
            text_w = self._text_widths(size)
            spare = avail - sum(text_w) - self.GAP * (len(self.titles) - 1)
            pad = min(self.PAD, spare / (2 * len(self.titles))) if self.titles else self.PAD
            if pad >= px(6) or size <= 8:
                pad = max(pad, px(4))
                break
            size -= 1
        if size != self._size:              # only touch the real font when needed
            self._size = size
            self.font.configure(size=size)
        self.spans, x = [], 0
        for item, w in zip(self.texts, text_w):
            w += 2 * pad
            self.spans.append((x, x + w))
            self.coords(item, x + w / 2, self.H / 2 + 1)
            x += w + self.GAP
        self._paint()

    def select(self, i, animate=True):
        self.current = i
        if animate:
            self._tween("_pos", float(i), 280, ease_out)
        else:
            self._pos = float(i)
            self._paint()

    def apply(self, th):
        self.th = th
        self.configure(bg=th[self.surface])
        self._paint()

    # ---- mouse
    def _index_at(self, x):
        for i, (x0, x1) in enumerate(self.spans):
            if x0 <= x <= x1:
                return i
        return None

    def _click(self, e):
        i = self._index_at(e.x)
        if i is not None and i != self.current:
            self.on_select(i)

    def _motion(self, e):
        self._set_hover(self._index_at(e.x))

    def _set_hover(self, i):
        if i != self._hover:
            self._hover = i
            self.configure(cursor="hand2" if i is not None and i != self.current else "")
            self._paint()

    # ---- drawing
    def _span_at(self, pos):
        """x-range of the highlight at a fractional tab position."""
        i = min(int(pos), len(self.spans) - 1)
        j = min(i + 1, len(self.spans) - 1)
        t = pos - i
        (a0, a1), (b0, b1) = self.spans[i], self.spans[j]
        return a0 + (b0 - a0) * t, a1 + (b1 - a1) * t

    def _paint(self):
        th = self.th
        if th is None or not self.spans:
            return
        x0, x1 = self._span_at(self._pos)
        self.coords(self.pill, *round_rect(x0, px(3), x1, self.H + px(10), px(9)))
        self.itemconfigure(self.pill, fill=th["panel" if self.surface == "bg" else "bg"])
        self.coords(self.bar, x0 + px(10), px(3), x1 - px(10), px(6))
        self.itemconfigure(self.bar, fill=th["accent"])
        self.itemconfigure(self.line, fill=th["border"])
        for i, item in enumerate(self.texts):
            near = max(0.0, 1 - abs(self._pos - i))      # colour follows the highlight
            base = th["text"] if i == self._hover else th["muted"]
            self.itemconfigure(item, fill=mix(base, th["accent"], near))
        self.tag_raise(self.pill)
        self.tag_raise(self.bar)
        for item in self.texts:
            self.tag_raise(item)
