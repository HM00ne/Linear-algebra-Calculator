"""
settings_panel.py
-----------------
The slide-in Settings panel (opened with the ⚙ Settings button in the header)
and its small animated controls:

  ThemeCard   – a live mini preview of a theme; lifts and glows on hover
  Segmented   – a row of options with a pill that slides to the chosen one
  Toggle      – an on/off switch whose knob slides and whose colour fades

All animation goes through the app's shared Animator, so it respects the
Motion setting and never runs a second timer.
"""

import tkinter as tk

import themes
from animation import ease, ease_out
from themes import mix
import ui_scale
from ui_scale import px

FONT = "Segoe UI"


def round_rect(x0, y0, x1, y1, r):
    """Points for a rounded rectangle, drawn as a polygon with smooth=True."""
    return (x0 + r, y0, x0 + r, y0, x1 - r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
            x1, y0 + r, x1, y1 - r, x1, y1 - r, x1, y1, x1 - r, y1, x1 - r, y1,
            x0 + r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y1 - r, x0, y0 + r,
            x0, y0 + r, x0, y0)


class Tweened:
    """Mixin: animate one float attribute (0..1) and repaint each frame."""

    def _tween(self, attr, target, ms, curve=ease):
        job = f"_{attr}_job"
        self.anim.cancel(getattr(self, job, None))
        start = getattr(self, attr)

        def frame(p):
            setattr(self, attr, start + (target - start) * curve(p))
            self._paint()

        setattr(self, job, self.anim.start(ms, frame))


class ThemeCard(tk.Canvas, Tweened):
    W, H = 150, 112

    def __init__(self, master, name, on_pick, anim):
        super().__init__(master, width=px(self.W), height=px(self.H), highlightthickness=0,
                         bd=0, cursor="hand2")
        self.name, self.anim = name, anim
        self.th = None
        self.selected = False
        self._hover = 0.0
        self._lift = 0.0                # how far the preview is currently raised (px)
        self._build(themes.get(name))
        if ui_scale.S != 1:              # drawn at 100 %, scaled to the screen in one go
            self.scale("all", 0, 0, ui_scale.S, ui_scale.S)
        self.bind("<Enter>", lambda e: self._tween("_hover", 1.0, 160))
        self.bind("<Leave>", lambda e: self._tween("_hover", 0.0, 220))
        self.bind("<Button-1>", lambda e: on_pick(name))

    def _build(self, t):
        """A miniature of the app window, drawn in the card's own colours."""
        W = self.W
        P = "preview"
        self.frame = self.create_polygon(round_rect(2, 2, W - 2, self.H - 2, 12),
                                         smooth=True, width=2)
        x0, y0, x1, y1 = 10, 10, W - 10, 76
        self.create_rectangle(x0, y0, x1, y1, fill=t["bg"], outline="", tags=P)
        a, b = t["header"]
        for i in range(8):                                   # header gradient
            xa = x0 + (x1 - x0) * i / 8
            self.create_rectangle(xa, y0, xa + (x1 - x0) / 8 + 1, y0 + 12,
                                  fill=mix(a, b, i / 7), outline="", tags=P)
        self.create_line(x0, y0 + 12, x1, y0 + 12, fill=t["accent"], width=2, tags=P)
        # side panel with a couple of "controls"
        self.create_rectangle(x0 + 4, y0 + 17, x0 + 36, y1 - 4, fill=t["panel"],
                              outline="", tags=P)
        for j in range(3):
            yy = y0 + 23 + j * 9
            self.create_line(x0 + 8, yy, x0 + 30 - j * 5, yy, fill=t["muted"], width=2, tags=P)
        self.create_rectangle(x0 + 8, y1 - 14, x0 + 32, y1 - 9, fill=t["accent"],
                              outline="", tags=P)
        # plot with grid, axes, v and k·v
        px0, py0, px1, py1 = x0 + 40, y0 + 17, x1 - 4, y1 - 4
        self.create_rectangle(px0, py0, px1, py1, fill=t["plot_bg"], outline="", tags=P)
        for gx in range(int(px0) + 10, int(px1), 12):
            self.create_line(gx, py0, gx, py1, fill=mix(t["plot_bg"], t["grid"], 0.45), tags=P)
        ox, oy = (px0 + px1) / 2, (py0 + py1) / 2
        self.create_line(px0, oy, px1, oy, fill=t["axis"], tags=P)
        self.create_line(ox, py0, ox, py1, fill=t["axis"], tags=P)
        self.create_line(ox, oy, ox - 26, oy + 16, fill=t["kv"], width=2, arrow="last",
                         arrowshape=(6, 7, 3), tags=P)
        self.create_line(ox, oy, ox + 16, oy - 10, fill=t["v"], width=3, arrow="last",
                         arrowshape=(6, 7, 3), tags=P)

        self.label = self.create_text(12, 94, text=self.name, anchor="w",
                                      font=(FONT, 9, "bold"))
        self.check_bg = self.create_oval(W - 27, 87, W - 13, 101, outline="")
        self.check = self.create_text(W - 20, 94, text="✓", font=(FONT, 8, "bold"))

    def apply(self, th, selected):
        self.th, self.selected = th, selected
        self.configure(bg=th["panel"])
        self._paint()

    def _paint(self):
        th, h = self.th, self._hover
        if th is None:
            return
        glow = 1.0 if self.selected else 0.55 * h
        self.itemconfigure(self.frame, fill=mix(th["panel"], th["soft"], h * 0.7),
                           outline=mix(th["border"], th["accent"], glow))
        self.itemconfigure(self.label, fill=th["accent"] if self.selected else th["text"])
        state = "normal" if self.selected else "hidden"
        self.itemconfigure(self.check_bg, fill=th["accent"], state=state)
        self.itemconfigure(self.check, fill=th["accent_fg"], state=state)
        lift = px(3) * h                       # preview floats up a little on hover
        self.move("preview", 0, self._lift - lift)
        self._lift = lift


class Segmented(tk.Canvas, Tweened):
    def __init__(self, master, options, value, on_change, anim, width=310, height=34):
        width, height = px(width), px(height)
        super().__init__(master, width=width, height=height, highlightthickness=0,
                         bd=0, cursor="hand2")
        self.options, self.on_change, self.anim = list(options), on_change, anim
        self.value = value
        self.th = None
        self.w, self.h = width, height
        self.seg = (width - px(8)) / len(self.options)
        self._pos = float(self.options.index(value))      # pill position, in segments
        self.track = self.create_polygon(round_rect(1, 1, width - 1, height - 1, px(11)),
                                         smooth=True, outline="")
        self.pill = self.create_polygon(0, 0, 0, 0, smooth=True, outline="")
        self.labels = [self.create_text(px(4) + self.seg * (i + 0.5), height / 2, text=o,
                                        font=(FONT, 9, "bold"))
                       for i, o in enumerate(self.options)]
        self.bind("<Button-1>", self._click)

    def _click(self, e):
        i = min(len(self.options) - 1, max(0, int((e.x - px(4)) // self.seg)))
        if self.options[i] != self.value:
            self.value = self.options[i]
            self._tween("_pos", float(i), 260, ease_out)
            self.on_change(self.value)

    def apply(self, th):
        self.th = th
        self.configure(bg=th["panel"])
        self._paint()

    def _paint(self):
        th = self.th
        if th is None:
            return
        self.itemconfigure(self.track, fill=th["button"])
        x = px(4) + self._pos * self.seg
        self.coords(self.pill, *round_rect(x, px(4), x + self.seg, self.h - px(4), px(8)))
        self.itemconfigure(self.pill, fill=th["accent"])
        for i, item in enumerate(self.labels):     # text colour follows the pill
            near = max(0.0, 1 - abs(self._pos - i))
            self.itemconfigure(item, fill=mix(th["text"], th["accent_fg"], near))


class Toggle(tk.Canvas, Tweened):
    W, H = 46, 26

    def __init__(self, master, value, on_change, anim):
        self.W, self.H = px(self.W), px(self.H)
        super().__init__(master, width=self.W, height=self.H, highlightthickness=0,
                         bd=0, cursor="hand2")
        self.value, self.on_change, self.anim = value, on_change, anim
        self.th = None
        self._on = 1.0 if value else 0.0
        self.track = self.create_polygon(round_rect(1, 1, self.W - 1, self.H - 1, px(12)),
                                         smooth=True, outline="")
        self.knob = self.create_oval(0, 0, 0, 0, outline="")
        self.bind("<Button-1>", self._click)

    def _click(self, _e):
        self.value = not self.value
        self._tween("_on", 1.0 if self.value else 0.0, 200, ease_out)
        self.on_change(self.value)

    def apply(self, th):
        self.th = th
        self.configure(bg=th["panel"])
        self._paint()

    def _paint(self):
        th = self.th
        if th is None:
            return
        self.itemconfigure(self.track, fill=mix(th["button"], th["accent"], self._on))
        m = px(4)
        x = m + self._on * (self.W - self.H)
        self.coords(self.knob, x, m, x + self.H - 2 * m, self.H - m)
        self.itemconfigure(self.knob, fill="#ffffff" if self._on > 0.5 or not th["dark"]
                           else th["muted"])


class SettingsPanel(tk.Frame):
    WIDTH = 352

    def __init__(self, app):
        super().__init__(app, bd=0, highlightthickness=0)
        self.app, self.anim = app, app.anim
        self.th = None
        self.is_open = False
        self._shown = 0.0            # 0 = hidden off the right edge, 1 = fully in
        self._slide = None
        self._text, self._muted = [], []

        self.edge = tk.Frame(self, width=1)
        self.edge.pack(side="left", fill="y")
        body = tk.Frame(self, padx=18, pady=14)
        body.pack(side="left", fill="both", expand=True)

        head = tk.Frame(body)
        head.pack(fill="x")
        self.title = tk.Label(head, text="Settings", font=(FONT, 15, "bold"))
        self.title.pack(side="left")
        self.close_btn = tk.Label(head, text="✕", font=(FONT, 12), width=3, cursor="hand2")
        self.close_btn.pack(side="right")
        self.close_btn.bind("<Button-1>", lambda e: self.close())
        self.close_btn.bind("<Enter>", lambda e: self.close_btn.configure(bg=self.th["soft"]))
        self.close_btn.bind("<Leave>", lambda e: self.close_btn.configure(bg=self.th["panel"]))

        self._section(body, "APPEARANCE", "Colour theme")
        grid = tk.Frame(body)
        grid.pack(fill="x", pady=(4, 0))
        self.cards = {}
        for i, name in enumerate(themes.THEMES):
            card = ThemeCard(grid, name, app.apply_theme, self.anim)
            card.grid(row=i // 2, column=i % 2, padx=(0, 10) if i % 2 == 0 else 0,
                      pady=5, sticky="w")
            self.cards[name] = card

        self._section(body, "MOTION", "Animation speed")
        self.motion = Segmented(body, themes.MOTION, app.settings["motion"],
                                app.set_motion, self.anim)
        self.motion.pack(anchor="w", pady=(6, 0))
        row = tk.Frame(body)
        row.pack(fill="x", pady=(14, 0))
        lbl = tk.Label(row, text="Play intro animation on start", font=(FONT, 10))
        lbl.pack(side="left")
        self._text.append(lbl)
        self.splash = Toggle(row, app.settings["splash"], app.set_splash, self.anim)
        self.splash.pack(side="right")

        hint = tk.Label(body, font=(FONT, 8), justify="left", anchor="w",
                        text="Shortcuts\n"
                             "Ctrl+1 … Ctrl+5   colour theme\n"
                             "Ctrl+Tab   next tab        Ctrl+,   settings\n"
                             "Ctrl+0 / Home / double-click   reset the plot view\n"
                             "Plot: drag to move or rotate, scroll to zoom")
        hint.pack(side="bottom", fill="x")
        self._muted.append(hint)

    def _section(self, parent, caps, title):
        a = tk.Label(parent, text=caps, font=(FONT, 8, "bold"), anchor="w")
        a.pack(fill="x", pady=(18, 0))
        b = tk.Label(parent, text=title, font=(FONT, 10), anchor="w")
        b.pack(fill="x")
        self._muted.append(a)
        self._text.append(b)

    # ------------------------------------------------------------------ theme
    def apply_theme(self, th):
        self.th = th
        stack = [self]
        while stack:                                  # every plain frame/label
            w = stack.pop()
            stack.extend(w.winfo_children())
            if isinstance(w, (tk.Frame, tk.Label)):
                w.configure(bg=th["panel"])
        self.edge.configure(bg=th["border"])
        self.title.configure(fg=th["text"])
        self.close_btn.configure(fg=th["muted"])
        for w in self._text:
            w.configure(fg=th["text"])
        for w in self._muted:
            w.configure(fg=th["muted"])
        for name, card in self.cards.items():
            card.apply(th, name == th["name"])
        self.motion.apply(th)
        self.splash.apply(th)

    # ------------------------------------------------------------- slide in/out
    def toggle(self):
        self.close() if self.is_open else self.open()

    def open(self):
        if self.is_open:
            return
        self.is_open = True
        self.lift()
        self._animate(1.0, 340)
        self.app.on_settings_toggled(True)

    def close(self):
        if not self.is_open:
            return
        self.is_open = False
        self._animate(0.0, 240)
        self.app.on_settings_toggled(False)

    def _animate(self, target, ms):
        self.anim.cancel(self._slide)
        start = self._shown

        def frame(p):
            self._shown = start + (target - start) * ease_out(p)
            self._place()

        def done():
            if target == 0:
                self.place_forget()

        self._slide = self.anim.start(ms, frame, done)

    def _place(self):
        top = self.app.header.winfo_height()
        bottom = self.app.status_lbl.winfo_height()
        width = px(self.WIDTH)
        self.place(in_=self.app, relx=1.0, x=round(width * (1 - self._shown)), y=top,
                   anchor="ne", width=width, relheight=1.0, height=-(top + bottom))
