"""
Linear Algebra Calculator
=========================
A desktop application (Tkinter + Matplotlib) that shows vector operations
geometrically, with the full working written out step by step and the
same calculation as runnable Python code.

Run:
    pip install numpy matplotlib
    python vector_app.py              # with animated start-up
    python vector_app.py --no-splash  # skip the intro

Tabs (one file each):
    tab_scalar.py      – scalar × vector
    tab_dot.py         – dot product and its properties
    tab_cauchy.py      – Cauchy–Schwarz inequality
    tab_length.py      – vector length, unit vector
    tab_hadamard.py    – Hadamard (element-wise) product
    tab_rowcol.py      – row and column vectors, inner and outer product
    tab_notation.py    – product notations side by side
    tab_two.py         – everything between two vectors (sum, distance, angle, cross)
    tab_cosines.py     – law of cosines
    tab_complex.py     – complex numbers as vectors

Shared pieces:
    tab_base.py        – what every tab has in common (layout, rendering, steps)
    geo_scene.py       – the vector picture used by the tabs
    plot_scene.py      – the scalar tab's picture
    linalg_core.py     – the calculations
    algebra_render.py  – draws the algebraic working (vectors, roots, angles)
    animation.py       – shared time-based animation clock
    themes.py          – colour themes (⚙ Settings)
    settings_panel.py  – the slide-in settings panel
    window_chrome.py   – themed Windows title bar, fast resizing
    ui_scale.py        – sharp rendering on scaled / 1920×1080 screens
    plot_nav.py        – mouse control of the plots (move, zoom, reset)
    splash.py          – animated start-up screen
    vector_app.py      – this file, the application window
"""

import sys
import time
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

import matplotlib
matplotlib.use("TkAgg")

import themes
from animation import Animator, ease, ease_out, enable_precise_timers
from settings_panel import SettingsPanel, Tweened, round_rect
from splash import Splash
from tab_base import shutdown_worker
from tab_cauchy import CauchySchwarzTab
from tab_complex import ComplexTab
from tab_cosines import LawOfCosinesTab
from tab_dot import DotProductTab
from tab_hadamard import HadamardTab
from tab_length import VectorLengthTab
from tab_notation import NotationTab
from tab_rowcol import RowColumnTab
from tab_scalar import ScalarMultiplicationTab
from tab_two import TwoVectorsTab
from tab_bar import TabBar
import ui_scale
from ui_scale import px
from window_chrome import smooth_resizing, style_titlebar

HEADER_H = 54
TABS = (ScalarMultiplicationTab, DotProductTab, CauchySchwarzTab, VectorLengthTab,
        HadamardTab, RowColumnTab, NotationTab, TwoVectorsTab, LawOfCosinesTab, ComplexTab)


class LinearAlgebraApp(tk.Tk, Tweened):
    def __init__(self):
        ui_scale.enable_dpi_awareness()       # must happen before Tk starts
        super().__init__()
        ui_scale.init(self)
        self._end_precise_timers = enable_precise_timers()
        self._last_input = 0.0
        self.anim = Animator(self)
        self.settings = themes.load_settings()
        self.anim.speed = themes.MOTION[self.settings["motion"]]
        self.th = themes.get(self.settings["theme"])
        self._shine_id = None
        self._btn_hover = self._btn_active = self._gear = 0.0   # settings button tweens
        self._hover_target = 0.0
        self._btn_box = (0, 0, 0, 0)

        self.title("Linear Algebra Calculator")
        self._initial_geometry()
        self.configure(bg=self.th["bg"])
        self._style(self.th)

        self.header = tk.Canvas(self, height=px(HEADER_H), highlightthickness=0)
        self.header.pack(fill="x")
        self._build_header()

        self.status_var = tk.StringVar()
        # plain tk.Label: it spans the window width, so it repaints on every resize
        self._status_error = False
        self.status_lbl = tk.Label(self, textvariable=self.status_var, anchor="w",
                                   padx=12, pady=5, font=("Segoe UI", 10), bd=0)
        self.status_lbl.pack(fill="x", side="bottom")

        # one tab per topic; each builds its widgets the first time it is opened.
        # A canvas tab bar + plain frame instead of ttk.Notebook: much cheaper
        # to repaint while the window is being resized (see tab_bar.py).
        self.tab_bar = TabBar(self, self.anim, self.select_tab)
        self.tab_bar.pack(fill="x", padx=8, pady=(6, 0))
        self.tab_host = tk.Frame(self, bd=0, highlightthickness=0)
        self.tab_host.pack(fill="both", expand=True, padx=8)
        self.current_tab = None
        ctx = SimpleNamespace(status=self.set_status, anim=self.anim, theme=self.th,
                              is_current=lambda tab: tab is self.current_tab)
        self.tabs = [cls(self.tab_host, ctx) for cls in TABS]
        for tab in self.tabs:
            self.tab_bar.add(tab.title.strip())
        self.scalar_tab = self.tabs[0]
        self._color_tab_area()
        self.set_status("")
        self.select_tab(0, animate=False)
        smooth_resizing(self)

        self.settings_panel = SettingsPanel(self)       # created last: floats on top
        self.settings_panel.apply_theme(self.th)
        self._style_titlebar()

        for i, name in enumerate(themes.THEMES, 1):
            self.bind_all(f"<Control-Key-{i}>", lambda e, n=name: self.apply_theme(n))
        self.bind_all("<Control-comma>", lambda e: self.settings_panel.toggle())
        self.bind_all("<Control-Tab>", lambda e: self._step_tab(1))
        self.bind_all("<Control-Shift-Tab>", lambda e: self._step_tab(-1))
        self.bind_all("<Control-Key-0>", lambda e: self.current_tab.reset_view())
        self.bind_all("<Home>", self._home_key)
        for seq in ("<KeyPress>", "<ButtonPress>", "<MouseWheel>", "<B1-Motion>"):
            self.bind_all(seq, self._note_input, add="+")
        self.bind("<Escape>", lambda e: self.settings_panel.close())
        self.bind("<Button-1>", self._click_outside)     # any click in the window
        self.protocol("WM_DELETE_WINDOW", self._quit)

    # ---- settings
    def apply_theme(self, name):
        """Switch theme with a short, soft dim of the window. The expensive
        part (restyling every widget, re-rendering the plot and the maths)
        happens while dimmed, so the change reads as one smooth transition."""
        new = themes.get(name)
        if new["name"] == self.th["name"]:
            return
        old, self.th = self.th, new
        self.settings["theme"] = new["name"]
        themes.save_settings(self.settings)
        self.settings_panel.apply_theme(new)            # the clicked card reacts at once

        def swap():
            self._style(new)
            self.configure(bg=new["bg"])
            self._color_tab_area()
            self.set_status(self.status_var.get(), self._status_error)
            for tab in self.tabs:
                tab.apply_theme(new)         # unopened tabs just remember it
            self._style_titlebar()
            self.update_idletasks()                      # relayout + plot redraw now
            self.anim.start(220, lambda p: self._set_alpha(0.9 + 0.1 * p))

        self.anim.start(110, lambda p: self._set_alpha(1 - 0.1 * p), swap)
        self.anim.start(450, lambda p: self._paint_header(old, new, ease(p)))
        self._shine()

    def select_tab(self, index, animate=True):
        tab = self.tabs[index]
        if tab is self.current_tab:
            return
        if self.current_tab is not None:
            self.current_tab.pack_forget()
        self.current_tab = tab
        tab.pack(fill="both", expand=True)
        self.tab_bar.select(index, animate)
        tab.on_show()

    def _initial_geometry(self):
        """Fit the window to the screen: about 92 % of it, centred (1920×1080
        and larger get a roomy window, small laptops still get a usable one)."""
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        min_w, min_h = min(px(1240), sw - px(40)), min(px(680), sh - px(80))
        w = max(min_w, min(int(sw * 0.92), px(1720)))
        h = max(min_h, min(int(sh * 0.88), px(1000)))
        x, y = (sw - w) // 2, max(0, (sh - h) // 2 - px(16))
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(min_w, min_h)

    def _home_key(self, event):
        if not isinstance(event.widget, (tk.Entry, ttk.Entry)):   # Home still works in fields
            self.current_tab.reset_view()

    def _note_input(self, _event):
        self._last_input = time.perf_counter()

    # ---- building the other tabs in the background
    def _prebuild_next(self):
        """Opening a tab for the first time used to take 0.3-0.9 s, mostly its
        first plot draw. While the app is idle, build one unopened tab at a
        time behind the current one (same size, so the work carries over) and
        draw it there; opening it later is then as quick as a visited tab."""
        pending = [t for t in self.tabs if not t.built]
        if not pending:
            return
        if self.anim.busy() or time.perf_counter() - self._last_input < 0.6:
            self.after(400, self._prebuild_next)       # the user is busy: wait
            return
        tab = pending[0]
        tab.ensure_built()
        tab.place(in_=self.tab_host, x=0, y=0, relwidth=1, relheight=1)
        tab.lower(self.current_tab)                    # hidden behind the visible tab
        self.after(40, self._prewarm, tab)

    def _prewarm(self, tab):
        if tab is not self.current_tab:
            self.update_idletasks()                    # give it its real size
            tab.prewarm()
            tab.place_forget()
        self.after(200, self._prebuild_next)

    def _step_tab(self, step):
        i = self.tabs.index(self.current_tab)
        self.select_tab((i + step) % len(self.tabs))
        return "break"

    def _color_tab_area(self):
        self.tab_bar.apply(self.th)
        self.tab_host.configure(bg=self.th["bg"])

    def set_motion(self, name):
        self.settings["motion"] = name
        self.anim.speed = themes.MOTION[name]
        themes.save_settings(self.settings)
        self.set_status(f"Animation speed: {name}")

    def set_splash(self, on):
        self.settings["splash"] = on
        themes.save_settings(self.settings)

    def on_settings_toggled(self, is_open):
        self._tween("_btn_active", 1.0 if is_open else 0.0, 240)
        self._tween("_gear", 120.0 if is_open else 0.0, 420, ease_out)

    def _click_outside(self, event):
        panel = self.settings_panel
        if (panel.is_open and event.widget is not self.header
                and not str(event.widget).startswith(str(panel))):
            panel.close()

    def _style_titlebar(self):
        th = self.th
        style_titlebar(self, th["dark"], caption=th["header"][0], text=th["header_fg"],
                       border=th["header"][0])

    # ---- header: gradient bar, built once; resizing only moves the pieces
    def _build_header(self):
        c = self.header
        self._grad = [c.create_rectangle(0, 0, 0, 0, outline="") for _ in range(60)]
        self._shine_items = [c.create_rectangle(-px(200), 0, -px(170), px(50), outline="")
                             for _ in range(3)]          # sits under the text
        title = "Linear Algebra Calculator"
        font = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self._h_title = c.create_text(px(20), px(27), text=title, anchor="w", font=font)
        self._h_sub = c.create_text(px(20) + font.measure(title) + px(16), px(29), anchor="w",
                                    text="visualise vector operations geometrically",
                                    font=("Segoe UI", 10))
        self._h_line = c.create_line(0, px(52), 0, px(52), width=px(3))
        # ⚙ Settings button: a pill on the right of the header
        self._btn = c.create_polygon(0, 0, 0, 0, smooth=True, outline="")
        self._btn_gear = c.create_text(0, 0, text="⚙", font=("Segoe UI Symbol", 14))
        self._btn_text = c.create_text(0, 0, text="Settings", anchor="w",
                                       font=("Segoe UI", 10, "bold"))
        self._header_w = 0
        self._paint_header(self.th, self.th, 1.0)
        c.bind("<Configure>", self._layout_header)
        c.bind("<Motion>", lambda e: self._set_btn_hover(self._in_btn(e)))
        c.bind("<Leave>", lambda e: self._set_btn_hover(False))
        c.bind("<Button-1>", lambda e: self._in_btn(e) and self.settings_panel.toggle())

    def _layout_header(self, event):
        w = event.width
        if w == self._header_w:
            return
        self._header_w = w
        c = self.header
        n = len(self._grad)
        for i, item in enumerate(self._grad):
            c.coords(item, w * i / n, 0, w * (i + 1) / n + 1, px(HEADER_H))
        c.coords(self._h_line, 0, px(52), w, px(52))
        x1 = w - px(16)
        x0 = x1 - px(112)
        self._btn_box = (x0, px(11), x1, px(42))
        c.coords(self._btn, *round_rect(*self._btn_box, px(15)))
        c.coords(self._btn_gear, x0 + px(21), px(26))
        c.coords(self._btn_text, x0 + px(38), px(26))

    def _paint_header(self, old, new, t):
        """Header colours, blended from theme `old` to theme `new` (t = 0..1)."""
        c = self.header
        a = themes.mix(old["header"][0], new["header"][0], t)
        b = themes.mix(old["header"][1], new["header"][1], t)
        n = len(self._grad)
        for i, item in enumerate(self._grad):
            c.itemconfigure(item, fill=themes.mix(a, b, i / (n - 1)))
        mid = themes.mix(a, b, 0.5)
        glow, core = themes.mix(mid, "#ffffff", 0.15), themes.mix(mid, "#ffffff", 0.3)
        for item, col in zip(self._shine_items, (glow, core, glow)):
            c.itemconfigure(item, fill=col)
        self._hdr_b = b
        self._hdr_fg = themes.mix(old["header_fg"], new["header_fg"], t)
        c.itemconfigure(self._h_title, fill=self._hdr_fg)
        c.itemconfigure(self._h_sub, fill=themes.mix(old["header_sub"], new["header_sub"], t))
        c.itemconfigure(self._h_line, fill=themes.mix(old["accent"], new["accent"], t))
        c.configure(bg=a)
        self._paint()

    def _paint(self):
        """Settings button: hover brightens it, open tints it and turns the gear."""
        c = self.header
        fill = themes.mix(self._hdr_b, "#ffffff", 0.12 + 0.12 * self._btn_hover)
        fill = themes.mix(fill, self.th["accent"], 0.55 * self._btn_active)
        c.itemconfigure(self._btn, fill=fill)
        c.itemconfigure(self._btn_gear, fill=self._hdr_fg, angle=self._gear)
        c.itemconfigure(self._btn_text, fill=self._hdr_fg)

    def _in_btn(self, event):
        x0, y0, x1, y1 = self._btn_box
        return x0 <= event.x <= x1 and y0 <= event.y <= y1

    def _set_btn_hover(self, inside):
        target = 1.0 if inside else 0.0
        if target != self._hover_target:
            self._hover_target = target
            self.header.configure(cursor="hand2" if inside else "")
            self._tween("_btn_hover", target, 150 if inside else 220)

    def _shine(self):
        """A highlight that sweeps across the header once."""
        if self._shine_id is not None:
            self.anim.cancel(self._shine_id)
        c = self.header

        def frame(p):
            x = -px(120) + (c.winfo_width() + px(240)) * ease(p)
            band = px(30)
            for j, item in enumerate(self._shine_items):
                c.coords(item, x + j * band, 0, x + (j + 1) * band, px(50))

        self._shine_id = self.anim.start(800, frame)

    def set_status(self, text, error=False):
        self.status_var.set(text)
        self._status_error = error
        th = self.th
        self.status_lbl.configure(bg=th["error_bg"] if error else th["status_bg"],
                                  fg=th["error_fg"] if error else th["text"])

    # ---- start-up
    def _set_alpha(self, a):
        try:
            self.attributes("-alpha", a)
        except tk.TclError:
            pass

    def reveal(self):
        """Show the main window with a fade-in, then play the intro."""
        self._set_alpha(0.0)
        self.deiconify()
        self.lift()
        self._style_titlebar()          # Windows 10 repaints it once the window is active
        self.anim.start(300, self._set_alpha)
        self.after(250, self._shine)
        self.after(350, self.scalar_tab.intro)
        self.after(2500, self._prebuild_next)

    def _quit(self):
        for tab in self.tabs:
            tab.shutdown()
        shutdown_worker()
        self.anim.stop_all()
        self._end_precise_timers()
        self.quit()
        self.destroy()

    def _style(self, th):
        s = ttk.Style(self)
        if s.theme_use() != "clam":
            s.theme_use("clam")
        bg, panel, field, border = th["bg"], th["panel"], th["field"], th["border"]
        text, muted, accent, soft, button = th["text"], th["muted"], th["accent"], th["soft"], th["button"]
        bold = ("Segoe UI", 10, "bold")

        s.configure(".", background=panel, foreground=text, font=("Segoe UI", 10),
                    bordercolor=border, lightcolor=panel, darkcolor=panel, troughcolor=bg,
                    fieldbackground=field, selectbackground=accent,
                    selectforeground=th["accent_fg"], insertcolor=text, arrowcolor=muted,
                    focuscolor=accent)
        s.map(".", foreground=[("disabled", muted)])
        s.configure("Bg.TFrame", background=bg)
        s.configure("Panel.TFrame", background=panel)
        s.configure("TLabelframe", background=panel, bordercolor=border)
        s.configure("TLabelframe.Label", background=panel, foreground=accent, font=bold)
        s.configure("Head.TLabel", foreground=accent, font=("Segoe UI", 11, "bold"))
        s.configure("Muted.TLabel", foreground=muted, font=("Segoe UI", 8))
        s.configure("Status.TLabel", background=th["status_bg"], foreground=text)
        s.configure("Error.TLabel", background=th["error_bg"], foreground=th["error_fg"])
        for role, colour in (("U", th["v"]), ("V", th["kv"]), ("W", th["w"]), ("P", th["proj"])):
            s.configure(f"Role{role}.TLabel", foreground=colour, font=bold)

        s.configure("TButton", background=button, foreground=text, bordercolor=border,
                    lightcolor=button, darkcolor=button)
        s.map("TButton", background=[("disabled", panel), ("pressed", soft), ("active", soft)],
              foreground=[("disabled", muted)])
        s.configure("Accent.TButton", background=accent, foreground=th["accent_fg"],
                    bordercolor=accent, lightcolor=accent, darkcolor=accent, font=bold, padding=6)
        s.map("Accent.TButton",
              background=[("disabled", th["accent_disabled"]), ("pressed", th["accent_hover"]),
                          ("active", th["accent_hover"])],
              foreground=[("disabled", muted if th["dark"] else th["accent_fg"])])
        s.configure("Small.TButton", padding=(4, 2), font=("Segoe UI", 9))

        s.configure("TEntry", fieldbackground=field, foreground=text, insertcolor=text,
                    bordercolor=border, lightcolor=field, darkcolor=field)
        s.map("TEntry", fieldbackground=[("disabled", bg)], foreground=[("disabled", muted)],
              bordercolor=[("focus", accent)], lightcolor=[("focus", accent)])
        for w in ("TCheckbutton", "TRadiobutton"):
            s.configure(w, background=panel, foreground=text, indicatorbackground=field,
                        indicatorforeground=accent, upperbordercolor=muted,
                        lowerbordercolor=muted)
            s.map(w, background=[("active", panel)],
                  indicatorbackground=[("pressed", soft), ("selected", field)])
        s.configure("TScale", background=accent, troughcolor=button, bordercolor=border,
                    lightcolor=accent, darkcolor=accent)
        s.map("TScale", background=[("active", th["accent_hover"])])
        s.configure("TScrollbar", background=button, troughcolor=panel, bordercolor=border,
                    arrowcolor=muted, lightcolor=button, darkcolor=button)
        s.map("TScrollbar", background=[("active", soft)])
        s.configure("TPanedwindow", background=panel)

        s.configure("TNotebook", background=bg, bordercolor=border, lightcolor=bg, darkcolor=bg)
        s.configure("TNotebook.Tab", padding=(10, 5), background=button, foreground=muted,
                    bordercolor=border, lightcolor=button)
        s.map("TNotebook.Tab", background=[("selected", panel), ("active", soft)],
              foreground=[("selected", accent), ("active", text)],
              lightcolor=[("selected", panel)])


def main():
    app = LinearAlgebraApp()
    if "--no-splash" in sys.argv or not app.settings["splash"] or app.anim.speed == 0:
        app.after(50, app.reveal)
    else:
        app.withdraw()
        Splash(app, on_done=app.reveal)
    app.mainloop()


if __name__ == "__main__":
    main()
