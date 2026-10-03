"""
tab_base.py
-----------
Everything the calculator tabs have in common, so a new topic only has to
describe its inputs, its picture and its maths.

  BaseTab – the layout (controls | plot | step-by-step), colour themes, the
            fast blitted plot, the algebra picture typeset on a worker
            thread, the "written on the board" replay, and the generated
            Python code with its real output. Tabs are built the first time
            they are opened, so start-up stays fast however many there are.
  GeoTab  – a BaseTab whose plot is vectors (geo_scene.py). When an input
            changes, the picture glides to the new values instead of
            jumping, and the vectors grow in when the tab is first opened.

A subclass of BaseTab implements:
    build_controls(side)   the input widgets (use the add_* helpers)
    params_for_plot()      -> (params, error)
    draw(params)           update the scene; True if a full redraw is needed
    status_text(params)    one line for the status bar
    algebra(params)        -> (builder, args) for algebra_render.render_steps
    python_code(params)    a runnable program that does the same calculation
"""

import math
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import algebra_render
import linalg_core as la
from animation import ease, FPS
from geo_scene import GeoScene2D, GeoScene3D
from plot_nav import PlotNavigator
import ui_scale
from ui_scale import px
from tab_bar import TabBar

ALGEBRA_W = 400
RESIZE_SETTLE_MS = 90      # redraw the plot this long after the window stops resizing

# The algebra picture takes 0.2-0.4 s to typeset, so it is drawn on one
# background thread instead of freezing the window. This is safe: it uses its
# own Figure (no pyplot), Matplotlib keeps fonts per thread, and only this
# worker ever parses maths text.
_ALGEBRA_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="algebra")


def shutdown_worker():
    _ALGEBRA_WORKER.shutdown(wait=False, cancel_futures=True)


def pick_mono():
    have = set(tkfont.families())
    for name in ("Consolas", "Cascadia Mono", "Menlo", "DejaVu Sans Mono",
                 "Liberation Mono", "Courier New"):
        if name in have:
            return name
    return "TkFixedFont"


def freeze(x):
    """Turn params (dicts, arrays, lists) into something hashable."""
    if isinstance(x, dict):
        return tuple(sorted((k, freeze(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple, np.ndarray)):
        return tuple(freeze(i) for i in x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)) and not isinstance(x, bool):
        return float(x)
    return x


class BaseTab(tk.Frame):
    # A plain tk.Frame rather than ttk.Frame: ttk repaints through an
    # off-screen bitmap the size of the widget, which makes big containers
    # slow to redraw while the window is being resized.
    title = "Tab"
    python_keywords = ("import|as|if|elif|else|for|in|print|round|abs|float|def|return|"
                       "True|False|and|or|not|sum|zip|range|len|min|max|assert")

    def __init__(self, master, ctx):
        super().__init__(master, bg=ctx.theme["bg"], bd=0, highlightthickness=0)
        self.ctx = ctx
        self._panels = []            # big plain frames recoloured with the theme
        self.status = ctx.status
        self.anim = ctx.anim
        self.th = ctx.theme
        self.built = False
        self.scene = None
        self.dim = None
        self.anim_btn = None
        self._z_entries = []         # z fields, disabled while in 2D
        self.entries = {}            # id(vars) -> Entry widgets of a vector
        self._bg = None              # bitmap of the static plot layer (for blitting)
        self._full_pending = True    # a full redraw is queued; don't blit until it ran
        self._render_job = None
        self._explain_job = None
        self._resize_job = None
        self._pending_size = None
        self._sized_once = False
        self._poll_job = None
        self._alg_future = None
        self._alg_img = None
        self._alg_steps = ()
        self._alg_h = 0
        self._replay = None
        self._replay_pending = False
        self._last_key = None

    # ===================================================== to implement ===
    def build_controls(self, side):
        raise NotImplementedError

    def params_for_plot(self):
        raise NotImplementedError

    def params_for_explain(self):
        return self.params_for_plot()

    def is_animating(self):
        return False

    def draw(self, params):
        raise NotImplementedError

    def status_text(self, params):
        return ""

    def algebra(self, params):
        raise NotImplementedError

    def python_code(self, params):
        raise NotImplementedError

    def explain_key(self, params):
        return params

    def on_first_show(self):
        pass

    def play(self):
        self.replay_steps()

    # ========================================================= lifecycle ===
    def ensure_built(self):
        if self.built:
            return
        self.built = True
        self.mono = pick_mono()
        side = self.panel(padx=14, pady=14)
        side.pack(side="left", fill="y", padx=(10, 0), pady=10)
        self.build_controls(side)
        self._build_explanations()
        self._build_plot()
        self.apply_theme(self.th)

    def on_show(self):
        """Called when the user opens this tab."""
        first = not self.built
        self.ensure_built()
        self.refresh()
        self._schedule_explanations(20)
        if first:
            self.after(80, self.on_first_show)

    def is_shown(self):
        return self.built and self.ctx.is_current(self)

    def panel(self, **kw):
        """A large plain-tk container in the panel colour (see class comment)."""
        frame = tk.Frame(self, bg=self.th["panel"], bd=0, highlightthickness=0, **kw)
        self._panels.append(frame)
        return frame

    def shutdown(self):
        """Cancel pending timers so none fires after the window is gone."""
        self._alg_future = None
        for job in (self._resize_job, self._poll_job, self._explain_job, self._render_job):
            if job is not None:
                self.after_cancel(job)

    # ==================================================== control helpers ===
    def watch(self, *variables):
        for var in variables:
            var.trace_add("write", lambda *_: self.refresh())

    def add_box(self, parent, title, role=None, pady=(12, 0)):
        """A titled group; `role` (U/V/W/P) puts a coloured dot in the title."""
        if role:
            label = ttk.Label(parent, text=f"●  {title}", style=f"Role{role}.TLabel")
            box = ttk.LabelFrame(parent, labelwidget=label, padding=(10, 8))
        else:
            box = ttk.LabelFrame(parent, text=f" {title} ", padding=(10, 8))
        box.pack(fill="x", pady=pady)
        return box

    def add_dim_selector(self, parent, pady=(0, 0)):
        self.dim = tk.StringVar(value="2D")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=pady)
        ttk.Label(row, text="Space:").pack(side="left", padx=(0, 8))
        for d in ("2D", "3D"):
            ttk.Radiobutton(row, text=d, value=d, variable=self.dim,
                            command=self._on_dim).pack(side="left", padx=(0, 12))
        return self.dim

    def _on_dim(self):
        state = "normal" if self.ndim() == 3 else "disabled"
        for e in self._z_entries:
            e.configure(state=state)
        self.refresh()

    def ndim(self):
        return 3 if self.dim is not None and self.dim.get() == "3D" else 2

    def add_vector_box(self, parent, title, role, defaults, pady=(12, 0)):
        """A titled box of x/y/z entry fields; returns their StringVars."""
        box = self.add_box(parent, title, role, pady=pady)
        return self.add_xyz_row(box, defaults)

    def add_xyz_row(self, parent, defaults, label=None, pady=(0, 0)):
        """x/y/z entry fields in `parent`; returns their StringVars (the Entry
        widgets are kept in self.entries[id(vars)] for enabling/disabling)."""
        variables = [tk.StringVar(value=s) for s in defaults]
        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=pady)
        col = 0
        if label:
            ttk.Label(grid, text=label).grid(row=0, column=0, padx=(0, 6))
            col = 1
        entries = []
        for i, name in enumerate("xyz"[:len(defaults)]):
            ttk.Label(grid, text=f"{name} =").grid(row=0, column=col + 2 * i, padx=(0, 3))
            e = ttk.Entry(grid, textvariable=variables[i], width=6, justify="center")
            e.grid(row=0, column=col + 2 * i + 1, padx=(0, 8))
            entries.append(e)
            if i == 2:
                self._z_entries.append(e)
                e.configure(state="disabled" if self.ndim() == 2 else "normal")
        self.entries[id(variables)] = entries
        self.watch(*variables)
        return variables

    def add_entry_row(self, parent, label, default, width=7, pady=(0, 0)):
        var = tk.StringVar(value=default)
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=pady)
        ttk.Label(row, text=label).pack(side="left")
        entry = ttk.Entry(row, textvariable=var, width=width, justify="center")
        entry.pack(side="left", padx=6)
        self.watch(var)
        return var, entry

    def add_checks(self, parent, title, items):
        """items: (text, default) pairs; returns BooleanVars."""
        box = self.add_box(parent, title)
        out = []
        for text, default in items:
            var = tk.BooleanVar(value=default)
            ttk.Checkbutton(box, text=text, variable=var).pack(anchor="w")
            self.watch(var)
            out.append(var)
        return out

    def add_actions(self, parent, label="▶  Animate"):
        acts = ttk.Frame(parent)
        acts.pack(fill="x", pady=(14, 0))
        self.anim_btn = ttk.Button(acts, text=label, style="Accent.TButton", command=self.play)
        self.anim_btn.pack(fill="x")
        ttk.Button(acts, text="Save image…", command=self.save_image).pack(fill="x", pady=(6, 0))

    # ---- reading inputs
    @staticmethod
    def read_number(var, name):
        try:
            x = float(var.get().replace(",", "."))
        except ValueError:
            return None, f"{name} must be a number."
        if not math.isfinite(x):
            return None, f"{name} must be finite."
        if abs(x) > 1e6:
            return None, f"Please keep {name} below one million."
        return x, None

    def read_vector(self, variables, name, n=None):
        n = n or self.ndim()
        values = []
        for var, c in zip(variables[:n], "xyzw"):
            x, err = self.read_number(var, f"{name}'s {c}")
            if err:
                return None, err
            values.append(x)
        return la.to_vector(values), None

    # ============================================================ layout ===
    def _build_explanations(self):
        side = self.panel(padx=10, pady=10)
        side.pack(side="right", fill="y", padx=(0, 10), pady=10)

        head = ttk.Frame(side)
        head.pack(fill="x", pady=(0, 6))
        ttk.Label(head, text="Step-by-step calculation", style="Head.TLabel").pack(side="left")
        ttk.Button(head, text="▶ Replay steps", style="Small.TButton",
                   command=self.replay_steps).pack(side="right")

        # Algebra / Python pages under a small canvas tab bar (a ttk.Notebook
        # here would be repainted through a big bitmap on every resize step)
        self.expl_bar = TabBar(side, self.anim, self._show_page, height=32)
        self.expl_bar.pack(fill="x")
        pages = tk.Frame(side, bd=0, highlightthickness=0)
        pages.pack(fill="both", expand=True)
        self._pages = []

        # --- Algebra tab: rendered maths on a scrollable canvas
        alg = tk.Frame(pages, bd=0, highlightthickness=0)
        self._pages.append(alg)
        self.expl_bar.add("∑  Algebra")
        self.alg_canvas = tk.Canvas(alg, width=px(ALGEBRA_W), highlightthickness=0)
        sb = ttk.Scrollbar(alg, orient="vertical", command=self.alg_canvas.yview)
        self.alg_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.alg_canvas.pack(side="left", fill="both", expand=True)
        self.alg_canvas.bind("<Enter>", lambda e: self._bind_wheel(self.alg_canvas))
        self.alg_canvas.bind("<Leave>", lambda e: self._unbind_wheel())

        # --- Python tab: generated code + its real output
        py = tk.Frame(pages, bd=0, highlightthickness=0)
        self._pages.append(py)
        self.expl_bar.add("</>  Python")
        self._page = None
        self._show_page(0)
        bar = ttk.Frame(py)
        bar.pack(fill="x", pady=(6, 4))
        ttk.Label(bar, text="Code for the current values", style="Muted.TLabel").pack(side="left")
        ttk.Button(bar, text="Copy code", style="Small.TButton",
                   command=self.copy_code).pack(side="right")
        ttk.Button(bar, text="Run ▶", style="Small.TButton",
                   command=self._run_code).pack(side="right", padx=4)

        pw = ttk.PanedWindow(py, orient="vertical")
        pw.pack(fill="both", expand=True)
        code_frame = ttk.Frame(pw)
        self.code = tk.Text(code_frame, width=48, height=20, font=(self.mono, 10),
                            relief="flat", wrap="none", padx=8, pady=8)
        csb = ttk.Scrollbar(code_frame, orient="vertical", command=self.code.yview)
        self.code.configure(yscrollcommand=csb.set)
        csb.pack(side="right", fill="y")
        self.code.pack(fill="both", expand=True)
        pw.add(code_frame, weight=3)

        out_frame = ttk.Frame(pw)
        ttk.Label(out_frame, text="Output", style="Muted.TLabel").pack(anchor="w", pady=(6, 2))
        self.output = tk.Text(out_frame, width=48, height=8, font=(self.mono, 10),
                              relief="flat", wrap="word", padx=8, pady=6)
        self.output.pack(fill="both", expand=True)
        pw.add(out_frame, weight=1)

        for tag, colour in (("num", "#fbbf24"), ("kw", "#c084fc"), ("np", "#67e8f9"),
                            ("str", "#86efac"), ("com", "#64748b")):
            self.code.tag_configure(tag, foreground=colour)
        self.code.tag_configure("com", font=(self.mono, 10, "italic"))

    def _build_plot(self):
        mid = self.panel(padx=6, pady=6)
        mid.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.fig = Figure(figsize=(6.5, 6.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=mid)
        # no toolbar: the mouse moves / zooms / rotates the plot (plot_nav.py)
        ttk.Label(mid, style="Muted.TLabel",
                  text="Drag: move (2D) or rotate (3D)   ·   Scroll: zoom   ·   "
                       "Double-click or Ctrl+0: reset view").pack(side="bottom", anchor="w",
                                                                   pady=(4, 0))
        self.nav = PlotNavigator(self)
        widget = self.canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        # every full redraw (first show, resize, pan/zoom) refreshes the cached
        # static layer that the animation frames are blitted onto
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("resize_event", self._on_mpl_resize)
        # Matplotlib re-renders the whole figure for every pixel the window is
        # dragged by (~45 ms each), which makes resizing judder. Replace its
        # handler: while the size keeps changing just keep the old picture
        # centred, and render once when the resizing settles.
        widget.bind("<Configure>", self._on_plot_configure)

    def _on_plot_configure(self, event):
        self._pending_size = (event.width, event.height)
        if not self._sized_once:         # first appearance: no reason to wait
            self._sized_once = True
            self._apply_plot_resize()
            return
        try:
            self.canvas.get_tk_widget().coords(self.canvas._tkcanvas_image_region,
                                               event.width // 2, event.height // 2)
        except (AttributeError, tk.TclError):
            pass                         # private Matplotlib detail; centring is cosmetic
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(RESIZE_SETTLE_MS, self._apply_plot_resize)

    def _on_mpl_resize(self, _event):
        self._invalidate_bg()
        self.on_resize()

    def on_resize(self):
        pass

    def _apply_plot_resize(self):
        self._resize_job = None
        w, h = self._pending_size
        if (w, h) != tuple(self.canvas.get_width_height()):
            self.canvas.resize(SimpleNamespace(width=w, height=h))

    def _show_page(self, i):
        if self._page == i:
            return
        if self._page is not None:
            self._pages[self._page].pack_forget()
        self._page = i
        self._pages[i].pack(fill="both", expand=True)
        self.expl_bar.select(i, animate=self.built)

    def _bind_wheel(self, widget):
        def on_wheel(e):
            step = -1 if (getattr(e, "delta", 0) > 0 or e.num == 4) else 1
            widget.yview_scroll(step * 2, "units")
        self.bind_all("<MouseWheel>", on_wheel)
        self.bind_all("<Button-4>", on_wheel)
        self.bind_all("<Button-5>", on_wheel)

    def _unbind_wheel(self):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.unbind_all(seq)

    # ============================================================= theme ===
    def apply_theme(self, th):
        self.th = th
        self.configure(bg=th["bg"])
        if not self.built:
            return                       # applied when the tab is first opened
        for frame in self._panels + self._pages:
            frame.configure(bg=th["panel"])
        self.expl_bar.apply(th)
        self.alg_canvas.configure(bg=th["panel"])
        # shows around the plot for a moment while the window is being resized
        self.canvas.get_tk_widget().configure(bg=th["plot_bg"])
        self.code.configure(bg=th["code_bg"], fg=th["code_fg"], insertbackground=th["code_fg"],
                            selectbackground=th["accent"], selectforeground=th["accent_fg"])
        self.output.configure(bg=th["out_bg"], fg=th["out_fg"])
        self._last_key = None            # algebra picture must be re-rendered in new colours
        self.alg_canvas.delete("all")    # rather blank for a moment than in the old colours
        self.refresh()
        self._update_explanations()

    def _palette(self):
        th = self.th
        return dict(ink=th["text"], muted=th["muted"], accent=th["accent"],
                    on_accent=th["accent_fg"], v=th["v"], kv=th["kv"], w=th["w"],
                    proj=th["proj"], paper=th["panel"], result_bg=th["result_bg"],
                    rule=th["border"])

    # =========================================================== drawing ===
    def refresh(self):
        """Request a redraw. Requests arriving together (a slider drag, an
        animation frame, several traces) are merged into a single render.
        Hidden tabs skip it; they refresh when opened."""
        if self._render_job is None and self.is_shown():
            self._render_job = self.after_idle(self._render)

    def _render(self):
        self._render_job = None
        params, err = self.params_for_plot()
        if err:
            self.status("⚠  " + err, error=True)
            return
        full = self.draw(params)
        if full or not self.scene.blit:
            self._full_pending = True
            self.canvas.draw_idle()
        else:
            self._blit()
        self.status(self.status_text(params))
        if not self.is_animating():
            self._schedule_explanations()

    def _blit(self):
        """Fast frame: paste the cached static layer, draw only what moved."""
        if self._bg is None or self._full_pending:
            self._full_pending = True
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self._bg)
        self.scene.draw_frame()
        self.canvas.blit(self.fig.bbox)

    def _on_draw(self, _event):
        self._full_pending = False
        if self.scene is None or not self.scene.blit or self.canvas.is_saving():
            return
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self.scene.draw_frame()          # animated artists are skipped by a full draw

    def _invalidate_bg(self):
        self._bg = None

    def request_full_draw(self):
        """After the view changed (move/zoom/rotate): one full redraw, merged
        with any others requested before it runs."""
        self._full_pending = True
        self.canvas.draw_idle()

    def reset_view(self):
        """Back to the default view (double-click, Ctrl+0, Home)."""
        lim = getattr(self.scene, "lim", None)
        if lim:
            self.nav.reset(lim)
        else:                               # diagrams: draw them again from scratch
            self.refresh()

    def prewarm(self):
        """Do the slow first work of this tab while it is still hidden (called
        by the app when idle): size the plot, draw it once (Matplotlib caches
        its text measurements per figure size) and typeset the maths."""
        widget = self.canvas.get_tk_widget()
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
            self._resize_job = None
        self._sized_once = True
        self._pending_size = (widget.winfo_width(), widget.winfo_height())
        self._apply_plot_resize()
        params, err = self.params_for_plot()
        if err:
            return
        self.draw(params)
        self.canvas.draw()
        params, err = self.params_for_explain()
        if not err:
            builder, args = self.algebra(params)
            _ALGEBRA_WORKER.submit(algebra_render.render_steps, builder, freeze(args),
                                   ALGEBRA_W, self._palette(), ui_scale.S)

    # ===================================================== explanations ===
    def _schedule_explanations(self, delay=220):
        if self._explain_job:
            self.after_cancel(self._explain_job)
        self._explain_job = self.after(delay, self._update_explanations)

    def _update_explanations(self):
        if self._explain_job:
            self.after_cancel(self._explain_job)
            self._explain_job = None
        if not self.is_shown():
            return
        params, err = self.params_for_explain()
        if err:
            return
        key = (freeze(self.explain_key(params)), self.th["name"])
        if key == self._last_key:
            return
        self._last_key = key
        builder, args = self.algebra(params)
        self._show_algebra(builder, freeze(args))
        self._show_python(self.python_code(params))

    def _show_algebra(self, builder, args):
        self._cancel_replay()
        future = _ALGEBRA_WORKER.submit(algebra_render.render_steps, builder, args, ALGEBRA_W,
                                        self._palette(), ui_scale.S)
        self._alg_future = future
        self._poll_algebra(future)       # cached results come back immediately

    def _poll_algebra(self, future):
        self._poll_job = None
        if future is not self._alg_future:
            return                       # newer values arrived meanwhile
        if not future.done():
            self._poll_job = self.after(15, self._poll_algebra, future)
            return
        self._alg_future = None
        try:
            data, h, steps = future.result()
        except Exception as e:           # shown to the user rather than lost in a thread
            self.status(f"⚠  Could not draw the algebra: {e}", error=True)
            return
        self._alg_img = tk.PhotoImage(data=data)      # keep a reference
        self._alg_steps, self._alg_h = steps, h
        c = self.alg_canvas
        c.delete("all")
        c.create_image(0, 0, image=self._alg_img, anchor="nw")
        c.configure(scrollregion=(0, 0, px(ALGEBRA_W), h))
        if self._replay_pending:
            self._replay_pending = False
            self._start_replay()

    def _show_python(self, code):
        t = self.code
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("1.0", code)
        self._highlight(code)
        t.configure(state="disabled")
        self._run_code()

    def _highlight(self, code):
        patterns = [
            ("num", r"(?<![\w.])-?\d+(\.\d+)?"),
            ("kw", rf"\b({self.python_keywords})\b"),
            ("np", r"\bnp(\.\w+)+"),
            ("str", r'f?"[^"\n]*"'),
            ("com", r"#.*"),
        ]
        for tag, pat in patterns:
            for m in re.finditer(pat, code):
                self.code.tag_add(tag, f"1.0+{m.start()}c", f"1.0+{m.end()}c")

    def _run_code(self):
        out = la.run_python(self.code.get("1.0", "end"))
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", ">>> running code…\n" + out)
        self.output.configure(state="disabled")

    def copy_code(self):
        self.clipboard_clear()
        self.clipboard_append(self.code.get("1.0", "end-1c"))
        self.status("Python code copied to clipboard")

    # ---- "written on the board" reveal of the algebra steps
    def _cancel_replay(self):
        if self._replay is not None:
            self.anim.cancel(self._replay)
            self._replay = None
        self.alg_canvas.delete("cover")

    def replay_steps(self):
        self._update_explanations()
        if self._alg_future is not None:
            self._replay_pending = True      # starts as soon as the picture is ready
            return
        self._start_replay()

    def _start_replay(self):
        if not self._alg_steps:
            return
        self._cancel_replay()
        self._show_page(0)
        c = self.alg_canvas
        H, W = self._alg_h, px(ALGEBRA_W)
        m, speed = px(12), 7 * ui_scale.S       # pen margin; px written per frame
        top = self._alg_steps[0][0]
        accent = self.th["accent"]
        cover = c.create_rectangle(0, top, W, H + 2, fill=self.th["panel"], outline="",
                                   tags="cover")
        pen = c.create_line(m, top, W - m, top, fill=accent, width=px(2), tags="cover")
        nib = c.create_oval(0, 0, 0, 0, fill=accent, outline="", tags="cover")
        c.yview_moveto(0)

        # timeline (seconds): each step is written at ~7 px per 60 fps frame
        # (at least 14 frames), followed by a short 12-frame pause
        segments, t = [], 0.0
        for s0, s1 in self._alg_steps:
            dur = max(14, int((s1 - s0) / speed)) / FPS
            segments.append((t, t + dur, s0, s1))
            t += dur + 12 / FPS
        total = t

        def frame(p):
            now = p * total
            y = top
            for a, b, s0, s1 in segments:
                if now < a:
                    break                # pausing after the previous step
                y = s0 + (s1 - s0) * ease((now - a) / (b - a)) if now < b else s1
            c.coords(cover, 0, y, W, H + 2)
            c.coords(pen, m, y, W - m, y)
            x = m + (W - 2 * m) * ((now * FPS * 7) % 100) / 100
            r = px(4)
            c.coords(nib, x - r, y - r, x + r, y + r)
            # keep the pen in view
            view_h = c.winfo_height() or 600
            first, _ = c.yview()
            if y > first * H + view_h * 0.8:
                c.yview_moveto(max(0, (y - view_h * 0.5) / H))

        self._replay = self.anim.start(total * 1000, frame, self._cancel_replay)

    # =========================================================== actions ===
    def save_image(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile=self.title.strip().lower().replace(" ", "_").replace("&", "and") + ".png")
        if path:
            try:
                self.fig.savefig(path, dpi=150, bbox_inches="tight",
                                 facecolor=self.fig.get_facecolor())
                self.status(f"Saved to {path}")
            except OSError as e:
                messagebox.showerror("Save failed", str(e))
            self._full_pending = True
            self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# GeoTab: vector pictures that glide between values
# ---------------------------------------------------------------------------

def _is_num(x):
    return isinstance(x, (float, np.ndarray)) and not isinstance(x, bool)


def _compatible(a, b):
    """Same structure, so the picture can glide from a to b."""
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
                    and x.shape == y.shape):
                return False
        elif not _is_num(y) and x != y:  # a mode, a dimension or a checkbox changed
            return False
    return True


def _equal(a, b):
    return all(np.array_equal(a[k], b[k]) if isinstance(b[k], np.ndarray) else a[k] == b[k]
               for k in b)


def _lerp(a, b, t):
    return {k: a[k] + (b[k] - a[k]) * t if _is_num(b[k]) else b[k] for k in b}


def _zeros(p):
    return {k: v * 0 if _is_num(v) else v for k, v in p.items()}


def nice_lim(extent, margin=1.2):
    """Axis half-width from a short list of round numbers, so small edits do
    not change the axes (which would force a full redraw)."""
    x = max(extent * margin, 1.0)
    e = 10 ** math.floor(math.log10(x))
    for m in (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * e >= x - 1e-12:
            return m * e
    return 10 * e


class GeoTab(BaseTab):
    """Subclasses implement read_params(), geometry(scene, p) and usually
    extent(p), legend(p), status_text(p), algebra(p), python_code(p)."""

    glide_ms = 380
    intro_ms = 950

    def __init__(self, master, ctx):
        super().__init__(master, ctx)
        self._target = None            # values from the inputs
        self._shown = None             # values currently drawn (glide towards target)
        self._glide_id = None
        self._lim_now = None
        self._scene_sig = None
        self._view = (25, -60)

    # ---- to implement
    def read_params(self):
        raise NotImplementedError

    def geometry(self, s, p):
        raise NotImplementedError

    def legend(self, p):
        return []

    def scene_key(self, p):
        return ()

    def is_3d(self, p):
        return p.get("dim", 2) == 3

    def on_scene(self, scene):
        """Called once when a new picture is set up (e.g. to rename the axes)."""

    def extent(self, p):
        arrays = [v for v in p.values() if isinstance(v, np.ndarray) and v.size]
        return max((float(np.max(np.abs(a))) for a in arrays), default=1.0)

    # ---- plumbing
    def refresh(self):
        if not self.is_shown():
            return
        target, err = self.read_params()
        if err:
            self.status("⚠  " + err, error=True)
            return
        old, self._target = self._shown, target
        if old is None or not _compatible(old, target):
            self._stop_glide()
            self._shown = target
            super().refresh()
        elif not _equal(old, target):
            self._glide(old, target, self.glide_ms)
        else:
            super().refresh()

    def _choose_lim(self, p):
        """Axis size for p. Keeps the current size while p still fits and fills
        at least half of it: changing it costs a full redraw on every frame
        of a glide instead of a cheap blit."""
        # 3D axes already pad their box, so they need less margin
        need = nice_lim(self.extent(p), 1.0 if self.is_3d(p) else 1.2)
        cur = self.scene.lim if self.scene is not None else None
        if cur and cur * 0.5 <= need <= cur:
            return cur
        return need

    def _glide(self, a, b, ms, hold_lim=False, on_done=None):
        self._stop_glide()
        lim_b = self._choose_lim(b)
        cur = self.scene.lim if self.scene is not None else None
        lim_a = lim_b if hold_lim or not cur else cur

        def frame(p):
            t = ease(p)
            self._shown = _lerp(a, b, t)
            self._lim_now = lim_a + (lim_b - lim_a) * t
            BaseTab.refresh(self)

        def done():
            self._glide_id = None
            self._shown, self._lim_now = b, None
            BaseTab.refresh(self)
            if on_done:
                on_done()

        gid = self.anim.start(ms, frame, done)
        self._glide_id = gid if self.anim.running(gid) else None

    def _stop_glide(self):
        if self._glide_id is not None:
            self.anim.cancel(self._glide_id)
            self._glide_id = None
        self._lim_now = None

    def is_animating(self):
        return self._glide_id is not None

    def params_for_plot(self):
        if self._shown is None:
            return self.read_params()
        return self._shown, None

    def params_for_explain(self):
        if self._target is None:
            return self.read_params()
        return self._target, None

    def draw(self, p):
        is_3d = self.is_3d(p)
        sig = (is_3d, freeze(self.scene_key(p)), self.th["name"])
        full = False
        if sig != self._scene_sig:
            if self.scene is not None and self.scene.view():
                self._view = self.scene.view()
            cls = GeoScene3D if is_3d else GeoScene2D
            self.scene = cls(self.fig, self.th, self.legend(p), self._view)
            self.on_scene(self.scene)
            self._scene_sig = sig
            self._invalidate_bg()
            full = True
        s = self.scene
        s.begin()
        self.geometry(s, p)
        lim = self._lim_now if self._lim_now is not None else self._choose_lim(p)
        return s.finish(lim) or full

    def play(self):
        """Grow the picture from nothing, then write out the steps."""
        target, err = self.read_params()
        if err:
            self.status("⚠  " + err, error=True)
            return
        self._target = target
        self._glide(_zeros(target), target, self.intro_ms, hold_lim=True,
                    on_done=self.replay_steps)

    def on_first_show(self):
        self.play()

    def shutdown(self):
        self._stop_glide()
        super().shutdown()
