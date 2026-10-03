"""
tab_scalar.py
-------------
Scalar multiplication k·v: what multiplying a vector by a number does.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np

import algebra_render
import linalg_core as la
from animation import ease
from plot_scene import Scene2D, Scene3D, TRAIL_LEN
from tab_base import BaseTab

K_MIN, K_MAX = -5.0, 5.0


class ScalarMultiplicationTab(BaseTab):
    title = "  Scalar × Vector  "

    def __init__(self, master, ctx):
        super().__init__(master, ctx)
        self._view = (25, -60)
        self._scene_sig = None
        self._tween = None
        self._anim_k = None          # exact k while tweening (the entry shows it rounded)
        self._anim_lim = None
        self._trail = []

        self.dim = tk.StringVar(value="2D")
        self.comp = [tk.StringVar(value=s) for s in ("3", "2", "2")]
        self.k_text = tk.StringVar(value="1.5")
        self.k_slider = tk.DoubleVar(value=1.5)
        self.show_span = tk.BooleanVar(value=True)
        self.show_proj = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_trail = tk.BooleanVar(value=True)

    # ============================================================ layout ===
    def build_controls(self, side):
        box = ttk.LabelFrame(side, text=" Vector v ", padding=10)
        box.pack(fill="x")
        dims = ttk.Frame(box)
        dims.pack(fill="x", pady=(0, 8))
        for d in ("2D", "3D"):
            ttk.Radiobutton(dims, text=d, value=d, variable=self.dim,
                            command=self._on_dim).pack(side="left", padx=(0, 12))
        grid = ttk.Frame(box)
        grid.pack(fill="x")
        for i, name in enumerate(("x", "y", "z")):
            ttk.Label(grid, text=f"{name} =").grid(row=0, column=2 * i, padx=(0, 3))
            e = ttk.Entry(grid, textvariable=self.comp[i], width=6, justify="center")
            e.grid(row=0, column=2 * i + 1, padx=(0, 10))
        self._z_entries.append(e)
        e.configure(state="disabled")

        box = ttk.LabelFrame(side, text=" Scalar k ", padding=10)
        box.pack(fill="x", pady=(12, 0))
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="k =").pack(side="left")
        ttk.Entry(row, textvariable=self.k_text, width=8,
                  justify="center").pack(side="left", padx=6)
        self.scale = ttk.Scale(box, from_=K_MIN, to=K_MAX, variable=self.k_slider,
                               command=self._on_slider)
        self.scale.pack(fill="x", pady=(10, 2))
        ticks = ttk.Frame(box)
        ticks.pack(fill="x")
        ttk.Label(ticks, text=f"{K_MIN:g}", style="Muted.TLabel").pack(side="left")
        ttk.Label(ticks, text="0", style="Muted.TLabel").pack(side="left", expand=True)
        ttk.Label(ticks, text=f"{K_MAX:g}", style="Muted.TLabel").pack(side="right")
        quick = ttk.Frame(box)
        quick.pack(fill="x", pady=(8, 0))
        for val in (-2, -1, -0.5, 0, 0.5, 2, 3):
            ttk.Button(quick, text=f"{val:g}", width=4, style="Small.TButton",
                       command=lambda v=val: self.tween_k(v, 450)).pack(side="left", padx=1)

        acts = ttk.Frame(side)
        acts.pack(fill="x", pady=(12, 0))
        self.anim_btn = ttk.Button(acts, text="▶  Animate 1 → k", style="Accent.TButton",
                                   command=self.play)
        self.anim_btn.pack(fill="x")
        two = ttk.Frame(acts)
        two.pack(fill="x", pady=(6, 0))
        ttk.Button(two, text="Reset", command=self.reset).pack(side="left", expand=True, fill="x")
        ttk.Button(two, text="Save image…", command=self.save_image).pack(
            side="left", expand=True, fill="x", padx=(6, 0))

        box = ttk.LabelFrame(side, text=" Display ", padding=(10, 6))
        box.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(box, text="Span of v (line of all k·v)", variable=self.show_span).pack(anchor="w")
        ttk.Checkbutton(box, text="Component guides", variable=self.show_proj).pack(anchor="w")
        ttk.Checkbutton(box, text="Arrow labels", variable=self.show_labels).pack(anchor="w")
        ttk.Checkbutton(box, text="Motion trail when animating", variable=self.show_trail).pack(anchor="w")

        self.watch(*self.comp, self.k_text, self.show_span, self.show_proj, self.show_labels)

    # ============================================================ events ===
    def _on_slider(self, _value):
        if self._anim_k is not None:
            return
        k = round(self.k_slider.get() * 20) / 20     # snap to 0.05
        if self._read_k() != k:
            self.k_text.set(f"{k:g}")

    def reset(self):
        self._stop_tween()
        for var, s in zip(self.comp, ("3", "2", "2")):
            var.set(s)
        self._view = (25, -60)
        self._scene_sig = None           # rebuild so the 3D camera resets too
        self.tween_k(1.5, 500)
        self.status("Reset to v = (3, 2), k = 1.5")

    # ======================================================== animation ===
    def _stop_tween(self):
        if self._tween is not None:
            self.anim.cancel(self._tween)
            self._tween = None
        self._anim_k = self._anim_lim = None
        self._trail = []
        if self.anim_btn is not None:
            self.anim_btn.configure(state="normal")

    def tween_k(self, target, duration_ms, start=None, on_done=None, trail=False):
        """Smoothly change k from its current value to target (eased).

        The plot is drawn with the exact k of each frame (not the rounded
        value shown in the entry), and the axis limits glide along with it,
        so the motion has no visible steps."""
        self._stop_tween()
        if start is None:
            start = self._read_k() or 0.0
        v, err = self._read_vector()
        if err:
            self.k_text.set(f"{target:g}")
            return
        lim0, lim1 = self._axes_limit(v, start), self._axes_limit(v, target)
        use_trail = trail and self.show_trail.get()

        def frame(p):
            t = ease(p)
            k = start + (target - start) * t
            self._anim_k = k
            self._anim_lim = lim0 + (lim1 - lim0) * t
            if use_trail:
                self._trail.append(k)
                del self._trail[:-TRAIL_LEN]
            self.k_text.set(f"{k:.2f}")

        def done():
            self._tween = None
            self._anim_k = self._anim_lim = None
            self._trail = []
            self.anim_btn.configure(state="normal")
            self.k_text.set(f"{target:g}")
            self.refresh()
            if on_done:
                on_done()

        self.anim_btn.configure(state="disabled")
        self._tween = self.anim.start(duration_ms, frame, done)

    def play(self):
        target = self._read_k()
        if target is None:
            return
        self.tween_k(target, 1600, start=1.0, trail=True,
                     on_done=lambda: (self._update_explanations(), self.replay_steps()))

    def intro(self):
        """Played once when the window first appears."""
        self.tween_k(1.5, 1300, start=0.0, trail=True,
                     on_done=lambda: (self._update_explanations(), self.replay_steps()))

    def is_animating(self):
        return self._anim_k is not None

    def shutdown(self):
        self._stop_tween()
        super().shutdown()

    # ============================================================= input ===
    def _read_k(self):
        try:
            return float(self.k_text.get().replace(",", "."))
        except ValueError:
            return None

    def _read_vector(self):
        return self.read_vector(self.comp, "v")

    def params_for_plot(self):
        v, err = self._read_vector()
        if err:
            return None, err
        if self._anim_k is not None:
            return (v, self._anim_k), None
        k, err = self.read_number(self.k_text, "k")
        if err:
            return None, err
        return (v, k), None

    # =========================================================== drawing ===
    def _axes_limit(self, v, k):
        reach = max(3.0, abs(k)) * float(np.max(np.abs(v)))
        return max(reach * 1.12, 1.0)

    def draw(self, p):
        v, k = p
        animating = self._anim_k is not None
        shown_k = round(k, 2) if animating else k
        if K_MIN <= k <= K_MAX and abs(self.k_slider.get() - k) > 1e-9:
            self.k_slider.set(k)

        is_3d = len(v) == 3
        opts = (self.show_span.get(), self.show_proj.get(), self.show_labels.get())
        sig = (is_3d, opts, self.th["name"])
        full = False
        if sig != self._scene_sig:       # dimension, display options or theme changed
            if self.scene is not None and self.scene.view():
                self._view = self.scene.view()
            if is_3d:
                self.scene = Scene3D(self.fig, self.th, *opts, view=self._view)
            else:
                self.scene = Scene2D(self.fig, self.th, *opts)
            self._scene_sig = sig
            self._invalidate_bg()
            full = True

        lim = self._anim_lim if animating else self._axes_limit(v, k)
        full |= self.scene.set_vector(v, lim)
        self.scene.set_k(k, f"k = {shown_k:g}\n{la.effect_text(shown_k)}", self._trail)
        return full

    def status_text(self, p):
        v, k = p
        k = round(k, 2) if self._anim_k is not None else k
        return f"k·v = {la.fmt_vec(la.scalar_multiply(k, v))}   —   {la.effect_text(k)}"

    # ===================================================== explanations ===
    def explain_key(self, p):
        v, k = p
        return (round(k, 6), tuple(np.round(v, 6)))

    def algebra(self, p):
        v, k = p
        return algebra_render.scalar_steps, (float(k), tuple(float(x) for x in v))

    def python_code(self, p):
        v, k = p
        return la.python_code(k, v)
