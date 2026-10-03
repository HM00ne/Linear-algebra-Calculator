"""
plot_scene.py
-------------
The vector plot as a set of long-lived Matplotlib artists.

The first version cleared the axes and rebuilt everything (ticks, grid,
legend, arrows, labels) on every animation frame, which cost ~75 ms per
frame. Here the artists are created once and only their data changes:

  * Scene2D: everything that depends only on v (axes, grid, span line,
    v arrow, legend) is rendered once and kept as a bitmap. Each frame
    restores that bitmap and draws only the k-dependent artists on top
    ("blitting"), which is several times faster than a full redraw.
  * Scene3D: the same idea. The static layer depends on the camera angle,
    so rotating the view triggers a full redraw (which re-caches it), but
    animating k with the camera still is blitted.
"""

import numpy as np
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

import linalg_core as la
from themes import mix

TRAIL_LEN = 14


def _legend(ax, th, show_span, loc):
    handles = [Line2D([], [], color=th["v"], lw=3),
               Line2D([], [], color=th["kv"], lw=2)]
    labels = ["v", "k·v"]
    if show_span:
        handles.insert(0, Line2D([], [], ls="--", color=th["span"], lw=1))
        labels.insert(0, "span{v}")
    ax.legend(handles, labels, loc=loc, fontsize=9, framealpha=0.9,
              facecolor=th["plot_bg"], edgecolor=th["border"], labelcolor=th["text"])


def style_axes(ax, th):
    ax.set_facecolor(th["plot_bg"])
    ax.tick_params(colors=th["muted"], which="both")
    ax.xaxis.label.set_color(th["text"])
    ax.yaxis.label.set_color(th["text"])
    for spine in ax.spines.values():
        spine.set_color(th["border"] if th["dark"] else th["muted"])
    # two lines (k, then its effect) so the title fits even when the window is narrow
    ax.set_title("", fontsize=11.5, color=th["text"], pad=10, linespacing=1.5)


class Scene2D:
    blit = True

    def __init__(self, fig, th, show_span, show_proj, show_labels):
        self.th = th
        self.show_span, self.show_proj, self.show_labels = show_span, show_proj, show_labels
        self.v = None
        self.lim = None
        self._n = np.array([0.0, 1.0])     # label offset direction (normal to v)

        fig.clf()
        fig.set_facecolor(th["plot_bg"])
        self.fig = fig
        ax = self.ax = fig.add_subplot(111)
        style_axes(ax, th)
        ax.set_aspect("equal")
        ax.axhline(0, color=th["axis"], lw=0.8)
        ax.axvline(0, color=th["axis"], lw=0.8)
        ax.grid(True, color=th["grid"], alpha=0.35)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        kv_c, v_c = th["kv"], th["v"]
        arrow = dict(angles="xy", scale_units="xy", scale=1)
        self._box = dict(boxstyle="round,pad=0.25", fc=th["plot_bg"], ec="none", alpha=0.85)
        label = dict(fontsize=10, fontweight="bold", ha="center", va="center",
                     bbox=self._box, zorder=6)

        # ---- static: only change when v or the axis limits change
        self.span, = ax.plot([], [], "--", color=th["span"], lw=1)
        self.v_glow, = ax.plot([], [], color=v_c, alpha=0.15, lw=11,
                               solid_capstyle="round", zorder=2)
        self.v_arrow = ax.quiver([0], [0], [0], [0], color=v_c, width=0.011, zorder=5, **arrow)
        self.v_text = ax.text(0, 0, "", color=v_c, **label)

        # ---- dynamic: change with k, redrawn every frame by blitting
        A = dict(animated=True)
        self.trail = LineCollection([], linewidths=2.6, capstyle="round", zorder=2, **A)
        ax.add_collection(self.trail, autolim=False)
        self.proj = [ax.plot([], [], ":", color=kv_c, lw=1, alpha=0.6, **A)[0] for _ in range(2)]
        self.kv_glow, = ax.plot([], [], color=kv_c, alpha=0.13, lw=9,
                                solid_capstyle="round", zorder=2, **A)
        self.kv_arrow = ax.quiver([0], [0], [1], [0], color=kv_c, width=0.006,
                                  zorder=4, alpha=0.95, **arrow, **A)
        self.zero = [ax.plot([0], [0], "o", color=kv_c, ms=16, alpha=0.2, zorder=4, **A)[0],
                     ax.plot([0], [0], "o", color=kv_c, ms=9, zorder=5, **A)[0]]
        self.kv_text = ax.text(0, 0, "", color=kv_c, **label, **A)
        ax.title.set_animated(True)

        _legend(ax, th, show_span, "lower right")
        self._trail_rgba = np.array(to_rgba(kv_c))

        # what gets painted per frame, in z-order; v's arrow and label are
        # painted again on top so k·v never covers them (as in a full draw)
        dynamic = [self.trail, *self.proj, self.kv_glow, self.kv_arrow, *self.zero,
                   self.kv_text, self.v_arrow, self.v_text]
        self.frame_artists = sorted(dynamic, key=lambda a: a.get_zorder()) + [ax.title]

    # ------------------------------------------------------------------
    def set_vector(self, v, lim):
        """Update the static layer. Returns True if a full redraw is needed."""
        if self.v is not None and lim == self.lim and np.array_equal(v, self.v):
            return False
        self.v, self.lim = v.copy(), lim
        ax = self.ax
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        mag = la.magnitude(v)
        if mag > 0:
            u = v / mag
            s = u * lim * 2
            self.span.set_data([-s[0], s[0]], [-s[1], s[1]])
            n = np.array([-u[1], u[0]])
            self._n = -n if n[1] < 0 else n          # "above" the span line
        self.span.set_visible(self.show_span and mag > 0)
        self.v_glow.set_data([0, v[0]], [0, v[1]])
        self.v_arrow.set_UVC([v[0]], [v[1]])
        p = v * 0.6 + self._n * lim * 0.11
        self.v_text.set_position((p[0], p[1]))
        self.v_text.set_text(f"v {la.fmt_vec(v, 2)}")
        self.v_text.set_visible(self.show_labels and mag > 0)
        return True

    def set_k(self, k, title, trail):
        v = self.v
        kv = k * v
        nonzero = k != 0
        self.kv_glow.set_data([0, kv[0]], [0, kv[1]])
        self.kv_glow.set_visible(nonzero)
        self.kv_arrow.set_UVC([kv[0]], [kv[1]])
        self.kv_arrow.set_visible(nonzero)
        for z in self.zero:
            z.set_visible(not nonzero)
        self.proj[0].set_data([kv[0], kv[0]], [0, kv[1]])
        self.proj[1].set_data([0, kv[0]], [kv[1], kv[1]])
        for line in self.proj:
            line.set_visible(self.show_proj and nonzero)
        if self.show_labels and nonzero and la.magnitude(v) > 0:
            q = kv - self._n * self.lim * 0.09
            # the text runs away from the origin so it never sits under v's
            # label, but turns inward near the edge so it stays on the plot
            outward = q[0] >= 0
            if abs(q[0]) > self.lim * 0.68:
                outward = not outward
            self.kv_text.set_horizontalalignment("left" if outward else "right")
            self.kv_text.set_position((q[0], q[1]))
            self.kv_text.set_text(f"{la.fmt(k, 2)}v {la.fmt_vec(kv, 2)}")
            self.kv_text.set_visible(True)
        else:
            self.kv_text.set_visible(False)
        self.ax.title.set_text(title)

        old = trail[:-1]
        if old:
            segs = np.zeros((len(old), 2, 2))
            segs[:, 1, :] = np.outer(old, v)               # tips of k_i·v, all at once
            colors = np.tile(self._trail_rgba, (len(old), 1))
            colors[:, 3] = 0.06 + 0.28 * np.arange(len(old)) / max(1, len(trail) - 1)
            self.trail.set_segments(segs)
            self.trail.set_color(colors)
            self.trail.set_visible(True)
        else:
            self.trail.set_visible(False)

    def draw_frame(self):
        for artist in self.frame_artists:
            self.fig.draw_artist(artist)

    def view(self):
        return None


class Scene3D:
    blit = True

    def __init__(self, fig, th, show_span, show_proj, show_labels, view=(25, -60)):
        self.th = th
        self.fig = fig
        self.show_span, self.show_proj, self.show_labels = show_span, show_proj, show_labels
        self.v = None
        self.lim = None
        self.v_arrow = None
        self.kv_arrow = None

        fig.clf()
        fig.set_facecolor(th["plot_bg"])
        with matplotlib.rc_context({"grid.color": th["grid"], "axes.edgecolor": th["muted"]}):
            ax = self.ax = fig.add_subplot(111, projection="3d")
        style_axes(ax, th)
        pane = to_rgba(mix(th["plot_bg"], th["text"], 0.05))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(pane)
            axis.line.set_color(th["muted"])
        ax.zaxis.label.set_color(th["text"])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(*view)

        kv_c = th["kv"]
        A = dict(animated=True)
        self.axis_lines = [ax.plot([], [], [], color=th["axis"], lw=0.6)[0] for _ in range(3)]
        self.span = ax.plot([], [], [], "--", color=th["span"], lw=1)[0]
        self.v_text = ax.text(0, 0, 0, "  v", color=th["v"], fontweight="bold")
        # dynamic (blitted)
        self.proj = [ax.plot([], [], [], ":", color=kv_c, alpha=0.6, **A)[0] for _ in range(2)]
        self.trail = [ax.plot([], [], [], color=kv_c, alpha=0.15, lw=1.5, **A)[0]
                      for _ in range(TRAIL_LEN - 1)]
        self.zero = ax.plot([0], [0], [0], "o", color=kv_c, ms=7, **A)[0]
        self.kv_text = ax.text(0, 0, 0, "", color=kv_c, fontweight="bold", **A)
        ax.title.set_animated(True)
        _legend(ax, th, show_span, "upper left")

    def set_vector(self, v, lim):
        if self.v is not None and lim == self.lim and np.array_equal(v, self.v):
            return False
        self.v, self.lim = v.copy(), lim
        ax = self.ax
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        for line, d in zip(self.axis_lines, np.eye(3) * lim):
            line.set_data_3d(*zip(-d, d))
        mag = la.magnitude(v)
        if mag > 0:
            s = v / mag * lim * 1.6
            self.span.set_data_3d([-s[0], s[0]], [-s[1], s[1]], [-s[2], s[2]])
        self.span.set_visible(self.show_span and mag > 0)
        if self.v_arrow is not None:
            self.v_arrow.remove()
        self.v_arrow = ax.quiver(0, 0, 0, *v, color=self.th["v"], lw=3.2, arrow_length_ratio=0.12)
        self.v_text.set_position_3d(v)
        self.v_text.set_visible(self.show_labels)
        return True

    def set_k(self, k, title, trail):
        v, ax = self.v, self.ax
        kv = k * v
        nonzero = k != 0
        if self.kv_arrow is not None:
            self.kv_arrow.remove()
            self.kv_arrow = None
        if nonzero:
            self.kv_arrow = ax.quiver(0, 0, 0, *kv, color=self.th["kv"], lw=1.8,
                                      arrow_length_ratio=0.08, animated=True)
        self.zero.set_visible(not nonzero)
        self.proj[0].set_data_3d([kv[0], kv[0]], [kv[1], kv[1]], [0, kv[2]])
        self.proj[1].set_data_3d([0, kv[0]], [0, kv[1]], [0, 0])
        for line in self.proj:
            line.set_visible(self.show_proj and nonzero)
        self.kv_text.set_position_3d(kv)
        self.kv_text.set_text(f"  {la.fmt(k, 2)}v")
        self.kv_text.set_visible(self.show_labels and nonzero)
        old = trail[:-1]
        for i, line in enumerate(self.trail):
            if i < len(old):
                p = old[i] * v
                line.set_data_3d([0, p[0]], [0, p[1]], [0, p[2]])
                line.set_visible(True)
            else:
                line.set_visible(False)
        ax.title.set_text(title)

    def draw_frame(self):
        # the camera matrix from the last full draw is still valid, so the
        # moving artists only need projecting with it, then painting;
        # v's arrow is painted again on top so k·v never hides it
        frame = [*self.trail, *self.proj, self.zero, self.kv_arrow, self.v_arrow,
                 self.kv_text, self.ax.title]
        for artist in frame:
            if artist is None:
                continue
            if hasattr(artist, "do_3d_projection"):
                artist.do_3d_projection()
            self.fig.draw_artist(artist)

    def view(self):
        return (self.ax.elev, self.ax.azim)
