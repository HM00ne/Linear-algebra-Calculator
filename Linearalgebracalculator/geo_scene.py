"""
geo_scene.py
------------
A reusable vector picture for the calculator tabs (dot product, lengths,
two vectors, law of cosines...).

Tabs draw "immediate mode": every frame they call
    scene.begin()
    scene.arrow("u", u, color) ; scene.label(...) ; scene.polygon(...) ...
    scene.finish(lim)
Behind the scenes each key keeps one long-lived Matplotlib artist that only
has its data updated, and anything not drawn this frame is hidden. The
axes, grid and legend are the cached static layer; all keyed artists are
"animated", so frames are blitted (see plot_scene.py for why that is fast).
"""

import math

import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plot_scene import style_axes
from themes import mix


def roles(th):
    """Colour roles shared by every vector tab."""
    return dict(u=th["v"], v=th["kv"], w=th["w"], p=th["proj"], a=th["accent"],
                x=mix(th["v"], th["kv"], 0.5),            # cross product: between u and v
                text=th["text"], muted=th["muted"], span=th["span"], axis=th["axis"],
                bg=th["plot_bg"])


def unit(x):
    m = float(np.linalg.norm(x))
    return None if m == 0 else np.asarray(x, float) / m


def arc_points(a, b, r, n=40):
    """Points of the circular arc of radius r from direction a to direction b
    (the angle between two vectors). Works in 2D and 3D; None if undefined."""
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return None
    c = float(np.clip(np.dot(ua, ub), -1, 1))
    theta = math.acos(c)
    if theta < 1e-6:
        return None
    t = np.linspace(0, 1, n)[:, None]
    if math.sin(theta) < 1e-6:                        # opposite: any half circle
        if len(ua) == 2:
            perp = np.array([-ua[1], ua[0]])
        else:
            e = np.eye(3)[int(np.argmin(np.abs(ua)))]
            perp = unit(np.cross(ua, e))
        return r * (np.cos(t * math.pi) * ua + np.sin(t * math.pi) * perp)
    s = math.sin(theta)
    return r * (np.sin((1 - t) * theta) / s * ua + np.sin(t * theta) / s * ub)


def right_angle_points(corner, d1, d2, size):
    """The little square that marks a right angle at `corner`."""
    a, b = unit(d1), unit(d2)
    if a is None or b is None:
        return None
    corner = np.asarray(corner, float)
    return np.array([corner + a * size, corner + (a + b) * size, corner + b * size])


class _Scene:
    blit = True

    def __init__(self, fig, th):
        self.fig, self.th = fig, th
        self.col = roles(th)
        self.items = {}               # key -> list of artists
        self.used = set()
        self.lim = None
        self._order = None
        fig.clf()
        fig.set_facecolor(th["plot_bg"])

    # -- immediate-mode bookkeeping
    def begin(self):
        self.used = set()

    def _get(self, key, make):
        arts = self.items.get(key)
        if arts is None:
            arts = self.items[key] = make()
            self._order = None
        self.used.add(key)
        return arts

    def finish(self, lim):
        """Hide what was not drawn; returns True if a full redraw is needed."""
        for key, arts in self.items.items():
            shown = key in self.used
            for a in arts:
                a.set_visible(shown)
        if lim == self.lim:
            return False
        self.lim = lim
        self._set_lim(lim)
        return True

    def title(self, text):
        self.ax.title.set_text(text)

    def _legend(self, legend, loc):
        if not legend:
            return
        handles = []
        for label, color, style in legend:
            if style == "fill":
                handles.append(Patch(facecolor=to_rgba(color, 0.25), edgecolor=color))
            else:
                handles.append(Line2D([], [], color=color, lw=2.5 if style == "line" else 1.4,
                                      ls={"dash": "--", "dot": ":"}.get(style, "-")))
        th = self.th
        self.ax.legend(handles, [l for l, _, _ in legend], loc=loc, fontsize=9, framealpha=0.9,
                       facecolor=th["plot_bg"], edgecolor=th["border"], labelcolor=th["text"])

    def _artists(self):
        if self._order is None:
            arts = [a for group in self.items.values() for a in group]
            self._order = sorted(arts, key=lambda a: a.get_zorder())
        return self._order

    def view(self):
        return None


class GeoScene2D(_Scene):
    def __init__(self, fig, th, legend=(), view=None):
        super().__init__(fig, th)
        ax = self.ax = fig.add_subplot(111)
        style_axes(ax, th)
        ax.set_aspect("equal")
        ax.axhline(0, color=th["axis"], lw=0.8)
        ax.axvline(0, color=th["axis"], lw=0.8)
        ax.grid(True, color=th["grid"], alpha=0.35)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.title.set_animated(True)
        self._legend(legend, "lower right")
        self._box = dict(boxstyle="round,pad=0.22", fc=th["plot_bg"], ec="none", alpha=0.85)

    def _set_lim(self, lim):
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)

    # -- drawing primitives
    def arrow(self, key, vec, color, origin=(0, 0), width=0.009, glow=True, zorder=4, alpha=1.0):
        vec, origin = np.asarray(vec, float), np.asarray(origin, float)
        ax = self.ax

        def make():
            arts = []
            if glow:
                arts.append(ax.plot([], [], color=color, alpha=0.14, lw=width * 1000,
                                    solid_capstyle="round", zorder=zorder - 0.5,
                                    animated=True)[0])
            arts.append(ax.quiver([0], [0], [1], [0], angles="xy", scale_units="xy", scale=1,
                                  color=color, width=width, zorder=zorder, alpha=alpha,
                                  animated=True))
            return arts

        arts = self._get(key, make)
        if not np.any(vec):
            self.used.discard(key)                 # a zero vector has no arrow
            return
        if glow:
            arts[0].set_data([origin[0], origin[0] + vec[0]], [origin[1], origin[1] + vec[1]])
        q = arts[-1]
        q.set_offsets([origin[:2]])
        q.set_UVC([vec[0]], [vec[1]])

    def path(self, key, pts, color, ls="-", lw=1.4, alpha=1.0, zorder=3):
        if pts is None:
            return
        pts = np.asarray(pts, float)
        line = self._get(key, lambda: [self.ax.plot([], [], color=color, ls=ls, lw=lw,
                                                    alpha=alpha, zorder=zorder,
                                                    animated=True)[0]])[0]
        line.set_data(pts[:, 0], pts[:, 1])

    def segment(self, key, p0, p1, color, **kw):
        self.path(key, [p0, p1], color, **kw)

    def polygon(self, key, pts, color, alpha=0.13, zorder=1):
        if pts is None:
            return
        poly = self._get(key, lambda: [self.ax.add_patch(Polygon(
            np.zeros((3, 2)), closed=True, fc=to_rgba(color, alpha), ec="none",
            zorder=zorder, animated=True))])[0]
        poly.set_xy(np.asarray(pts, float)[:, :2])

    def point(self, key, p, color, ms=6, zorder=6):
        dot = self._get(key, lambda: [self.ax.plot([0], [0], "o", color=color, ms=ms,
                                                   zorder=zorder, animated=True)[0]])[0]
        dot.set_data([p[0]], [p[1]])

    def label(self, key, p, text, color, size=10, bold=True, ha="center", va="center",
              box=True, zorder=7):
        t = self._get(key, lambda: [self.ax.text(
            0, 0, "", color=color, fontsize=size, fontweight="bold" if bold else "normal",
            ha=ha, va=va, bbox=self._box if box else None, zorder=zorder, animated=True)])[0]
        t.set_position((p[0], p[1]))
        t.set_text(text)
        t.set_horizontalalignment(ha)

    def draw_frame(self):
        draw = self.fig.draw_artist
        for a in self._artists():
            if a.get_visible():
                draw(a)
        draw(self.ax.title)


class GeoScene3D(_Scene):
    def __init__(self, fig, th, legend=(), view=(25, -60)):
        super().__init__(fig, th)
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
        ax.view_init(*(view or (25, -60)))
        self._axes = [ax.plot([], [], [], color=th["axis"], lw=0.6)[0] for _ in range(3)]
        ax.title.set_animated(True)
        self._legend(legend, "upper left")

    def _set_lim(self, lim):
        ax = self.ax
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        for line, d in zip(self._axes, np.eye(3) * lim):
            line.set_data_3d(*zip(-d, d))

    def view(self):
        return (self.ax.elev, self.ax.azim)

    # -- drawing primitives
    def arrow(self, key, vec, color, origin=(0, 0, 0), width=0.009, glow=True, zorder=4,
              alpha=1.0):
        vec, origin = np.asarray(vec, float), np.asarray(origin, float)
        old = self.items.pop(key, None)            # 3D arrows are rebuilt each frame
        if old:
            old[0].remove()
            self._order = None
        if not np.any(vec):
            return
        q = self.ax.quiver(*origin, *vec, color=color, lw=width * 300, alpha=alpha,
                           arrow_length_ratio=0.1, animated=True)
        q.set_zorder(zorder)
        self._get(key, lambda: [q])

    def path(self, key, pts, color, ls="-", lw=1.4, alpha=1.0, zorder=3):
        if pts is None:
            return
        pts = np.asarray(pts, float)
        line = self._get(key, lambda: [self.ax.plot([], [], [], color=color, ls=ls, lw=lw,
                                                    alpha=alpha, zorder=zorder,
                                                    animated=True)[0]])[0]
        line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])

    def segment(self, key, p0, p1, color, **kw):
        self.path(key, [p0, p1], color, **kw)

    def polygon(self, key, pts, color, alpha=0.13, zorder=1):
        if pts is None:
            return

        def make():
            poly = Poly3DCollection([np.zeros((3, 3))], facecolor=to_rgba(color, alpha),
                                    edgecolor="none", animated=True)
            self.ax.add_collection3d(poly)
            return [poly]

        self._get(key, make)[0].set_verts([np.asarray(pts, float)])

    def point(self, key, p, color, ms=6, zorder=6):
        dot = self._get(key, lambda: [self.ax.plot([0], [0], [0], "o", color=color, ms=ms,
                                                   zorder=zorder, animated=True)[0]])[0]
        dot.set_data_3d([p[0]], [p[1]], [p[2]])

    def label(self, key, p, text, color, size=10, bold=True, ha="center", va="center",
              box=False, zorder=7):
        t = self._get(key, lambda: [self.ax.text(
            0, 0, 0, "", color=color, fontsize=size, fontweight="bold" if bold else "normal",
            ha=ha, va=va, zorder=zorder, animated=True)])[0]
        t.set_position_3d(p)
        t.set_text(text)
        t.set_horizontalalignment(ha)

    def draw_frame(self):
        # camera matrix from the last full draw is still valid: project, paint
        draw = self.fig.draw_artist
        for a in self._artists():
            if a.get_visible():
                if hasattr(a, "do_3d_projection"):
                    a.do_3d_projection()
                draw(a)
        draw(self.ax.title)
