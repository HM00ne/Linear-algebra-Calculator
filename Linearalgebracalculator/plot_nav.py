"""
plot_nav.py
-----------
Mouse control for the plots, replacing Matplotlib's zoom / move buttons:

  2D:  drag with the left button  -> move around
       scroll wheel               -> zoom in/out around the mouse pointer
  3D:  drag with the left button  -> rotate (Matplotlib's own 3D control)
       scroll wheel               -> zoom in/out
  any: double-click, Ctrl+0, Home -> glide back to the default view

Zooming glides over a few frames instead of jumping, and wheel turns that
arrive mid-glide add up, so fast scrolling still feels continuous.
"""

import math

from animation import ease_out

ZOOM_STEP = 1.25          # one wheel notch
ZOOM_MS = 160
RESET_MS = 420
DEFAULT_VIEW = (25, -60)  # 3D camera (elevation, azimuth)


class PlotNavigator:
    def __init__(self, tab):
        self.tab = tab
        self.canvas = tab.canvas
        self.widget = self.canvas.get_tk_widget()
        self._drag = None
        self._glide = None
        self._target = None          # limits the current zoom glide is heading to
        for name, handler in (("button_press_event", self._press),
                              ("button_release_event", self._release),
                              ("motion_notify_event", self._move),
                              ("scroll_event", self._scroll)):
            self.canvas.mpl_connect(name, handler)

    # ------------------------------------------------------------- helpers
    def _axes(self):
        scene = self.tab.scene
        return getattr(scene, "ax", None) if scene is not None else None

    @staticmethod
    def _is_3d(ax):
        return ax.name == "3d"

    @staticmethod
    def _lims(ax):
        lims = [ax.get_xlim(), ax.get_ylim()]
        if ax.name == "3d":
            lims.append(ax.get_zlim())
        return lims

    @staticmethod
    def _set_lims(ax, lims):
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        if len(lims) == 3:
            ax.set_zlim(*lims[2])

    def _redraw(self):
        self.tab.request_full_draw()

    def _stop_glide(self):
        if self._glide is not None:
            self.tab.anim.cancel(self._glide)
            self._glide = None

    def _glide_to(self, ax, target, ms, view=None):
        """Animate the limits (and optionally the 3D camera) to `target`."""
        self._stop_glide()
        start = self._lims(ax)
        view0 = (ax.elev, ax.azim) if view is not None else None
        self._target = target

        def frame(p):
            t = ease_out(p)
            self._set_lims(ax, [(a0 + (b0 - a0) * t, a1 + (b1 - a1) * t)
                                for (a0, a1), (b0, b1) in zip(start, target)])
            if view is not None:
                ax.view_init(view0[0] + (view[0] - view0[0]) * t,
                             view0[1] + (_wrap(view[1] - view0[1])) * t)
            self._redraw()

        def done():
            self._glide = self._target = None

        gid = self.tab.anim.start(ms, frame, done)
        self._glide = gid if self.tab.anim.running(gid) else None

    # --------------------------------------------------------------- mouse
    def _press(self, e):
        ax = self._axes()
        if ax is None or e.inaxes is not ax:
            return
        if e.dblclick:
            self.tab.reset_view()
            return
        if e.button == 1 and not self._is_3d(ax):      # 3D: Matplotlib rotates
            self._stop_glide()
            self._drag = (e.x, e.y, ax.get_xlim(), ax.get_ylim())
            self.widget.configure(cursor="fleur")

    def _move(self, e):
        if self._drag is None:
            return
        ax = self._axes()
        if ax is None:
            return
        x0, y0, (xa, xb), (ya, yb) = self._drag
        box = ax.bbox
        dx = (e.x - x0) * (xb - xa) / box.width
        dy = (e.y - y0) * (yb - ya) / box.height
        ax.set_xlim(xa - dx, xb - dx)
        ax.set_ylim(ya - dy, yb - dy)
        self._redraw()

    def _release(self, _e):
        if self._drag is not None:
            self._drag = None
            self.widget.configure(cursor="")

    def _scroll(self, e):
        ax = self._axes()
        if ax is None or e.inaxes is not ax:
            return
        factor = 1 / ZOOM_STEP if e.step > 0 else ZOOM_STEP
        base = self._target or self._lims(ax)           # keep adding up mid-glide
        if self._is_3d(ax):
            target = [((a + b) / 2 - (b - a) / 2 * factor, (a + b) / 2 + (b - a) / 2 * factor)
                      for a, b in base]
        else:
            cx, cy = e.xdata, e.ydata                    # zoom around the pointer
            target = [(c + (a - c) * factor, c + (b - c) * factor)
                      for (a, b), c in zip(base, (cx, cy))]
        self._glide_to(ax, target, ZOOM_MS)

    # ---------------------------------------------------------------- reset
    def reset(self, lim):
        """Glide back to the symmetric default view ±lim (and default camera)."""
        ax = self._axes()
        if ax is None:
            return
        n = 3 if self._is_3d(ax) else 2
        view = DEFAULT_VIEW if n == 3 else None
        self._glide_to(ax, [(-lim, lim)] * n, RESET_MS, view)


def _wrap(deg):
    """Shortest way round for an angle difference."""
    return (deg + 180) % 360 - 180 if math.isfinite(deg) else 0.0
