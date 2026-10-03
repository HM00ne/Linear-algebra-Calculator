"""
animation.py
------------
A single, time-based frame clock shared by every animation in the app.

Why time-based?  The old animations advanced one step per `after(16)` call,
so whenever a frame took longer than 16 ms the whole animation slowed down
and stuttered. Here each animation gets its progress from the real clock
(elapsed / duration): a slow frame is simply skipped over, and every
animation always finishes on time.

Why one shared clock?  Several animations (tween, header shine, replay of
the algebra steps) can run at the same time. One timer drives them all, so
they update together in the same frame instead of each forcing a redraw.
"""

import sys
import time

FPS = 60
FRAME_S = 1.0 / FPS


def ease(t):
    """Smooth ease-in-out (cubic)."""
    t = max(0.0, min(1.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out(t):
    """Fast start, gentle stop (cubic) – feels right for things sliding in."""
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def enable_precise_timers():
    """On Windows the default timer tick is ~15.6 ms, which turns a 16 ms
    `after()` into ~31 ms (about 30 fps). Asking for 1 ms resolution lets
    Tk actually hit 60 fps. Returns a function that undoes the request."""
    if sys.platform != "win32":
        return lambda: None
    try:
        import ctypes
        winmm = ctypes.WinDLL("winmm")
        winmm.timeBeginPeriod(1)
        return lambda: winmm.timeEndPeriod(1)
    except (OSError, AttributeError):
        return lambda: None


class Animator:
    """Runs any number of animations on one Tk timer.

    start(duration_ms, on_frame, on_done) calls on_frame(p) once per frame,
    with p going linearly from 0 to 1 (apply ease() yourself), then on_done().

    `speed` scales every duration (Settings ▸ Motion); 0 means "reduce
    motion": animations jump straight to their end state.
    """

    def __init__(self, widget):
        self.widget = widget
        self.speed = 1.0
        self._anims = {}
        self._next_id = 0
        self._job = None
        self._next_frame = 0.0

    def start(self, duration_ms, on_frame, on_done=None):
        self._next_id += 1
        aid = self._next_id
        if self.speed == 0:
            on_frame(1.0)
            if on_done:
                on_done()
            return aid
        self._anims[aid] = (time.perf_counter(), max(duration_ms * self.speed, 1) / 1000,
                            on_frame, on_done)
        on_frame(0.0)                     # first frame immediately, no 16 ms lag
        self._schedule()
        return aid

    def cancel(self, aid):
        self._anims.pop(aid, None)

    def busy(self):
        """True while any animation is playing."""
        return bool(self._anims)

    def running(self, aid):
        return aid in self._anims

    def _schedule(self):
        if self._job is not None or not self._anims:
            return
        now = time.perf_counter()
        # aim at a steady 60 Hz grid; if we fell behind, just take the next slot
        self._next_frame = max(self._next_frame + FRAME_S, now + 0.002)
        delay = max(1, round((self._next_frame - now) * 1000))
        self._job = self.widget.after(delay, self._tick)

    def _tick(self):
        self._job = None
        now = time.perf_counter()
        try:
            for aid, (t0, dur, on_frame, on_done) in list(self._anims.items()):
                if aid not in self._anims:          # cancelled by an earlier callback
                    continue
                p = min(1.0, (now - t0) / dur)
                on_frame(p)
                if p >= 1.0 and self._anims.pop(aid, None) and on_done:
                    on_done()
        finally:
            self._schedule()                         # an error must not freeze the clock

    def stop_all(self):
        self._anims.clear()
        if self._job is not None:
            self.widget.after_cancel(self._job)
            self._job = None
