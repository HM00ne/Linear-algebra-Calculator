"""
ui_scale.py
-----------
Sharp, correctly sized UI on high-resolution screens.

By default Windows treats a Tk app as "DPI-unaware": with display scaling at
125 % on a 1920×1080 screen, the app is drawn as if the screen were
1536×864 and then stretched up, which looks blurry. enable_dpi_awareness()
asks Windows for the real pixels instead. Fonts (sized in points) and the
Matplotlib plots then scale by themselves; anything the code sizes in raw
pixels goes through px(), which multiplies by the screen's scale factor S
(1.0 at 100 %, 1.25 at 125 %, 1.5 at 150 % ...).
"""

import sys

S = 1.0


def enable_dpi_awareness():
    """Call before the first Tk window is created."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # 1 = "system aware": real pixels on the main screen; Windows still
        # scales the window if it is dragged to a screen with another scaling
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (OSError, AttributeError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()      # Windows 7 / 8
        except (OSError, AttributeError):
            pass


def init(root):
    """Read the scale factor from the first window (96 dpi = 100 %)."""
    global S
    S = round(root.winfo_fpixels("1i") / 96.0, 2) or 1.0
    return S


def px(n):
    """A size designed in 100 %-scaling pixels, in real screen pixels."""
    return int(round(n * S))
