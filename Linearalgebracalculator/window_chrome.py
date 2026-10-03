"""
window_chrome.py
----------------
Makes the Windows title bar follow the app's colour theme.

Tk cannot style the title bar itself (Windows draws it), so we ask the
Desktop Window Manager directly through ctypes:
  * Windows 10 (build 19041+) and 11: dark or light title bar.
  * Windows 11: the title bar can also take the theme's exact colours.
On other systems, or older Windows, this quietly does nothing.
"""

import sys

DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19     # Windows 10 builds before 19041
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_BORDER_COLOR = 34                    # Windows 11 only
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36
WM_NCACTIVATE = 0x0086


GCL_STYLE = -26
CS_VREDRAW, CS_HREDRAW = 0x1, 0x2


def smooth_resizing(window):
    """Tk registers its window classes with CS_HREDRAW | CS_VREDRAW, which
    tells Windows to repaint every widget whenever the window changes size,
    even widgets that did not move (~110 repaints per resize step here).
    Tk widgets already redraw themselves when their own size changes, so
    those flags only cost time: clear them (for the whole class, once)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.GetClassLongPtrW.restype = ctypes.c_size_t
        user32.GetClassLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.SetClassLongPtrW.restype = ctypes.c_size_t
        user32.SetClassLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        window.update_idletasks()
        child = window.winfo_id()                       # class "TkChild"
        top = user32.GetParent(child) or child          # class "TkTopLevel"
        for hwnd in (top, child):
            style = user32.GetClassLongPtrW(hwnd, GCL_STYLE)
            user32.SetClassLongPtrW(hwnd, GCL_STYLE, style & ~(CS_HREDRAW | CS_VREDRAW))
    except (OSError, AttributeError, ValueError):
        pass


def _colorref(hex_color):
    """'#rrggbb' -> Windows COLORREF (0x00bbggrr)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return r | (g << 8) | (b << 16)


def style_titlebar(window, dark, caption=None, text=None, border=None):
    if sys.platform != "win32":
        return
    try:
        import ctypes
        user32, dwm = ctypes.windll.user32, ctypes.windll.dwmapi
        window.update_idletasks()                      # make sure the frame exists
        hwnd = user32.GetParent(window.winfo_id()) or window.winfo_id()

        def set_attr(attr, value):
            v = ctypes.c_int(value)
            return dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v)) == 0

        if not set_attr(DWMWA_USE_IMMERSIVE_DARK_MODE, int(dark)):
            set_attr(DWMWA_USE_IMMERSIVE_DARK_MODE_OLD, int(dark))
        for attr, colour in ((DWMWA_CAPTION_COLOR, caption), (DWMWA_TEXT_COLOR, text),
                             (DWMWA_BORDER_COLOR, border)):
            if colour:
                set_attr(attr, _colorref(colour))      # fails harmlessly before Windows 11

        # Windows 10 only repaints the title bar when the window's active state
        # changes, so nudge it: inactive -> active (no visible flicker)
        if user32.GetForegroundWindow() == hwnd:
            user32.SendMessageW(hwnd, WM_NCACTIVATE, 0, 0)
            user32.SendMessageW(hwnd, WM_NCACTIVATE, 1, 0)
    except (OSError, AttributeError, ValueError):
        pass
