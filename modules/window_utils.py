"""
Windows-only: paint the native title bar dark instead of the default
white one, so it matches the black visualizer content below it.

GLFW doesn't expose this itself -- it's a DWM (Desktop Window Manager)
attribute, set directly on the win32 HWND after the window is created.
No-op on non-Windows platforms.

Also used by main.py to dark-theme the Tkinter launcher's title bar, so
the launcher and every visualizer window look consistent.
"""

import ctypes
import sys

import glfw

# available on Windows 10 1809+ and all of Windows 11
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19  # pre-20H1 builds used this value

WM_SETICON = 0x0080
ICON_SMALL = 0
ICON_BIG = 1
GCLP_HICON = -14
GCLP_HICONSM = -34


def _set_dark_mode(hwnd: int) -> None:
    """Low-level: flip the DWM dark-titlebar attribute for a raw HWND."""
    if not hwnd:
        return
    dwmapi = ctypes.windll.dwmapi
    value = ctypes.c_int(1)  # 1 = dark, 0 = light
    for attribute in (DWMWA_USE_IMMERSIVE_DARK_MODE, DWMWA_USE_IMMERSIVE_DARK_MODE_OLD):
        result = dwmapi.DwmSetWindowAttribute(
            hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
        )
        if result == 0:
            break  # succeeded, no need to try the fallback attribute id


def _remove_icon(hwnd: int) -> None:
    """
    Low-level: strip the title-bar/taskbar icon for a raw HWND.

    glfw.set_window_icon(window, 0, []) is not reliable here -- GLFW's
    "clear icon" path only resets to whatever icon the OS considers
    default for the window class, it doesn't guarantee no icon shows.
    Sending WM_SETICON with a NULL HICON directly is what every native
    Win32 app uses to actually blank the icon slot, for both the small
    (title bar) and big (Alt-Tab/taskbar) icon.
    """
    if not hwnd:
        return
    user32 = ctypes.windll.user32
    user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, 0)
    user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, 0)
    # also clear the window-class icon -- some Windows builds fall back
    # to this if WM_SETICON's per-window value is unset
    user32.SetClassLongPtrW(hwnd, GCLP_HICON, 0)
    user32.SetClassLongPtrW(hwnd, GCLP_HICONSM, 0)


def apply_dark_titlebar(window) -> None:
    """For GLFW windows (the visualizer modules)."""
    if sys.platform != "win32":
        return
    hwnd = glfw.get_win32_window(window)
    _set_dark_mode(hwnd)


def remove_titlebar_icon(window) -> None:
    """For GLFW windows -- blanks the title bar / taskbar icon."""
    if sys.platform != "win32":
        return
    hwnd = glfw.get_win32_window(window)
    _remove_icon(hwnd)


def apply_dark_titlebar_tk(root) -> None:
    """
    For the Tkinter launcher window. Tkinter's winfo_id() returns the
    handle of an internal child window, not the actual top-level frame
    that owns the title bar -- GetParent() walks up to the real
    top-level HWND that DWM needs.
    """
    if sys.platform != "win32":
        return
    root.update_idletasks()  # make sure the window exists before grabbing its handle
    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    _set_dark_mode(hwnd)