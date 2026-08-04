"""
Windows-only: paint the native title bar dark instead of the default
white one, so it matches the black visualizer content below it.

GLFW doesn't expose this itself -- it's a DWM (Desktop Window Manager)
attribute, set directly on the win32 HWND after the window is created.
No-op on non-Windows platforms.

Also used by main.py to dark-theme the Tkinter launcher's title bar, so
the launcher and every visualizer window look consistent.

set_window_icon() sets the actual title-bar/taskbar/Alt-Tab icon for a
GLFW window from a PNG file (via glfw.set_window_icon(), which -- unlike
the dark-titlebar attribute above -- is cross-platform and needs no
win32 calls). This is the opposite of remove_titlebar_icon() below:
that one blanks the icon slot, this one fills it with the app's logo.
"""

import ctypes
import sys
from pathlib import Path

import glfw
import numpy as np
from PIL import Image

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


def set_window_icon(window, icon_path) -> None:
    """
    Sets a GLFW window's title-bar/taskbar/Alt-Tab icon from a PNG file.

    glfw.set_window_icon() wants a list of raw RGBA pixel arrays (one
    per size, largest-first is fine -- GLFW/the OS picks whichever is
    closest to what it needs for a given context), not a file path, so
    the PNG has to be decoded first. Pillow handles arbitrary source
    sizes/modes (this only needs to run once at startup, so the extra
    dependency is cheap); converting to RGBA up front means a palette
    or grayscale source PNG still comes out as the 4-channel layout
    GLFW expects instead of erroring or rendering with wrong colors.

    Cross-platform (unlike apply_dark_titlebar/_remove_icon above,
    which are win32-only) -- glfw.set_window_icon() is a real GLFW API,
    not a DWM/win32 workaround -- but this project only runs on Windows
    in practice (see audio_capture.py's WASAPI dependency), so no
    platform branch is needed here.

    Silently does nothing if icon_path doesn't exist or fails to
    decode, rather than raising -- a missing/corrupt icon file
    shouldn't stop a visualizer window from opening.
    """
    try:
        img = Image.open(icon_path).convert("RGBA")
        pixels = np.array(img, dtype=np.uint8)
    except (OSError, ValueError):
        return

    # this glfw binding's _GLFWimage.wrap() does:
    #   self.width, self.height, pixels = image
    #   ...
    #   self.pixels_array[i][j][k] = pixels[i][j][k]
    # i.e. after unpacking the 3-tuple it indexes `pixels` as
    # [row][col][channel] -- so it wants the (H, W, 4) array itself
    # (numpy supports that indexing pattern natively), not raw bytes
    # (a bytes object indexed like pixels[i] gives back a plain int,
    # which isn't itself subscriptable -- hence "int object is not
    # subscriptable") and not the bare array on its own either (that
    # gets torn apart by the first unpacking line instead).
    height, width = pixels.shape[:2]
    glfw.set_window_icon(window, 1, [(width, height, pixels)])


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


def set_window_icon_tk(root, icon_path):
    """
    For the Tkinter launcher window. Sets both the title-bar icon and
    the taskbar/Alt-Tab icon from a PNG file via iconphoto(), which (on
    Tk 8.6+, same floor main.py already relies on for the GitHub footer
    icon) decodes PNG natively -- no Pillow dependency needed here,
    unlike set_window_icon() above for GLFW windows.

    Returns the tk.PhotoImage the caller must keep a reference to (e.g.
    `root.icon_img = set_window_icon_tk(root, path)`) -- Tk only holds
    a weak reference internally, so if nothing in Python keeps this
    object alive, it gets garbage-collected and the icon silently
    reverts/disappears sometime after this call returns.

    Returns None (and leaves the default icon in place) if the file is
    missing or Tk can't decode it, rather than raising -- a missing
    icon shouldn't stop the launcher from starting.
    """
    import tkinter as tk

    try:
        icon_img = tk.PhotoImage(file=str(icon_path))
    except tk.TclError:
        return None
    root.iconphoto(True, icon_img)
    return icon_img