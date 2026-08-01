"""
Windows-only: paint the native title bar dark instead of the default
white one, so it matches the black visualizer content below it.

GLFW doesn't expose this itself -- it's a DWM (Desktop Window Manager)
attribute, set directly on the win32 HWND after the window is created.
No-op on non-Windows platforms (glfw.get_win32_window returns 0 there).
"""

import ctypes
import sys

import glfw

# available on Windows 10 1809+ and all of Windows 11
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19  # pre-20H1 builds used this value


def apply_dark_titlebar(window) -> None:
    if sys.platform != "win32":
        return

    hwnd = glfw.get_win32_window(window)
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