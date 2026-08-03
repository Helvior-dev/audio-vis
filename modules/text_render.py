"""
Shared text rendering for the OpenGL visualizer modules, backed by
Windows GDI instead of a hand-rolled bitmap font.

Font size is resolved to actual on-screen pixels every draw() call
(via win_h, which every caller already passes in) rather than a fixed
constant. Earlier versions rendered the GDI bitmap once at a constant
pixel height and then stretched that same texture over whatever quad
size the NDC coordinates worked out to. That's fine at the window size
it was tuned for, but the moment the window is resized larger (maximize
/ fullscreen), the same fixed-resolution texture gets magnified well
past its native size and comes out visibly blurry -- vector-drawn
shapes (the diamond, bars) stay crisp because the GPU rasterizes them
fresh every frame at whatever resolution, but a bitmap texture doesn't
get that for free. Resolving font_px from win_h means the GDI bitmap is
always rendered near its final on-screen size, so it stays sharp at
any window size.

Windows-only (GDI). This project is already Windows-only (WASAPI
loopback in audio_capture.py), so that's not a new constraint.
"""

import ctypes
from ctypes import wintypes

import numpy as np
from OpenGL.GL import *

TEXT_VERTEX_SHADER = """
#version 330
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
out vec2 frag_uv;
void main() {
    frag_uv = uv;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

TEXT_FRAGMENT_SHADER = """
#version 330
in vec2 frag_uv;
uniform sampler2D tex;
uniform vec3 text_color;
out vec4 out_color;
void main() {
    float a = texture(tex, frag_uv).r;
    out_color = vec4(text_color, a);
}
"""


def compile_shader(source: str, shader_type) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error: {error}")
    return shader


def link_program(vertex_src: str, fragment_src: str) -> int:
    vs = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program link error: {error}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


# pixel_scale historically meant "multiply this base size" -- kept as a
# reference constant so existing pixel_scale call-site values (tuned by
# eye against the old fixed-size renderer) keep roughly the same visual
# proportions relative to each other, now anchored to a fraction of the
# window height instead of an absolute pixel count.
REFERENCE_WINDOW_HEIGHT = 500  # the size stereometer.py's calls were tuned against
BASE_FONT_PX_AT_REFERENCE = 14
FONT_FACE = "Segoe UI"  # matches main.py's Tkinter launcher font

gdi32 = ctypes.windll.gdi32
user32 = ctypes.windll.user32

TRANSPARENT = 1
DIB_RGB_COLORS = 0
BI_RGB = 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3),
    ]


def _measure_text(hdc, text: str) -> tuple[int, int]:
    size = wintypes.SIZE()
    gdi32.GetTextExtentPoint32W(hdc, text, len(text), ctypes.byref(size))
    return max(1, size.cx), max(1, size.cy)


def render_text_to_alpha(text: str, font_px: int) -> np.ndarray:
    """
    Renders `text` with GDI at the given pixel height and returns an
    (H, W) uint8 grayscale array (0..255) suitable for use as an alpha
    channel -- white text on black background, so pixel brightness
    directly becomes glyph coverage.
    """
    screen_dc = user32.GetDC(0)
    measure_dc = gdi32.CreateCompatibleDC(screen_dc)
    font = gdi32.CreateFontW(
        -font_px, 0, 0, 0, 400, 0, 0, 0,
        1,  # DEFAULT_CHARSET
        0, 0,
        4,  # CLEARTYPE_QUALITY (falls back gracefully if ClearType is off)
        0,
        FONT_FACE,
    )
    old_font = gdi32.SelectObject(measure_dc, font)
    w, h = _measure_text(measure_dc, text)
    gdi32.SelectObject(measure_dc, old_font)
    gdi32.DeleteDC(measure_dc)

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h  # negative = top-down
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 24
    bmi.bmiHeader.biCompression = BI_RGB

    dc = gdi32.CreateCompatibleDC(screen_dc)
    bits_ptr = ctypes.c_void_p()
    dib = gdi32.CreateDIBSection(
        dc, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(bits_ptr), None, 0
    )
    old_bmp = gdi32.SelectObject(dc, dib)
    old_font = gdi32.SelectObject(dc, font)

    gdi32.SetBkMode(dc, TRANSPARENT)
    gdi32.SetTextColor(dc, 0x00FFFFFF)
    black_brush = gdi32.GetStockObject(4)  # BLACK_BRUSH
    rect = wintypes.RECT(0, 0, w, h)
    user32.FillRect(dc, ctypes.byref(rect), black_brush)

    gdi32.TextOutW(dc, 0, 0, text, len(text))
    gdi32.GdiFlush()

    stride = ((w * 3 + 3) // 4) * 4
    buf = ctypes.cast(bits_ptr, ctypes.POINTER(ctypes.c_ubyte * (stride * h))).contents
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(h, stride)[:, : w * 3].reshape(h, w, 3)
    alpha = raw.max(axis=2).astype(np.uint8)

    gdi32.SelectObject(dc, old_bmp)
    gdi32.SelectObject(dc, old_font)
    gdi32.DeleteObject(dib)
    gdi32.DeleteDC(dc)
    gdi32.DeleteObject(font)
    user32.ReleaseDC(0, screen_dc)

    return alpha


class TextRenderer:
    """
    Uploads one texture per unique (string, font_px) pair (cached) and
    draws it as a textured quad at a given NDC position. font_px is
    resolved from the *current* window height every draw() call, so the
    cache naturally grows one entry per (text, size-at-that-window-size)
    combination -- sizes repeat whenever the window settles at a given
    size (including back-and-forth during a drag-resize, since each
    discrete pixel height it passes through reuses its own cache slot).
    """

    def __init__(self, program: int):
        self.program = program
        self.tex_loc = glGetUniformLocation(program, "tex")
        self.color_loc = glGetUniformLocation(program, "text_color")

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL_DYNAMIC_DRAW)
        stride = 4 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self._cache: dict[tuple[str, int], tuple[int, int, int]] = {}
        # simple cap so a continuously-resizing window (many distinct
        # font_px values) can't grow the texture cache without bound
        self._max_cache_entries = 400

    def _get_texture(self, text: str, font_px: int):
        key = (text, font_px)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if len(self._cache) >= self._max_cache_entries:
            # drop an arbitrary entry (oldest-inserted in dict order) --
            # this is a soft cap for pathological continuous-resize
            # cases, not a hot path in normal use
            oldest_key = next(iter(self._cache))
            old_tex = self._cache.pop(oldest_key)[0]
            glDeleteTextures([old_tex])

        bitmap = render_text_to_alpha(text, font_px)
        h, w = bitmap.shape
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE,
                     np.ascontiguousarray(bitmap))
        glBindTexture(GL_TEXTURE_2D, 0)

        result = (tex, w, h)
        self._cache[key] = result
        return result

    def draw(self, text: str, x_ndc: float, y_ndc: float, pixel_scale: float,
              win_w: int, win_h: int, color=(0.75, 0.75, 0.8), align: str = "left"):
        """
        Draws `text` with its top-left (or centered, if align='center') at
        (x_ndc, y_ndc). font_px is resolved from win_h so the same
        pixel_scale value always produces the same on-screen size
        relative to the window, and the GDI bitmap is rendered near its
        true final resolution regardless of window size -- this is what
        keeps text sharp when the window is maximized/fullscreen instead
        of stretching a small fixed-size texture.
        """
        if not text:
            return
        size_at_reference = BASE_FONT_PX_AT_REFERENCE * pixel_scale
        font_px = max(1, round(size_at_reference * (win_h / REFERENCE_WINDOW_HEIGHT)))
        tex, w, h = self._get_texture(text, font_px)

        ndc_w = (w / win_w) * 2.0
        ndc_h = (h / win_h) * 2.0

        x0 = x_ndc - ndc_w / 2.0 if align == "center" else x_ndc
        x1 = x0 + ndc_w
        y1 = y_ndc
        y0 = y_ndc - ndc_h

        verts = np.array([
            x0, y0, 0.0, 1.0,
            x1, y0, 1.0, 1.0,
            x1, y1, 1.0, 0.0,
            x0, y0, 0.0, 1.0,
            x1, y1, 1.0, 0.0,
            x0, y1, 0.0, 0.0,
        ], dtype=np.float32)

        glUseProgram(self.program)
        glUniform3f(self.color_loc, *color)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(self.tex_loc, 0)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)