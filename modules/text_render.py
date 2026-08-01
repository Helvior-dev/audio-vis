"""
Shared bitmap-font text rendering for the OpenGL visualizer modules.

Originally written inline inside spectrum_analyzer.py (5x7 glyph bitmap,
rasterized to a texture, drawn as a textured quad). Pulled out here so
loudness.py can reuse the exact same font/renderer instead of copy-pasting
~150 lines of glyph data -- there's no font file bundled with this
from-scratch OpenGL pipeline, so this bitmap font is the only text
rendering path available to any module.
"""

import ctypes

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


# 5x7 bitmap font -- only the glyphs the app actually needs. Each glyph is
# 7 row-strings (top to bottom), '1' = lit pixel. Uppercase letters were
# added alongside the original lowercase set so panel titles like
# "LOUDNESS" / "LEVEL L/R" can be rendered in caps, matching the reference.
GLYPH_5X7 = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "#": ["01010", "01010", "11111", "01010", "11111", "01010", "01010"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "/": ["00001", "00001", "00010", "00100", "01000", "10000", "10000"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "z": ["00000", "00000", "11111", "00010", "00100", "01000", "11111"],
    "k": ["10000", "10000", "10010", "10100", "11000", "10100", "10010"],
    "d": ["00001", "00001", "01101", "10011", "10001", "10011", "01101"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "n": ["00000", "00000", "10110", "11001", "10001", "10001", "10001"],
    "t": ["01000", "01000", "11111", "01000", "01000", "01000", "00110"],
    "s": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "e": ["00000", "00000", "01110", "10001", "11111", "10000", "01111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "N": ["10001", "11001", "10101", "10101", "10011", "10001", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}

GLYPH_W = 5
GLYPH_H = 7
GLYPH_PAD = 1  # 1px transparent padding column so adjacent glyphs don't touch


def rasterize_text(text: str) -> np.ndarray:
    """
    Returns an (H, W) uint8 array (0 or 255) for the given string, glyphs
    side by side with GLYPH_PAD empty columns between them. Unknown
    characters render as blank space.
    """
    cell_w = GLYPH_W + GLYPH_PAD
    width = max(1, len(text) * cell_w)
    height = GLYPH_H
    canvas = np.zeros((height, width), dtype=np.uint8)

    for i, ch in enumerate(text):
        rows = GLYPH_5X7.get(ch, GLYPH_5X7[" "])
        x0 = i * cell_w
        for y, row in enumerate(rows):
            for x, bit in enumerate(row):
                if bit == "1":
                    canvas[y, x0 + x] = 255
    return canvas


class TextRenderer:
    """
    Uploads one texture per unique string (cached by string) and draws it
    as a textured quad at a given NDC position. Cache is small (a handful
    of labels + readouts that change slowly relative to frame rate).
    """

    def __init__(self, program: int):
        self.program = program
        self.tex_loc = glGetUniformLocation(program, "tex")
        self.color_loc = glGetUniformLocation(program, "text_color")

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL_DYNAMIC_DRAW)  # 6 verts * (pos2+uv2) * 4 bytes
        stride = 4 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self._cache: dict[str, tuple[int, int, int]] = {}  # text -> (tex_id, w, h)

    def _get_texture(self, text: str):
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        bitmap = rasterize_text(text)
        h, w = bitmap.shape
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE,
                     np.ascontiguousarray(bitmap))
        glBindTexture(GL_TEXTURE_2D, 0)

        result = (tex, w, h)
        self._cache[text] = result
        return result

    def draw(self, text: str, x_ndc: float, y_ndc: float, pixel_scale: float,
              win_w: int, win_h: int, color=(0.75, 0.75, 0.8), align: str = "left"):
        """
        Draws `text` with its top-left (or centered, if align='center') at
        (x_ndc, y_ndc). pixel_scale = how many NDC-screen-pixels wide each
        glyph pixel is (i.e. font size in real pixels).
        """
        if not text:
            return
        tex, w, h = self._get_texture(text)

        ndc_w = (w * pixel_scale / win_w) * 2.0
        ndc_h = (h * pixel_scale / win_h) * 2.0

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