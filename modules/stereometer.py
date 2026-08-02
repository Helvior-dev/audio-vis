"""
Vectorscope module (renamed from Stereometer -- "vectorscope" is the
standard term for this display in audio engineering, matches the
reference app's naming).

Core visualization (the Lissajous scatter plot) is unchanged from the
original stereometer.py -- same Mid/Side transform, same point cloud.
This pass only builds out the surrounding chrome to match the
reference layout: title bar text, diagonal L/M/R/S guide lines with
axis labels, and a labeled correlation readout ("corr 0.XX") instead
of a bare color bar.
"""

import sys
import ctypes
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from audio_capture import AudioCapture
from window_utils import apply_dark_titlebar
from text_render import TextRenderer, TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER, link_program as text_link_program

HISTORY_POINTS = 1024  # how many recent samples are plotted as dots

VERTEX_SHADER = """
#version 330
in vec2 pos;
uniform vec2 offset;
uniform float scale;
void main() {
    gl_Position = vec4(pos * scale + offset, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform vec3 color;
uniform float alpha;
out vec4 out_color;
void main() {
    out_color = vec4(color, alpha);
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


class VectorscopeWindow:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.width, self.height = 500, 500
        self.window = glfw.create_window(self.width, self.height, "Stereometer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        # lock resize to a 1:1 aspect ratio -- without this, dragging a
        # corner unevenly would stretch the diamond grid into an ellipse
        glfw.set_window_aspect_ratio(self.window, 1, 1)
        apply_dark_titlebar(self.window)

        self.program = link_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.color_loc = glGetUniformLocation(self.program, "color")
        self.alpha_loc = glGetUniformLocation(self.program, "alpha")
        self.offset_loc = glGetUniformLocation(self.program, "offset")
        self.scale_loc = glGetUniformLocation(self.program, "scale")

        # text rendering (title, axis labels, corr readout) -- separate
        # program since it samples a glyph-atlas texture instead of
        # taking a flat color uniform
        self.text_program = text_link_program(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)
        self.text = TextRenderer(self.text_program)

        # scatter points for the Lissajous field -- unchanged from the
        # original stereometer.py, this is the part we do not touch
        self.points_vao = glGenVertexArrays(1)
        glBindVertexArray(self.points_vao)
        self.points_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.points_vbo)
        glBufferData(GL_ARRAY_BUFFER, HISTORY_POINTS * 2 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glPointSize(2.0)

        # generic dynamic-quad/line VBO reused for: correlation bar,
        # diagonal guide lines, background track
        self.line_vao = glGenVertexArrays(1)
        glBindVertexArray(self.line_vao)
        self.line_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.audio = AudioCapture(chunk_size=256)
        self.points = np.zeros((HISTORY_POINTS, 2), dtype=np.float32)
        self.correlation = 0.0

    def _on_resize(self, window, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    def _update_audio(self):
        chunk = self.audio.read_chunk()  # (N, 2)
        left = chunk[:, 0]
        right = chunk[:, 1]

        mid = (left + right) * 0.5
        side = (left - right) * 0.5

        n = len(mid)
        new_points = np.stack([side, mid], axis=1).astype(np.float32)

        if n >= HISTORY_POINTS:
            self.points = new_points[-HISTORY_POINTS:]
        else:
            self.points = np.roll(self.points, -n, axis=0)
            self.points[-n:] = new_points

        if np.std(left) > 1e-6 and np.std(right) > 1e-6:
            corr = np.corrcoef(left, right)[0, 1]
            self.correlation = float(np.clip(corr, -1.0, 1.0))

    def _quad_verts(self, x0, x1, y0, y1) -> np.ndarray:
        return np.array(
            [x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1], dtype=np.float32
        )

    def _draw_line(self, x0, y0, x1, y1, color, alpha=1.0, width=1.0):
        glUniform2f(self.offset_loc, 0.0, 0.0)
        glUniform1f(self.scale_loc, 1.0)
        glUniform3f(self.color_loc, *color)
        glUniform1f(self.alpha_loc, alpha)
        glLineWidth(width)
        verts = np.array([x0, y0, x1, y1], dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_LINES, 0, 2)

    def _draw_quad(self, x0, y0, x1, y1, color, alpha=1.0):
        glUniform2f(self.offset_loc, 0.0, 0.0)
        glUniform1f(self.scale_loc, 1.0)
        glUniform3f(self.color_loc, *color)
        glUniform1f(self.alpha_loc, alpha)
        verts = self._quad_verts(x0, x1, y0, y1)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def render_frame(self):
        glClearColor(0.03, 0.03, 0.04, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)

        # ---- vectorscope field layout ----
        # the scatter field is a square centered in the upper ~75% of the
        # window; correlation bar sits in a separate strip at the bottom,
        # matching the reference screenshot's proportions
        field_cy = 0.18     # NDC y-center of the diamond field
        field_scale = 0.62  # NDC half-size of the diamond

        # diagonal guide lines forming the diamond (L-M-R-S axes), dim
        diag_color = (0.28, 0.30, 0.34)
        self._draw_line(-field_scale, field_cy, 0.0, field_cy + field_scale, diag_color, alpha=0.8)
        self._draw_line(0.0, field_cy + field_scale, field_scale, field_cy, diag_color, alpha=0.8)
        self._draw_line(field_scale, field_cy, 0.0, field_cy - field_scale, diag_color, alpha=0.8)
        self._draw_line(0.0, field_cy - field_scale, -field_scale, field_cy, diag_color, alpha=0.8)

        # ---- scatter points (untouched core visualization) ----
        glUniform2f(self.offset_loc, 0.0, field_cy)
        glUniform1f(self.scale_loc, field_scale * 0.9)
        glUniform3f(self.color_loc, 0.6, 0.85, 0.95)  # original stereometer.py color, unchanged
        glUniform1f(self.alpha_loc, 0.8)

        glBindVertexArray(self.points_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.points_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.points.nbytes, self.points)
        glDrawArrays(GL_POINTS, 0, HISTORY_POINTS)
        glBindVertexArray(self.line_vao)  # switch back for the rest of this frame

        # ---- axis labels: L / M / R / S at the four diamond tips ----
        label_color = (0.55, 0.55, 0.6)
        label_offset = field_scale + 0.09
        self.text.draw("L", -label_offset, field_cy + 0.03, pixel_scale=1.6,
                        win_w=self.width, win_h=self.height, color=label_color, align="center")
        self.text.draw("M", -0.0, field_cy + field_scale + 0.10, pixel_scale=1.6,
                        win_w=self.width, win_h=self.height, color=label_color, align="center")
        self.text.draw("R", label_offset, field_cy + 0.03, pixel_scale=1.6,
                        win_w=self.width, win_h=self.height, color=label_color, align="center")
        self.text.draw("S", -0.0, field_cy - field_scale - 0.02, pixel_scale=1.6,
                        win_w=self.width, win_h=self.height, color=label_color, align="center")
        # second "S" tag near the right diagonal, matching the reference's
        # two-S layout (S appears at both side-channel extremes)
        self.text.draw("S", label_offset - 0.03, field_cy - 0.11, pixel_scale=1.2,
                        win_w=self.width, win_h=self.height, color=label_color, align="center")

        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)

        # ---- title, top-left ----
        self.text.draw("STEREOMETER", -0.94, 0.92, pixel_scale=1.5,
                        win_w=self.width, win_h=self.height, color=(0.85, 0.85, 0.9))

        # text.draw() switches to text_program internally -- must restore
        # self.program before issuing more glUniform*/glDrawArrays calls
        # that target it, or GL rejects them (invalid operation: wrong
        # program bound for that uniform location)
        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)

        # ---- correlation bar + numeric readout, bottom ----
        bar_y0, bar_y1 = -0.86, -0.80
        bar_x0, bar_x1 = -0.7, 0.7
        self._draw_quad(bar_x0, bar_y0, bar_x1, bar_y1, (0.14, 0.14, 0.17))  # track

        x_zero = 0.0
        x_val = (self.correlation * 0.9) * ((bar_x1 - bar_x0) / 2)
        fill_color = (0.35, 0.75, 0.95) if self.correlation >= -0.3 else (0.9, 0.25, 0.2)
        if self.correlation >= 0:
            self._draw_quad(x_zero, bar_y0, x_val, bar_y1, fill_color)
        else:
            self._draw_quad(x_val, bar_y0, x_zero, bar_y1, fill_color)

        self.text.draw(
            f"CORR {self.correlation:+.2f}", 0.0, bar_y0 - 0.05, pixel_scale=1.4,
            win_w=self.width, win_h=self.height, color=(0.8, 0.8, 0.85), align="center",
        )

        glBindVertexArray(0)

    def run(self):
        try:
            while not glfw.window_should_close(self.window):
                self._update_audio()
                self.render_frame()
                glfw.swap_buffers(self.window)
                glfw.poll_events()
        finally:
            self.audio.close()
            glfw.terminate()


if __name__ == "__main__":
    app = VectorscopeWindow()
    app.run()