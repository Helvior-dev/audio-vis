"""
Stereometer module: Lissajous-style stereo field display + correlation meter.

Standard technique: plot Mid (L+R) on Y and Side (L-R) on X. A mono
signal (L==R) collapses to a vertical line since Side=0. Fully
out-of-phase content spreads horizontally. This is the same
transform MiniMeters' "Linear" display mode uses.

Correlation meter (bottom bar) shows the same information as a single
number: +1 = perfectly in-phase/mono, 0 = uncorrelated/wide, -1 = out
of phase (potential mono-compatibility problem -- bass in particular
being out of phase is a real mixing issue, not just a visual curiosity).
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
out vec4 out_color;
void main() {
    out_color = vec4(color, 1.0);
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


class StereometerWindow:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.width, self.height = 500, 550
        self.window = glfw.create_window(self.width, self.height, "Stereometer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        apply_dark_titlebar(self.window)

        self.program = link_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.color_loc = glGetUniformLocation(self.program, "color")
        self.offset_loc = glGetUniformLocation(self.program, "offset")
        self.scale_loc = glGetUniformLocation(self.program, "scale")

        # scatter points for the Lissajous field
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

        # a single dynamic quad, reused for the correlation bar
        self.bar_vao = glGenVertexArrays(1)
        glBindVertexArray(self.bar_vao)
        self.bar_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.bar_vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

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

        # Pearson correlation between L and R over this chunk.
        # +1 = identical (mono), -1 = perfectly inverted, 0 = uncorrelated.
        if np.std(left) > 1e-6 and np.std(right) > 1e-6:
            corr = np.corrcoef(left, right)[0, 1]
            self.correlation = float(np.clip(corr, -1.0, 1.0))
        # if either channel is silent, correlation is undefined -- hold last value

    def _quad_verts(self, x0, x1, y0, y1) -> np.ndarray:
        return np.array(
            [x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1], dtype=np.float32
        )

    def render_frame(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        # Lissajous field occupies the top ~80% of the window, correlation
        # bar occupies the bottom ~15%, with a small gap between them.
        glUniform2f(self.offset_loc, 0.0, 0.15)
        glUniform1f(self.scale_loc, 0.8)
        glUniform3f(self.color_loc, 0.6, 0.85, 0.95)

        glBindVertexArray(self.points_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.points_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.points.nbytes, self.points)
        glDrawArrays(GL_POINTS, 0, HISTORY_POINTS)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # correlation bar: centered at correlation=0, fills toward +1 (right) or -1 (left)
        glUniform2f(self.offset_loc, 0.0, 0.0)
        glUniform1f(self.scale_loc, 1.0)

        bar_y0, bar_y1 = -0.95, -0.8
        x_zero = 0.0
        x_val = self.correlation * 0.9  # 0.9 so it doesn't touch the window edge at +-1

        glBindVertexArray(self.bar_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.bar_vbo)

        if self.correlation >= 0:
            verts = self._quad_verts(x_zero, x_val, bar_y0, bar_y1)
        else:
            verts = self._quad_verts(x_val, x_zero, bar_y0, bar_y1)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

        # green when correlated (safe, mono-compatible), red when
        # anti-correlated (phase issue -- bass cancellation risk in mono)
        if self.correlation < -0.3:
            glUniform3f(self.color_loc, 0.9, 0.25, 0.2)
        else:
            glUniform3f(self.color_loc, 0.4, 0.8, 0.45)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
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
    app = StereometerWindow()
    app.run()