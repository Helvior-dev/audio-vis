"""
Waveform module: draws raw audio samples as a scrolling line, GPU-rendered.

Uses PyOpenGL (raw OpenGL calls) instead of moderngl, because moderngl
has no precompiled wheel for Python 3.14 on Windows yet and building it
from source requires MSVC Build Tools. PyOpenGL ships precompiled, no
compiler needed. Same GPU, same performance -- just more verbose calls
since we manage buffer/shader state manually instead of through an
object wrapper.

Runs as its own window (own event loop, own OpenGL context) so it can be
launched independently of the launcher, always-on-top, resizable, and
targets the monitor's refresh rate (144Hz) via vsync instead of a fixed
sleep -- this is what keeps it smooth instead of choppy.
"""

import sys
import ctypes
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.audio_capture import AudioCapture
from window_utils import apply_dark_titlebar

# how many samples wide the visible waveform window is;
# smaller = more "zoomed in" / reacts faster, larger = smoother but laggier
HISTORY_SAMPLES = 2048

VERTEX_SHADER = """
#version 330
in float y;
uniform float x_step;
void main() {
    float x = gl_VertexID * x_step - 1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
out vec4 color;
void main() {
    color = vec4(0.75, 0.82, 0.95, 1.0);
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

    # shaders are copied into the program at link time, safe to delete now
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


class WaveformWindow:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)  # always-on-top
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.width, self.height = 800, 300
        self.window = glfw.create_window(
            self.width, self.height, "Waveform", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync: cap render rate to monitor refresh (144Hz)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        apply_dark_titlebar(self.window)

        self.program = link_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.x_step_loc = glGetUniformLocation(self.program, "x_step")

        # VAO + VBO setup: one float per vertex (y value), x is computed
        # in the vertex shader from gl_VertexID so we never touch it here
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(
            GL_ARRAY_BUFFER, HISTORY_SAMPLES * 4, None, GL_DYNAMIC_DRAW
        )  # reserve space, filled every frame

        # attribute location 0 = "y" in the vertex shader (implicit since
        # it's the only "in" variable; layout matches declaration order)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.history = np.zeros(HISTORY_SAMPLES, dtype=np.float32)

        self.audio = AudioCapture(chunk_size=256)

    def _on_resize(self, window, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    def _update_audio(self):
        chunk = self.audio.read_chunk()  # shape (N, channels)
        mono = chunk.mean(axis=1).astype(np.float32)  # L+R -> mono
        n = len(mono)
        if n >= HISTORY_SAMPLES:
            self.history[:] = mono[-HISTORY_SAMPLES:]
        else:
            self.history = np.roll(self.history, -n)
            self.history[-n:] = mono

    def render_frame(self):
        # must bind the buffer before writing to it -- glBufferSubData
        # writes to whatever's currently bound to GL_ARRAY_BUFFER
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.history.nbytes, self.history)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glUniform1f(self.x_step_loc, 2.0 / (HISTORY_SAMPLES - 1))

        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINE_STRIP, 0, HISTORY_SAMPLES)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

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
    app = WaveformWindow()
    app.run()