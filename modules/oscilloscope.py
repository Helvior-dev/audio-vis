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
from audio_capture import AudioCapture
from window_utils import apply_dark_titlebar
from text_render import TextRenderer, TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER, link_program as text_link_program

# how many samples wide the visible waveform window is;
# smaller = more "zoomed in" / reacts faster, larger = smoother but laggier
HISTORY_SAMPLES = 2048

# Kept as fairly short lines rather than a few long paragraphs -- the
# window is wide and short (800x300 default, and user-resizable), so
# short lines centered as a block use the width the overlay actually
# has instead of collapsing into a narrow strip in one corner.
HELP_TITLE = "OSCILLOSCOPE"

HELP_TEXT_LINES = [
    "Draws the raw shape of the sound wave as it plays,",
    "moving left to right in real time.",
    "",
    "Up and down movement is how loud the sound is at that",
    "instant -- tall spikes are loud, a flat line near the",
    "middle is quiet.",
    "",
    "Fast, jagged wiggles usually mean high-pitched or noisy",
    "sound. Slow, smooth curves usually mean bass or low",
    "tones. A dense, busy waveform generally means a louder,",
    "more compressed mix.",
]

HELP_FOOTER = "Click anywhere to close"

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

# solid-color program: used for the help icon circle and the overlay panel
SOLID_VERTEX_SHADER = """
#version 330
in vec2 pos;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

SOLID_FRAGMENT_SHADER = """
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
            self.width, self.height, "Oscilloscope", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync: cap render rate to monitor refresh (144Hz)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
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

        # solid-quad VAO/VBO -- shared by the help-icon circle outline
        # and the overlay's background quads
        self.solid_program = link_program(SOLID_VERTEX_SHADER, SOLID_FRAGMENT_SHADER)
        self.solid_color_loc = glGetUniformLocation(self.solid_program, "color")
        self.solid_alpha_loc = glGetUniformLocation(self.solid_program, "alpha")

        self.solid_vao = glGenVertexArrays(1)
        glBindVertexArray(self.solid_vao)
        self.solid_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.solid_vbo)
        glBufferData(GL_ARRAY_BUFFER, 32 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.text_program = text_link_program(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)
        self.text = TextRenderer(self.text_program)

        self.history = np.zeros(HISTORY_SAMPLES, dtype=np.float32)

        self.audio = AudioCapture(chunk_size=256)

        # help icon, positioned in window-space NDC like stereometer.py's
        self.help_icon_cx = 0.93
        self.help_icon_cy = 0.85
        self.help_icon_r = 0.06
        self.help_open = False
        self.mouse_x, self.mouse_y = -1.0, -1.0

    def _on_resize(self, window, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    def _on_cursor_move(self, window, xpos, ypos):
        self.mouse_x, self.mouse_y = xpos, ypos

    def _mouse_to_window_ndc(self, xpos, ypos):
        if self.width <= 0 or self.height <= 0:
            return 0.0, 0.0
        x = (xpos / self.width) * 2.0 - 1.0
        y = 1.0 - (ypos / self.height) * 2.0
        return x, y

    def _on_mouse_button(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        if self.help_open:
            self.help_open = False
            return
        x, y = self._mouse_to_window_ndc(self.mouse_x, self.mouse_y)
        dx = x - self.help_icon_cx
        dy = y - self.help_icon_cy
        if dx * dx + dy * dy <= self.help_icon_r * self.help_icon_r:
            self.help_open = True

    def _update_audio(self):
        chunk = self.audio.read_chunk()  # shape (N, channels)
        if chunk is None:
            return
        mono = chunk.mean(axis=1).astype(np.float32)  # L+R -> mono
        n = len(mono)
        if n >= HISTORY_SAMPLES:
            self.history[:] = mono[-HISTORY_SAMPLES:]
        else:
            self.history = np.roll(self.history, -n)
            self.history[-n:] = mono

    def _quad_verts(self, x0, x1, y0, y1) -> np.ndarray:
        return np.array(
            [x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1], dtype=np.float32
        )

    def _draw_quad_window(self, x0, y0, x1, y1, color, alpha=1.0):
        glUniform3f(self.solid_color_loc, *color)
        glUniform1f(self.solid_alpha_loc, alpha)
        verts = self._quad_verts(x0, x1, y0, y1)
        glBindBuffer(GL_ARRAY_BUFFER, self.solid_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def _draw_circle_window(self, cx, cy, r, color, alpha=1.0, segments=24):
        """
        Circle in window-NDC that stays round regardless of window aspect
        ratio -- pre-corrects the unit-circle vertices by the window's
        width/height ratio before the fixed (x, y) uniform-free vertex
        shader just passes them straight through.
        """
        if self.width <= 0 or self.height <= 0:
            return
        aspect = self.height / self.width
        glUniform3f(self.solid_color_loc, *color)
        glUniform1f(self.solid_alpha_loc, alpha)
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=True, dtype=np.float32)
        verts = np.stack(
            [cx + np.cos(angles) * r * aspect, cy + np.sin(angles) * r],
            axis=1,
        ).astype(np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.solid_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_LINE_STRIP, 0, segments)

    def _text_window(self, s, x_ndc, y_ndc, scale, color, align="left", valign="top"):
        self.text.draw(s, x_ndc, y_ndc, pixel_scale=scale,
                        win_w=self.width, win_h=self.height, color=color,
                        align=align, valign=valign)

    def _draw_help_icon(self):
        hovering = False
        x, y = self._mouse_to_window_ndc(self.mouse_x, self.mouse_y)
        dx, dy = x - self.help_icon_cx, y - self.help_icon_cy
        if dx * dx + dy * dy <= self.help_icon_r * self.help_icon_r:
            hovering = True

        glUseProgram(self.solid_program)
        glBindVertexArray(self.solid_vao)
        icon_color = (0.75, 0.75, 0.8) if hovering else (0.5, 0.5, 0.55)
        self._draw_circle_window(self.help_icon_cx, self.help_icon_cy, self.help_icon_r, icon_color, alpha=0.9)
        glBindVertexArray(0)

        self._text_window("?", self.help_icon_cx, self.help_icon_cy, 1.1,
                           icon_color, align="center", valign="middle")

    def _draw_help_overlay(self):
        glUseProgram(self.solid_program)
        glBindVertexArray(self.solid_vao)
        self._draw_quad_window(-1.0, -1.0, 1.0, 1.0, (0.0, 0.0, 0.0), alpha=0.72)
        self._draw_quad_window(-0.95, -0.85, 0.95, 0.85, (0.08, 0.08, 0.1), alpha=0.97)
        glBindVertexArray(0)

        # title centered across the full width, body text centered as a
        # block below it -- rather than everything left-anchored at a
        # fixed x, which on a wide/short window like this one leaves the
        # whole right half of the overlay empty
        #
        # Panel interior spans roughly y in [-0.85, 0.85] (1.7 NDC tall).
        # line_h and title_y are chosen so title + all body lines + the
        # footer are guaranteed to fit inside that span at any window
        # size, instead of being tuned by eye for one particular window
        # height and then overflowing at another (line_h=0.135 with 11
        # lines needed ~1.5 NDC just for the body, which pushed the last
        # couple of lines past the bottom edge of the panel).
        title_y = 0.72
        self._text_window(HELP_TITLE, 0.0, title_y, 1.4, (0.92, 0.92, 0.97), align="center")

        line_h = 0.082
        start_y = title_y - 0.18
        for i, line in enumerate(HELP_TEXT_LINES):
            if line:
                self._text_window(line, 0.0, start_y - i * line_h, 0.72, (0.7, 0.7, 0.75), align="center")

        footer_y = start_y - len(HELP_TEXT_LINES) * line_h - 0.05
        self._text_window(HELP_FOOTER, 0.0, footer_y, 0.7, (0.55, 0.55, 0.6), align="center")

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

        self._draw_help_icon()
        if self.help_open:
            self._draw_help_overlay()

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