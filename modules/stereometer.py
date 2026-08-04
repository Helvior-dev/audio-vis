"""
Vectorscope module (renamed from Stereometer -- "vectorscope" is the
standard term for this display in audio engineering, matches the
reference app's naming).

Core visualization (the Lissajous scatter plot) is unchanged from the
original stereometer.py -- same Mid/Side transform, same point cloud.
This pass builds out the surrounding chrome to match the reference
layout (title, diagonal L/MID/R/SIDE guide lines, correlation readout),
plus: aspect-correct rendering so the diamond stays square when the
window is maximized/fullscreen, a blanked title-bar icon, and a
click-to-open help overlay explaining what the display and CORR value
mean.

Two coordinate spaces are used:
  - "square" space (aspect_scale applied): the diamond, its axis labels,
    and the CORR bar all live here, centered in the largest square that
    fits the window -- this is what keeps the diamond a true diamond
    instead of an ellipse when the window isn't square.
  - "window" space (aspect_scale = 1,1 for position, but shape-only
    correction where needed -- see _draw_circle_window): UI chrome that
    should use the window's actual edges -- the title and the help
    icon/overlay -- lives here.
"""

import sys
import ctypes
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import *

sys.path.insert(0, str(Path(__file__).parent))
from audio_capture import AudioCapture
from window_utils import apply_dark_titlebar, remove_titlebar_icon
from text_render import TextRenderer, TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER, link_program as text_link_program
from audio_source_config import load_selected_source, SourceWatcher

HISTORY_POINTS = 1024  # how many recent samples are plotted as dots

# how fast the displayed correlation eases toward the measured value
# (per second, exponential) -- without this the readout and the bar
# fill jitter frame-to-frame since corrcoef() is recomputed on every
# small audio chunk
CORR_SMOOTHING_PER_SEC = 6.0

HELP_TEXT_LINES = [
    "STEREOMETER",
    "",
    "Shows the shape of your stereo sound as a cloud of dots.",
    "Each dot is one moment of audio, plotted by how loud it is",
    "(up/down) and how different the left and right speakers",
    "sound (left/right). A tall thin shape means the sound is",
    "close to mono (both speakers playing almost the same thing).",
    "A wide round shape means a wide, spacious stereo sound.",
    "",
    "CORR -- HOW SIMILAR LEFT AND RIGHT ARE",
    "",
    "+1.00  left and right are identical (mono)",
    " 0.00  left and right are unrelated (wide stereo)",
    "-1.00  left and right cancel out; playing this in mono",
    "       would go silent -- usually a mixing mistake",
    "",
    "Click anywhere to close",
]


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


VERTEX_SHADER = """
#version 330
in vec2 pos;
uniform vec2 offset;
uniform float scale;
uniform vec2 aspect_scale;
void main() {
    vec2 p = pos * scale * aspect_scale + offset * aspect_scale;
    gl_Position = vec4(p, 0.0, 1.0);
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
        
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        glfw.set_window_pos(self.window, (mode.size.width - self.width) // 2, (mode.size.height - self.height) // 2)


        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
        glfw.set_window_aspect_ratio(self.window, 1, 1)
        apply_dark_titlebar(self.window)
        remove_titlebar_icon(self.window)

        self.program = link_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.color_loc = glGetUniformLocation(self.program, "color")
        self.alpha_loc = glGetUniformLocation(self.program, "alpha")
        self.offset_loc = glGetUniformLocation(self.program, "offset")
        self.scale_loc = glGetUniformLocation(self.program, "scale")
        self.aspect_loc = glGetUniformLocation(self.program, "aspect_scale")

        self.text_program = text_link_program(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)
        self.text = TextRenderer(self.text_program)

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

        self.line_vao = glGenVertexArrays(1)
        glBindVertexArray(self.line_vao)
        self.line_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferData(GL_ARRAY_BUFFER, 32 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.audio = AudioCapture(chunk_size=256, source=load_selected_source())
        self.source_watcher = SourceWatcher()
        self.points = np.zeros((HISTORY_POINTS, 2), dtype=np.float32)
        self.correlation = 0.0
        self.displayed_correlation = 0.0
        self.last_time = glfw.get_time()

        self.help_icon_cx = 0.93
        self.help_icon_cy = 0.95
        self.help_icon_r = 0.035
        self.help_open = False
        self.mouse_x, self.mouse_y = -1.0, -1.0

    def _on_resize(self, window, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    def _on_cursor_move(self, window, xpos, ypos):
        self.mouse_x, self.mouse_y = xpos, ypos

    def _mouse_to_square_ndc(self, xpos, ypos):
        if self.width <= 0 or self.height <= 0:
            return 0.0, 0.0
        raw_x = (xpos / self.width) * 2.0 - 1.0
        raw_y = 1.0 - (ypos / self.height) * 2.0
        min_dim = min(self.width, self.height)
        ax = min_dim / self.width
        ay = min_dim / self.height
        return raw_x / ax, raw_y / ay

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

    def _aspect_scale(self):
        min_dim = min(self.width, self.height)
        if self.width <= 0 or self.height <= 0:
            return 1.0, 1.0
        return min_dim / self.width, min_dim / self.height

    def _update_audio(self):
        now = glfw.get_time()
        dt = max(1e-4, now - self.last_time)
        self.last_time = now

        new_source = self.source_watcher.check()
        if new_source is not None:
            self.audio.reopen(new_source)
            self.points[:] = 0.0
            self.correlation = 0.0
            self.displayed_correlation = 0.0

        chunk = self.audio.read_chunk()  # (N, channels) or None
        if chunk is None:
            # nothing new since the last frame (render loop polled
            # faster than the next audio callback) -- not silence, just
            # no update this frame; smoothing (correlation easing below)
            # still needs dt tracked via last_time above even when we
            # skip the rest of this update
            return

        # a microphone source may report a single channel -- the
        # vectorscope's whole point is the L/R relationship, so with
        # only one channel there's no "side" signal to show. Duplicate
        # the mono channel into both L and R rather than crashing on
        # chunk[:, 1]: mid ends up equal to the signal and side is
        # exactly zero, which draws as an honest flat vertical line
        # (pure mono) instead of an error.
        if chunk.shape[1] >= 2:
            left = chunk[:, 0]
            right = chunk[:, 1]
        else:
            left = chunk[:, 0]
            right = chunk[:, 0]

        mid = (left + right) * 0.5
        side = (left - right) * 0.5

        n = len(mid)
        new_points = np.stack([side, mid], axis=1).astype(np.float32)

        if n >= HISTORY_POINTS:
            self.points = new_points[-HISTORY_POINTS:]
        else:
            self.points = np.roll(self.points, -n, axis=0)
            self.points[-n:] = new_points

        # np.std(...) == 0 covers both real silence (a genuine zero chunk
        # from read_chunk()'s sustained-underrun path) and the edge case
        # of a dead-flat non-zero signal -- either way there's no
        # meaningful left/right relationship to report, so correlation
        # is defined as 0 (neutral) rather than left to hold its last
        # value indefinitely, which is what made CORR "stick" during
        # playback pauses instead of returning to 0.
        if np.std(left) > 1e-6 and np.std(right) > 1e-6:
            corr = np.corrcoef(left, right)[0, 1]
            self.correlation = float(np.clip(corr, -1.0, 1.0))
        else:
            self.correlation = 0.0

        max_step = CORR_SMOOTHING_PER_SEC * dt
        diff = self.correlation - self.displayed_correlation
        if abs(diff) <= max_step:
            self.displayed_correlation = self.correlation
        else:
            self.displayed_correlation += max_step if diff > 0 else -max_step

    def _quad_verts(self, x0, x1, y0, y1) -> np.ndarray:
        return np.array(
            [x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1], dtype=np.float32
        )

    def _draw_line(self, x0, y0, x1, y1, color, alpha=1.0, width=1.0):
        ax, ay = self._aspect_scale()
        glUniform2f(self.offset_loc, 0.0, 0.0)
        glUniform1f(self.scale_loc, 1.0)
        glUniform2f(self.aspect_loc, ax, ay)
        glUniform3f(self.color_loc, *color)
        glUniform1f(self.alpha_loc, alpha)
        glLineWidth(width)
        verts = np.array([x0, y0, x1, y1], dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_LINES, 0, 2)

    def _draw_quad_window(self, x0, y0, x1, y1, color, alpha=1.0):
        glUniform2f(self.offset_loc, 0.0, 0.0)
        glUniform1f(self.scale_loc, 1.0)
        glUniform2f(self.aspect_loc, 1.0, 1.0)
        glUniform3f(self.color_loc, *color)
        glUniform1f(self.alpha_loc, alpha)
        verts = self._quad_verts(x0, x1, y0, y1)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def _draw_circle_window(self, cx, cy, r, color, alpha=1.0, segments=24):
        """
        Window-NDC positioned circle that stays a true circle regardless
        of window aspect ratio. Pre-divides cx/cy by aspect_scale so
        that after the shader multiplies by aspect_scale again, offset
        lands back at exactly (cx, cy) in window NDC, while the vertex
        shape still gets the correction it needs to stay circular.
        """
        ax, ay = self._aspect_scale()
        glUniform2f(self.offset_loc, cx / ax, cy / ay)
        glUniform1f(self.scale_loc, r)
        glUniform2f(self.aspect_loc, ax, ay)
        glUniform3f(self.color_loc, *color)
        glUniform1f(self.alpha_loc, alpha)
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=True, dtype=np.float32)
        verts = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_LINE_STRIP, 0, segments)

    def _text_window(self, s, x_ndc, y_ndc, scale, color, align="left", valign="top"):
        self.text.draw(s, x_ndc, y_ndc, pixel_scale=scale,
                        win_w=self.width, win_h=self.height, color=color,
                        align=align, valign=valign)

    def _draw_quad(self, x0, y0, x1, y1, color, alpha=1.0):
        ax, ay = self._aspect_scale()
        glUniform2f(self.offset_loc, 0.0, 0.0)
        glUniform1f(self.scale_loc, 1.0)
        glUniform2f(self.aspect_loc, ax, ay)
        glUniform3f(self.color_loc, *color)
        glUniform1f(self.alpha_loc, alpha)
        verts = self._quad_verts(x0, x1, y0, y1)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def _draw_circle(self, cx, cy, r, color, alpha=1.0, segments=24):
        ax, ay = self._aspect_scale()
        glUniform2f(self.offset_loc, cx, cy)
        glUniform1f(self.scale_loc, r)
        glUniform2f(self.aspect_loc, ax, ay)
        glUniform3f(self.color_loc, *color)
        glUniform1f(self.alpha_loc, alpha)
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=True, dtype=np.float32)
        verts = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_LINE_STRIP, 0, segments)

    def _text_square(self, s, x_ndc, y_ndc, scale, color, align="left"):
        ax, ay = self._aspect_scale()
        x_win = x_ndc * ax
        y_win = y_ndc * ay
        self.text.draw(s, x_win, y_win, pixel_scale=scale,
                        win_w=self.width, win_h=self.height, color=color, align=align)

    def _draw_help_icon(self):
        hovering = False
        x, y = self._mouse_to_window_ndc(self.mouse_x, self.mouse_y)
        dx, dy = x - self.help_icon_cx, y - self.help_icon_cy
        if dx * dx + dy * dy <= self.help_icon_r * self.help_icon_r:
            hovering = True

        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)
        icon_color = (0.75, 0.75, 0.8) if hovering else (0.5, 0.5, 0.55)
        self._draw_circle_window(self.help_icon_cx, self.help_icon_cy, self.help_icon_r, icon_color, alpha=0.9)

        self._text_window("?", self.help_icon_cx, self.help_icon_cy, 1.1,
                           icon_color, align="center", valign="middle")

    def _draw_help_overlay(self):
        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)
        self._draw_quad_window(-1.0, -1.0, 1.0, 1.0, (0.0, 0.0, 0.0), alpha=0.72)
        self._draw_quad_window(-0.95, -0.85, 0.95, 0.85, (0.08, 0.08, 0.1), alpha=0.97)

        line_h = 0.075
        start_y = 0.68
        for i, line in enumerate(HELP_TEXT_LINES):
            color = (0.9, 0.9, 0.95) if (i == 0 or line.startswith("CORR")) else (0.68, 0.68, 0.73)
            scale = 1.15 if (i == 0 or line.startswith("CORR")) else 0.85
            if line:
                self._text_window(line, -0.88, start_y - i * line_h, scale, color, align="left")

    def render_frame(self):
        if self.width <= 0 or self.height <= 0:
            return

        glClearColor(0.03, 0.03, 0.04, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)

        field_cy = 0.18
        field_scale = 0.62

        diag_color = (0.28, 0.30, 0.34)
        self._draw_line(-field_scale, field_cy, 0.0, field_cy + field_scale, diag_color, alpha=0.8)
        self._draw_line(0.0, field_cy + field_scale, field_scale, field_cy, diag_color, alpha=0.8)
        self._draw_line(field_scale, field_cy, 0.0, field_cy - field_scale, diag_color, alpha=0.8)
        self._draw_line(0.0, field_cy - field_scale, -field_scale, field_cy, diag_color, alpha=0.8)

        ax, ay = self._aspect_scale()
        glUniform2f(self.offset_loc, 0.0, field_cy)
        glUniform1f(self.scale_loc, field_scale * 0.9)
        glUniform2f(self.aspect_loc, ax, ay)
        glUniform3f(self.color_loc, 0.6, 0.85, 0.95)
        glUniform1f(self.alpha_loc, 0.8)

        glBindVertexArray(self.points_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.points_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.points.nbytes, self.points)
        glDrawArrays(GL_POINTS, 0, HISTORY_POINTS)
        glBindVertexArray(self.line_vao)

        label_color = (0.55, 0.55, 0.6)
        label_offset = field_scale + 0.09
        self._text_square("L", -label_offset, field_cy + 0.03, 1.6, label_color, align="center")
        self._text_square("MID", 0.0, field_cy + field_scale + 0.10, 1.4, label_color, align="center")
        self._text_square("R", label_offset, field_cy + 0.03, 1.6, label_color, align="center")
        self._text_square("SIDE", 0.0, field_cy - field_scale - 0.02, 1.4, label_color, align="center")

        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)

        self._text_window("STEREOMETER", -0.94, 0.92, 1.15, (0.85, 0.85, 0.9))

        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)

        bar_y0, bar_y1 = -0.74, -0.68
        bar_x0, bar_x1 = -0.7, 0.7
        self._draw_quad(bar_x0, bar_y0, bar_x1, bar_y1, (0.14, 0.14, 0.17))

        x_zero = 0.0
        x_val = (self.displayed_correlation * 0.9) * ((bar_x1 - bar_x0) / 2)
        fill_color = (0.35, 0.75, 0.95) if self.displayed_correlation >= 0.0 else (0.9, 0.25, 0.2)
        if self.displayed_correlation >= 0:
            self._draw_quad(x_zero, bar_y0, x_val, bar_y1, fill_color)
        else:
            self._draw_quad(x_val, bar_y0, x_zero, bar_y1, fill_color)

        self._text_square(
            f"CORR {self.displayed_correlation:+.2f}", 0.0, bar_y0 - 0.05, 1.4,
            (0.8, 0.8, 0.85), align="center",
        )

        self._draw_help_icon()
        if self.help_open:
            self._draw_help_overlay()

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