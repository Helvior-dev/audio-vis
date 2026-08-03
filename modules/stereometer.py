"""
Vectorscope module (renamed from Stereometer -- "vectorscope" is the
standard term for this display in audio engineering, matches the
reference app's naming).

Core visualization (the Lissajous scatter plot) is unchanged from the
original stereometer.py -- same Mid/Side transform, same point cloud.
This pass builds out the surrounding chrome to match the reference
layout (title, diagonal L/MID/R/SIDE guide lines, correlation readout),
plus: aspect-correct rendering so the diamond stays square when the
window is maximized/fullscreen (GLFW's aspect-ratio hint only affects
interactive resize, not maximize), a blanked title-bar icon, and a
click-to-open help overlay explaining what the display and CORR value
mean.

Two coordinate spaces are used:
  - "square" space (aspect_scale applied): the diamond, its axis labels,
    and the CORR bar all live here, centered in the largest square that
    fits the window -- this is what keeps the diamond a true diamond
    instead of an ellipse when the window isn't square.
  - "window" space (aspect_scale = 1,1): UI chrome that should use the
    window's actual edges -- the title and the help icon/overlay --
    lives here. On a wide monitor the square is much narrower than the
    window, so chrome anchored to the square's edges would bunch up
    near the center instead of sitting at the window's real corners.
"""

import sys
import ctypes
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from audio_capture import AudioCapture
from window_utils import apply_dark_titlebar, remove_titlebar_icon
from text_render import TextRenderer, TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER, link_program as text_link_program

HISTORY_POINTS = 1024  # how many recent samples are plotted as dots

# how fast the displayed correlation eases toward the measured value
# (per second, exponential) -- without this the readout and the bar
# fill jitter frame-to-frame since corrcoef() is recomputed on every
# small audio chunk
CORR_SMOOTHING_PER_SEC = 6.0

HELP_TEXT_LINES = [
    "VECTORSCOPE",
    "",
    "Plots Mid (L+R, mono sum) against Side (L-R, stereo",
    "difference) for the current audio. A narrow vertical",
    "shape means mostly mono content; a wide round shape",
    "means a wide stereo image.",
    "",
    "CORR -- L/R CORRELATION",
    "",
    "+1.00  identical channels (mono)",
    " 0.00  fully independent channels (wide stereo)",
    "-1.00  out of phase (L = -R); sums to silence",
    "       in mono, usually a mixing problem",
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


# Vertex shader applies an aspect-correction scale on top of the
# existing offset/scale uniforms. GLFW's set_window_aspect_ratio() hint
# only constrains interactive corner-dragging -- maximizing the window
# (fullscreen / Win+Up) bypasses it entirely, so the framebuffer can end
# up non-square. aspect_scale = (1,1) draws in raw window NDC; any other
# value draws in the centered-square space the diamond uses.
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

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
        # kept as a hint for interactive corner-dragging; maximize/fullscreen
        # bypasses this, which is why aspect_scale exists in the shader too
        glfw.set_window_aspect_ratio(self.window, 1, 1)
        apply_dark_titlebar(self.window)
        remove_titlebar_icon(self.window)

        self.program = link_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.color_loc = glGetUniformLocation(self.program, "color")
        self.alpha_loc = glGetUniformLocation(self.program, "alpha")
        self.offset_loc = glGetUniformLocation(self.program, "offset")
        self.scale_loc = glGetUniformLocation(self.program, "scale")
        self.aspect_loc = glGetUniformLocation(self.program, "aspect_scale")

        # text rendering (title, axis labels, corr readout, help overlay)
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

        # generic dynamic-quad/line VAO/VBO reused for: correlation bar,
        # diagonal guide lines, background track, help icon, help panel.
        # Sized for the largest thing ever written here (the help-icon
        # circle, 24 segments = 192 bytes) -- quads/lines only use the
        # first 6 or 2 vertices of this same buffer.
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

        self.audio = AudioCapture(chunk_size=256)
        self.points = np.zeros((HISTORY_POINTS, 2), dtype=np.float32)
        self.correlation = 0.0          # raw measured value, updated every chunk
        self.displayed_correlation = 0.0  # eased value actually drawn
        self.last_time = glfw.get_time()

        # help icon position/size, in WINDOW NDC (real screen edges, not
        # the diamond's square) -- top-right corner
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
        """
        Converts a raw framebuffer mouse position into the aspect-
        corrected square NDC space the diamond diagram uses -- i.e. the
        space where (-1,-1) to (1,1) is the largest centered square in
        the window regardless of the window's actual shape.
        """
        if self.width <= 0 or self.height <= 0:
            return 0.0, 0.0
        raw_x = (xpos / self.width) * 2.0 - 1.0
        raw_y = 1.0 - (ypos / self.height) * 2.0
        min_dim = min(self.width, self.height)
        ax = min_dim / self.width
        ay = min_dim / self.height
        # inverse of the shader's forward transform (p = pos * aspect_scale)
        return raw_x / ax, raw_y / ay

    def _mouse_to_window_ndc(self, xpos, ypos):
        """Raw window NDC (no square correction) -- for hit-testing UI
        chrome that's drawn with _draw_*_window() / _text_window()."""
        if self.width <= 0 or self.height <= 0:
            return 0.0, 0.0
        x = (xpos / self.width) * 2.0 - 1.0
        y = 1.0 - (ypos / self.height) * 2.0
        return x, y

    def _on_mouse_button(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        if self.help_open:
            # any click closes the overlay
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

        # exponential smoothing toward the raw measured value, same
        # ballistics style as vu.py / loudness.py -- keeps the bar and
        # numeric readout from jittering every audio chunk
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
        """
        Same as _draw_quad but in real window NDC (aspect_scale = 1,1,
        no squaring). Used for UI chrome -- title, CORR bar, help
        icon/overlay -- that should use the window's full width instead
        of being confined to the centered square the diamond lives in.
        On a wide monitor the square is much narrower than the window,
        so UI anchored to its edges would bunch up near the center
        instead of sitting at the window's actual corners/edges.
        """
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
            of window aspect ratio. _draw_circle_window previously used
            aspect_scale=(1,1) to keep the *position* in real window NDC,
            but that also left the *shape* uncorrected -- on a non-square
            window, NDC units are physically different sizes on x vs y, so
            an uncorrected circle (equal radius in raw NDC) renders as an
            ellipse, same root cause as the diamond-turning-into-an-ellipse
            bug this whole aspect_scale mechanism was built to fix.

            The shader applies aspect_scale to both offset and the vertex
            shape (p = pos*scale*aspect_scale + offset*aspect_scale), so
            using the real aspect_scale here would correctly round out the
            shape but also pull the icon's position toward the window
            center on a wide monitor -- reintroducing the very "icon drifts
            out of the corner" bug from the previous pass. Pre-dividing cx/cy
            by aspect_scale here cancels that: after the shader multiplies by
            aspect_scale again, offset lands back at exactly (cx, cy) in
            window NDC, while the vertex shape still gets the correction it
            needs to stay circular.
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

    def _text_window(self, s, x_ndc, y_ndc, scale, color, align="left"):
        """
        Window-NDC counterpart of _text_square -- draws directly in the
        window's own NDC space using the real win_w/win_h, instead of
        the centered-square space the diamond diagram uses. Font size
        still scales with window height via TextRenderer.draw()'s own
        win_h-based sizing, so text stays proportional to the window,
        it just isn't confined to the square's footprint.
        """
        self.text.draw(s, x_ndc, y_ndc, pixel_scale=scale,
                        win_w=self.width, win_h=self.height, color=color, align=align)

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
            """
            Draws text positioned in the aspect-corrected square NDC space
            the diamond diagram uses, but SIZED in real window NDC.

            Earlier versions passed win_w=win_h=min(width,height) to size
            the glyph quad, reasoning that "square" units would keep glyph
            proportions correct. That was wrong: TEXT_VERTEX_SHADER writes
            gl_Position directly with no aspect correction of its own, so
            whatever ndc_w/ndc_h TextRenderer.draw() computes gets mapped
            straight onto the real (possibly non-square) framebuffer. Sizing
            against min_dim/min_dim produced a glyph quad whose width and
            height were equal fractions of a *hypothetical* square window --
            but stretched onto the *actual* non-square framebuffer, that
            equal-fraction quad renders at different physical width/height
            ratios than the source bitmap, which is exactly the horizontal
            stretching seen on wide monitors. Passing the real width/height
            here sizes the quad correctly for the framebuffer it's actually
            drawn on; only the *position* (x_win/y_win below) needs the
            square-space correction, so the text's anchor point still lines
            up with the aspect-corrected diamond geometry.
            """
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

        self._text_window("?", self.help_icon_cx, self.help_icon_cy + 0.018, 1.1,
                           icon_color, align="center")

    def _draw_help_overlay(self):
        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)
        # dim the whole scene, then a near-fullscreen panel on top --
        # all in window space, so it always covers the actual window
        # and the text uses the window's real width rather than being
        # squeezed into the (possibly much narrower) diamond's square.
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
            # window is minimized -- GLFW reports a 0x0 framebuffer in
            # this state. Nothing is visible anyway, and every NDC/pixel
            # calculation in this file (and in TextRenderer.draw(),
            # which divides by win_w/win_h) assumes a positive size, so
            # skip the frame entirely rather than let one of those
            # divisions raise ZeroDivisionError and crash the process.
            return

        glClearColor(0.03, 0.03, 0.04, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)

        # ---- vectorscope field layout ----
        # the scatter field is a square centered in the upper ~75% of the
        # window; correlation bar sits in a separate strip at the bottom,
        # matching the reference screenshot's proportions. All of this is
        # in "square NDC" -- see aspect_scale in the shader -- so it stays
        # a true diamond even when the window itself isn't square.
        field_cy = 0.18     # NDC y-center of the diamond field
        field_scale = 0.62  # NDC half-size of the diamond

        # diagonal guide lines forming the diamond (L-MID-R-SIDE axes), dim
        diag_color = (0.28, 0.30, 0.34)
        self._draw_line(-field_scale, field_cy, 0.0, field_cy + field_scale, diag_color, alpha=0.8)
        self._draw_line(0.0, field_cy + field_scale, field_scale, field_cy, diag_color, alpha=0.8)
        self._draw_line(field_scale, field_cy, 0.0, field_cy - field_scale, diag_color, alpha=0.8)
        self._draw_line(0.0, field_cy - field_scale, -field_scale, field_cy, diag_color, alpha=0.8)

        # ---- scatter points (untouched core visualization) ----
        ax, ay = self._aspect_scale()
        glUniform2f(self.offset_loc, 0.0, field_cy)
        glUniform1f(self.scale_loc, field_scale * 0.9)
        glUniform2f(self.aspect_loc, ax, ay)
        glUniform3f(self.color_loc, 0.6, 0.85, 0.95)  # original stereometer.py color, unchanged
        glUniform1f(self.alpha_loc, 0.8)

        glBindVertexArray(self.points_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.points_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.points.nbytes, self.points)
        glDrawArrays(GL_POINTS, 0, HISTORY_POINTS)
        glBindVertexArray(self.line_vao)  # switch back for the rest of this frame

        # ---- axis labels: L / MID / R / SIDE at the four diamond tips ----
        label_color = (0.55, 0.55, 0.6)
        label_offset = field_scale + 0.09
        self._text_square("L", -label_offset, field_cy + 0.03, 1.6, label_color, align="center")
        self._text_square("MID", 0.0, field_cy + field_scale + 0.10, 1.4, label_color, align="center")
        self._text_square("R", label_offset, field_cy + 0.03, 1.6, label_color, align="center")
        self._text_square("SIDE", 0.0, field_cy - field_scale - 0.02, 1.4, label_color, align="center")

        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)

        # ---- title, top-left -- window space, not square space, so it
        # sits at the window's actual corner on wide monitors instead of
        # bunching toward the center of a narrower diamond square ----
        self._text_window("STEREOMETER", -0.94, 0.92, 1.15, (0.85, 0.85, 0.9))

        # text.draw() switches to text_program internally -- must restore
        # self.program before issuing more glUniform*/glDrawArrays calls
        # that target it, or GL rejects them (invalid operation: wrong
        # program bound for that uniform location)
        glUseProgram(self.program)
        glBindVertexArray(self.line_vao)

        # ---- correlation bar + numeric readout, bottom ----
        # both driven by displayed_correlation (eased), not the raw
        # per-chunk measurement, so the fill and text move smoothly.
        # Kept in square space -- it reads as part of the diamond's own
        # composition, so it should scale/center with the diamond
        # rather than stretch across the full window width.
        bar_y0, bar_y1 = -0.74, -0.68
        bar_x0, bar_x1 = -0.7, 0.7
        self._draw_quad(bar_x0, bar_y0, bar_x1, bar_y1, (0.14, 0.14, 0.17))  # track

        x_zero = 0.0
        x_val = (self.displayed_correlation * 0.9) * ((bar_x1 - bar_x0) / 2)
        fill_color = (0.35, 0.75, 0.95) if self.displayed_correlation >= -0.3 else (0.9, 0.25, 0.2)
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