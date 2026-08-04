"""
Spectrum module: frequency-domain bar display via FFT.

Pipeline: audio chunk -> Hann window -> FFT -> magnitude -> log-scale
frequency bins -> dB -> smoothed over time -> drawn as vertical bars.

Log-scale binning matters here: linear FFT bins waste most of the
display on high frequencies (which carry little musical information)
and cram all the bass into a few pixels. Grouping bins logarithmically
(similar to how the ear perceives pitch) spreads the spectrum out the
way MiniMeters' Mel/Log scale does.
"""

import sys
import ctypes
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import *

sys.path.insert(0, str(Path(__file__).parent))
from audio_capture import AudioCapture
from window_utils import apply_dark_titlebar, set_window_icon
from text_render import TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER, TextRenderer
from audio_source_config import load_selected_source, SourceWatcher

ICON_PATH = Path(__file__).parent / "assets" / "icon.png"

FFT_SIZE = 2048
NUM_BARS = 64
DB_FLOOR = -70.0   # anything quieter than this renders as an empty bar
DB_CEIL = 0.0
SMOOTHING = 0.75    # 0 = no smoothing (jumpy), closer to 1 = smoother but laggier

HELP_TITLE = "SPECTRUM"

HELP_TEXT_LINES = [
    "Splits the sound into its individual frequencies -- bass on",
    "the left, treble on the right -- and shows how loud each one",
    "is right now, on a logarithmic scale that mirrors how pitch",
    "actually sounds to the ear.",
    "",
    "Taller bars mean more energy at that frequency. A bar turns",
    "red only when that frequency is close to clipping.",
]

HELP_FOOTER = "Click anywhere to close"

VERTEX_SHADER = """
#version 330
in vec2 pos;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
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

# solid-color program with alpha: used for the help icon circle and the overlay panel
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
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


def make_log_bin_edges(num_bars: int, fft_size: int, min_bin: int = 1) -> np.ndarray:
    """
    Returns num_bars+1 bin-index edges spaced logarithmically across the
    usable FFT range (skips bin 0 = DC, and the top half which is above
    Nyquist mirror / not perceptually useful at typical sample rates).
    """
    max_bin = fft_size // 2
    edges = np.logspace(np.log10(min_bin), np.log10(max_bin), num_bars + 1)
    return np.unique(edges.astype(int))


class SpectrumWindow:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.width, self.height = 800, 300
        # start hidden: create_window() shows the HWND immediately, but
        # nothing is drawn into it until the first swap_buffers() call
        # in run() -- without this hint that gap (shader compiles,
        # AudioCapture opening the WASAPI stream, icon decoding, etc.)
        # is visible as a blank white window. run() shows the window
        # itself right after the first real frame is drawn, so the
        # window only ever appears with content already in it.
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(self.width, self.height, "Spectrum", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        glfw.set_window_pos(self.window, (mode.size.width - self.width) // 2, (mode.size.height - self.height) // 2)

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        apply_dark_titlebar(self.window)
        set_window_icon(self.window, ICON_PATH)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.program = link_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.color_loc = glGetUniformLocation(self.program, "color")

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

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

        self.text_program = link_program(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)
        self.text = TextRenderer(self.text_program)

        self.read_chunk_size = 256
        self.audio = AudioCapture(chunk_size=self.read_chunk_size, source=load_selected_source())
        self.source_watcher = SourceWatcher()
        self.rolling_buffer = np.zeros(FFT_SIZE, dtype=np.float32)
        self.window_fn = np.hanning(FFT_SIZE).astype(np.float32)
        self.bin_edges = make_log_bin_edges(NUM_BARS, FFT_SIZE)
        self.actual_num_bars = len(self.bin_edges) - 1
        self.smoothed_db = np.full(self.actual_num_bars, DB_FLOOR, dtype=np.float32)

        self.mouse_x, self.mouse_y = -1.0, -1.0
        self.help_icon_cx = 0.93
        self.help_icon_cy = 0.85
        self.help_icon_r = 0.045
        self.help_open = False

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
        new_source = self.source_watcher.check()
        if new_source is not None:
            self.audio.reopen(new_source)
            self.rolling_buffer[:] = 0.0

        chunk = self.audio.read_chunk()  # (read_chunk_size, channels)
        if chunk is None:
            return
        mono = chunk.mean(axis=1).astype(np.float32)
        n = len(mono)

        if n > FFT_SIZE:
            mono = mono[-FFT_SIZE:]
            n = FFT_SIZE

        self.rolling_buffer = np.roll(self.rolling_buffer, -n)
        self.rolling_buffer[-n:] = mono

        windowed = self.rolling_buffer * self.window_fn
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum) / (self.window_fn.sum() / 2)

        bar_magnitudes = np.zeros(self.actual_num_bars, dtype=np.float32)
        for i in range(self.actual_num_bars):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            hi = max(hi, lo + 1)
            bar_magnitudes[i] = magnitude[lo:hi].mean()

        db = 20.0 * np.log10(np.maximum(bar_magnitudes, 1e-10))
        db = np.clip(db, DB_FLOOR, DB_CEIL)

        self.smoothed_db = SMOOTHING * self.smoothed_db + (1.0 - SMOOTHING) * db

    def _quad_verts(self, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        return np.array(
            [
                x0, y0,  x1, y0,  x1, y1,
                x0, y0,  x1, y1,  x0, y1,
            ],
            dtype=np.float32,
        )

    def _draw_quad_window(self, x0, y0, x1, y1, color, alpha=1.0):
        glUniform3f(self.solid_color_loc, *color)
        glUniform1f(self.solid_alpha_loc, alpha)
        verts = self._quad_verts(x0, x1, y0, y1)
        glBindBuffer(GL_ARRAY_BUFFER, self.solid_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def _draw_circle_window(self, cx, cy, r, color, alpha=1.0, segments=24):
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

        self.text.draw("?", self.help_icon_cx, self.help_icon_cy, pixel_scale=1.1,
                        win_w=self.width, win_h=self.height, color=icon_color,
                        align="center", valign="middle")

    def _draw_help_overlay(self):
        glUseProgram(self.solid_program)
        glBindVertexArray(self.solid_vao)
        self._draw_quad_window(-1.0, -1.0, 1.0, 1.0, (0.0, 0.0, 0.0), alpha=0.72)
        self._draw_quad_window(-0.95, -0.85, 0.95, 0.85, (0.08, 0.08, 0.1), alpha=0.97)
        glBindVertexArray(0)

        title_y = 0.68
        self.text.draw(HELP_TITLE, 0.0, title_y, pixel_scale=1.4,
                        win_w=self.width, win_h=self.height,
                        color=(0.92, 0.92, 0.97), align="center")

        line_h = 0.09
        start_y = title_y - 0.20
        for i, line in enumerate(HELP_TEXT_LINES):
            if line:
                self.text.draw(line, 0.0, start_y - i * line_h, pixel_scale=0.85,
                                win_w=self.width, win_h=self.height,
                                color=(0.7, 0.7, 0.75), align="center")

        footer_y = start_y - len(HELP_TEXT_LINES) * line_h - 0.06
        self.text.draw(HELP_FOOTER, 0.0, footer_y, pixel_scale=0.75,
                        win_w=self.width, win_h=self.height,
                        color=(0.55, 0.55, 0.6), align="center")

    def render_frame(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        n = self.actual_num_bars
        bar_width = 2.0 / n
        gap = bar_width * 0.15

        for i in range(n):
            x0 = -1.0 + i * bar_width + gap * 0.5
            x1 = -1.0 + (i + 1) * bar_width - gap * 0.5

            level = (self.smoothed_db[i] - DB_FLOOR) / (DB_CEIL - DB_FLOOR)
            level = max(0.0, min(1.0, level))
            y0 = -1.0
            y1 = -1.0 + level * 2.0

            verts = self._quad_verts(x0, x1, y0, y1)
            glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

            if level < 0.93:
                glUniform3f(self.color_loc, 0.55, 0.68, 0.85)
            else:
                glUniform3f(self.color_loc, 0.9, 0.3, 0.25)

            glDrawArrays(GL_TRIANGLES, 0, 6)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self._draw_help_icon()
        if self.help_open:
            self._draw_help_overlay()

    def run(self):
        try:
            # draw + present one real frame before revealing the
            # window (see the VISIBLE hint above) so nothing white
            # or half-initialized is ever shown
            self._update_audio()
            self.render_frame()
            glfw.swap_buffers(self.window)
            glfw.show_window(self.window)

            while not glfw.window_should_close(self.window):
                self._update_audio()
                self.render_frame()
                glfw.swap_buffers(self.window)
                glfw.poll_events()
        finally:
            self.audio.close()
            glfw.terminate()


def main():
    app = SpectrumWindow()
    app.run()


if __name__ == "__main__":
    main()