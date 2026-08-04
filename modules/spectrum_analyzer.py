"""
Spectrum Analyzer module (formerly "Oscilloscope" -- replaced entirely).

Dense log-spaced FFT bar spectrum matching the reference look: many thin
vertical bars colored by amplitude (dark purple -> magenta -> orange),
a blue peak-hold envelope line drawn on top, frequency axis labels
(100Hz / 1kHz / 10kHz) across the top, vertical gridlines at those same
frequencies, and a hover tooltip showing dB / Hz / nearest musical note
under the cursor.
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

FFT_SIZE = 4096
NUM_BARS = 320          # dense, thin bars -- close to the reference density
DB_FLOOR = -70.0
DB_CEIL = 0.0
SMOOTHING = 0.65         # bar smoothing (lower than spectrum.py -- reference looks fairly live)

PEAK_HOLD_SEC = 0.09
PEAK_FALL_DB_PER_SEC = 100.0

MIN_FREQ_HZ = 20.0        # bottom of the displayed/labelled range
MAX_FREQ_HZ = 20000.0     # top of the displayed/labelled range

HELP_TITLE = "SPECTRUM ANALYZER"

HELP_TEXT_LINES = [
    "Splits the sound into its individual frequencies -- from low",
    "bass on the left to high treble on the right -- and shows how",
    "loud each one is right now. Taller bars mean more energy at",
    "that frequency.",
    "",
    "The blue line is a peak hold: it jumps up the instant a",
    "frequency gets loud, hangs there briefly so short hits are",
    "still visible, then eases back down onto the bars.",
    "",
    "Hover over the display to see the exact dB level, frequency,",
    "and nearest musical note under your cursor.",
]

HELP_FOOTER = "Click anywhere to close"

# -------------------------------------------------------------------
# Shaders
# -------------------------------------------------------------------

BAR_VERTEX_SHADER = """
#version 330
layout(location = 0) in vec2 pos;
layout(location = 1) in vec3 color;
out vec3 frag_color;
void main() {
    frag_color = color;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

BAR_FRAGMENT_SHADER = """
#version 330
in vec3 frag_color;
out vec4 out_color;
void main() {
    out_color = vec4(frag_color, 1.0);
}
"""

LINE_VERTEX_SHADER = """
#version 330
layout(location = 0) in vec2 pos;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

LINE_FRAGMENT_SHADER = """
#version 330
uniform vec4 color;
out vec4 out_color;
void main() {
    out_color = color;
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
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


def make_log_bin_edges(num_bars: int, fft_size: int, min_bin: float = 1.0) -> np.ndarray:
    max_bin = fft_size // 2
    return np.logspace(np.log10(min_bin), np.log10(max_bin), num_bars + 1)


def colormap(level: np.ndarray) -> np.ndarray:
    dark_purple = np.array([0.10, 0.02, 0.15])
    magenta = np.array([0.75, 0.15, 0.55])
    orange = np.array([1.00, 0.55, 0.15])

    level = np.clip(level, 0.0, 1.0)
    t_low = np.clip(level * 2.0, 0.0, 1.0)[:, None]
    t_high = np.clip((level - 0.5) * 2.0, 0.0, 1.0)[:, None]

    low_mix = dark_purple[None, :] * (1 - t_low) + magenta[None, :] * t_low
    high_mix = magenta[None, :] * (1 - t_high) + orange[None, :] * t_high

    is_high = (level >= 0.5)[:, None]
    return np.where(is_high, high_mix, low_mix)


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note(freq_hz: float) -> str:
    """Nearest musical note + cents offset, e.g. 'F1 +2 Cents'. A4 = 440Hz."""
    if freq_hz <= 0:
        return "-- "
    semitones_from_a4 = 12.0 * np.log2(freq_hz / 440.0)
    nearest = round(semitones_from_a4)
    cents = int(round((semitones_from_a4 - nearest) * 100))
    midi_note = 69 + nearest
    name = NOTE_NAMES[midi_note % 12]
    octave = midi_note // 12 - 1
    sign = "+" if cents >= 0 else "-"
    return f"{name}{octave} {sign}{abs(cents)}Cents"


class SpectrumAnalyzerWindow:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.width, self.height = 900, 320
        # start hidden: create_window() shows the HWND immediately, but
        # nothing is drawn into it until the first swap_buffers() call
        # in run() -- without this hint that gap (shader compiles,
        # AudioCapture opening the WASAPI stream, icon decoding, etc.)
        # is visible as a blank white window. run() shows the window
        # itself right after the first real frame is drawn, so the
        # window only ever appears with content already in it.
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(self.width, self.height, "Spectrum Analyzer", None, None)
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
        glfw.set_cursor_enter_callback(self.window, self._on_cursor_enter)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        apply_dark_titlebar(self.window)
        set_window_icon(self.window, ICON_PATH)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.bar_program = link_program(BAR_VERTEX_SHADER, BAR_FRAGMENT_SHADER)
        self.bar_vao = glGenVertexArrays(1)
        glBindVertexArray(self.bar_vao)
        self.bar_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.bar_vbo)
        self.max_bar_verts = NUM_BARS * 6
        glBufferData(GL_ARRAY_BUFFER, self.max_bar_verts * 5 * 4, None, GL_DYNAMIC_DRAW)
        stride = 5 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.line_program = link_program(LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER)
        self.line_color_loc = glGetUniformLocation(self.line_program, "color")
        self.line_vao = glGenVertexArrays(1)
        glBindVertexArray(self.line_vao)
        self.line_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferData(GL_ARRAY_BUFFER, max(NUM_BARS, 64) * 2 * 4, None, GL_DYNAMIC_DRAW)
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
        self.actual_num_bars = NUM_BARS
        self.smoothed_db = np.full(self.actual_num_bars, DB_FLOOR, dtype=np.float32)

        self.peak_db = np.full(self.actual_num_bars, DB_FLOOR, dtype=np.float32)
        self._peak_age = np.zeros(self.actual_num_bars, dtype=np.float32)
        self.sample_rate = self.audio.rate
        self.last_time = glfw.get_time()

        self._recompute_bar_freqs()

        self.mouse_x, self.mouse_y = -1.0, -1.0
        self.mouse_in_window = False
        self.help_icon_cx = 0.965
        self.help_icon_cy = 0.90
        self.help_icon_r = 0.045
        self.help_open = False

    def _recompute_bar_freqs(self):
        """Sample-rate-dependent frequency axis -- recomputed after a
        source switch too, since a mic and the system output device can
        run at different native sample rates."""
        bin_hz = self.sample_rate / FFT_SIZE
        bar_center_bin = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        self.bar_freqs_hz = bar_center_bin * bin_hz
        self._fft_bin_axis = np.arange(FFT_SIZE // 2 + 1, dtype=np.float64)

    def _on_resize(self, window, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    def _on_cursor_move(self, window, xpos, ypos):
        self.mouse_x, self.mouse_y = xpos, ypos

    def _on_cursor_enter(self, window, entered):
        self.mouse_in_window = bool(entered)

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
            self.sample_rate = self.audio.rate
            self._recompute_bar_freqs()

        now = glfw.get_time()
        dt = max(1e-4, now - self.last_time)
        self.last_time = now

        chunk = self.audio.read_chunk()
        if chunk is None:
            self._update_peak_hold(dt)
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

        SUBSAMPLES = 6
        lo = self.bin_edges[:-1]
        hi = np.maximum(self.bin_edges[1:], lo + 1e-6)
        t = (np.arange(SUBSAMPLES) + 0.5) / SUBSAMPLES
        sample_positions = lo[:, None] + t[None, :] * (hi - lo)[:, None]
        sampled = np.interp(sample_positions.ravel(), self._fft_bin_axis, magnitude)
        bar_magnitudes = sampled.reshape(self.actual_num_bars, SUBSAMPLES).mean(axis=1).astype(np.float32)

        bar_magnitudes = (
            np.roll(bar_magnitudes, 2) * 0.08
            + np.roll(bar_magnitudes, 1) * 0.22
            + bar_magnitudes * 0.40
            + np.roll(bar_magnitudes, -1) * 0.22
            + np.roll(bar_magnitudes, -2) * 0.08
        ).astype(np.float32)

        db = 20.0 * np.log10(np.maximum(bar_magnitudes, 1e-10))
        db = np.clip(db, DB_FLOOR, DB_CEIL)

        self.smoothed_db = SMOOTHING * self.smoothed_db + (1.0 - SMOOTHING) * db

        self._update_peak_hold(dt, db)

    def _update_peak_hold(self, dt: float, db: np.ndarray = None):
        if db is not None:
            rising = db > self.peak_db
            self.peak_db = np.where(rising, db, self.peak_db)
            self._peak_age = np.where(rising, 0.0, self._peak_age + dt)
        else:
            self._peak_age += dt

        past_hold = self._peak_age >= PEAK_HOLD_SEC
        fall_amount = PEAK_FALL_DB_PER_SEC * dt
        self.peak_db = np.where(
            past_hold,
            np.maximum(self.peak_db - fall_amount, DB_FLOOR),
            self.peak_db,
        )

    def _freq_to_x_ndc(self, freq_hz: float) -> float:
        freq_hz = max(freq_hz, MIN_FREQ_HZ)
        t = np.log10(freq_hz / MIN_FREQ_HZ) / np.log10(MAX_FREQ_HZ / MIN_FREQ_HZ)
        t = min(max(t, 0.0), 1.0)
        return t * 2.0 - 1.0

    def _build_bar_vertices(self) -> np.ndarray:
        n = self.actual_num_bars
        level = (self.smoothed_db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)
        level = np.clip(level, 0.0, 1.0)
        colors = colormap(level)

        bin_hz = self.sample_rate / FFT_SIZE
        edge_freqs = self.bin_edges * bin_hz
        x_edges = np.array([self._freq_to_x_ndc(f) for f in edge_freqs], dtype=np.float32)

        y0 = -1.0
        verts = np.zeros((n * 6, 5), dtype=np.float32)

        gap_frac = 0.0
        max_gap_ndc = 0.0
        min_bar_width_ndc = 0.0012

        for i in range(n):
            x0, x1 = x_edges[i], x_edges[i + 1]
            width = x1 - x0
            gap = min(width * gap_frac, max_gap_ndc)
            bx0 = x0 + gap * 0.5
            bx1 = x1 - gap * 0.5
            if bx1 - bx0 < min_bar_width_ndc:
                mid = (x0 + x1) * 0.5
                bx0 = mid - min_bar_width_ndc * 0.5
                bx1 = mid + min_bar_width_ndc * 0.5
            y1 = -1.0 + level[i] * 2.0
            c = colors[i]

            base = i * 6
            verts[base + 0] = [bx0, y0, c[0], c[1], c[2]]
            verts[base + 1] = [bx1, y0, c[0], c[1], c[2]]
            verts[base + 2] = [bx1, y1, c[0], c[1], c[2]]
            verts[base + 3] = [bx0, y0, c[0], c[1], c[2]]
            verts[base + 4] = [bx1, y1, c[0], c[1], c[2]]
            verts[base + 5] = [bx0, y1, c[0], c[1], c[2]]

        self._bar_x_edges = x_edges
        return verts.reshape(-1)

    def _build_envelope_vertices(self) -> np.ndarray:
        n = self.actual_num_bars
        visible = self.bar_freqs_hz <= MAX_FREQ_HZ
        level = (self.peak_db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)
        level = np.clip(level, 0.0, 1.0)
        x_centers = np.array([self._freq_to_x_ndc(f) for f in self.bar_freqs_hz], dtype=np.float32)
        y = (-1.0 + level * 2.0).astype(np.float32)

        x_centers = x_centers[visible]
        y = y[visible]
        count = len(x_centers)
        verts = np.empty(count * 2, dtype=np.float32)
        verts[0::2] = x_centers
        verts[1::2] = y
        return verts, count

    def _hover_bar_index(self):
        if not self.mouse_in_window or self.width <= 0:
            return None
        x_ndc = (self.mouse_x / self.width) * 2.0 - 1.0
        edges = getattr(self, "_bar_x_edges", None)
        if edges is None:
            return None
        idx = np.searchsorted(edges, x_ndc) - 1
        if 0 <= idx < self.actual_num_bars:
            return int(idx)
        return None

    def _draw_axis_labels(self):
        label_freqs = [100, 1000, 10000]
        label_strs = ["100Hz", "1kHz", "10kHz"]
        top_y = 0.97
        for freq, label in zip(label_freqs, label_strs):
            x = self._freq_to_x_ndc(freq)
            self.text.draw(label, x, top_y, pixel_scale=2.0,
                            win_w=self.width, win_h=self.height,
                            color=(0.55, 0.55, 0.6), align="center")

    def _draw_gridlines(self):
        grid_freqs = [100, 1000, 10000]
        glUseProgram(self.line_program)
        glUniform4f(self.line_color_loc, 0.45, 0.45, 0.5, 0.35)
        glBindVertexArray(self.line_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        for freq in grid_freqs:
            x = self._freq_to_x_ndc(freq)
            verts = np.array([x, -1.0, x, 1.0], dtype=np.float32)
            glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
            glDrawArrays(GL_LINES, 0, 2)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _draw_hover(self):
        idx = self._hover_bar_index()
        if idx is None:
            return
        freq = self.bar_freqs_hz[idx]
        db = self.smoothed_db[idx]
        note = freq_to_note(freq)
        text = f"{db:.2f}dB {freq:.2f}Hz {note}"

        x_ndc = (self.mouse_x / self.width) * 2.0 - 1.0
        y_ndc = 1.0 - (self.mouse_y / self.height) * 2.0
        label_x = min(x_ndc, 0.5)
        self.text.draw(text, label_x, min(y_ndc + 0.05, 0.95), pixel_scale=2.5,
                        win_w=self.width, win_h=self.height,
                        color=(0.95, 0.95, 0.95), align="left")

        glUseProgram(self.line_program)
        glUniform4f(self.line_color_loc, 0.9, 0.9, 0.95, 0.5)
        glBindVertexArray(self.line_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        bx = self._freq_to_x_ndc(freq)
        verts = np.array([bx, -1.0, bx, 1.0], dtype=np.float32)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_LINES, 0, 2)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    # -----------------------------------------------------------------
    # help icon / overlay
    # -----------------------------------------------------------------

    def _draw_quad_window(self, x0, y0, x1, y1, color, alpha=1.0):
        glUniform3f(self.solid_color_loc, *color)
        glUniform1f(self.solid_alpha_loc, alpha)
        verts = np.array([x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1], dtype=np.float32)
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

        self._draw_gridlines()

        bar_verts = self._build_bar_vertices()
        glUseProgram(self.bar_program)
        glBindVertexArray(self.bar_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.bar_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, bar_verts.nbytes, bar_verts)
        glDrawArrays(GL_TRIANGLES, 0, self.actual_num_bars * 6)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        env_verts, env_count = self._build_envelope_vertices()
        glUseProgram(self.line_program)
        glUniform4f(self.line_color_loc, 0.98, 0.55, 0.65, 0.9)
        glBindVertexArray(self.line_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, env_verts.nbytes, env_verts)
        glDrawArrays(GL_LINE_STRIP, 0, env_count)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self._draw_axis_labels()
        self._draw_hover()

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


if __name__ == "__main__":
    app = SpectrumAnalyzerWindow()
    app.run()