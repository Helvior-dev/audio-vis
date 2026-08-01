"""
Spectrum Analyzer module (formerly "Oscilloscope" -- replaced entirely).

Dense log-spaced FFT bar spectrum matching the reference look: many thin
vertical bars colored by amplitude (dark purple -> magenta -> orange),
a smoothed white envelope line drawn on top, frequency axis labels
(100Hz / 1kHz / 10kHz) across the top, vertical gridlines at those same
frequencies, and a hover tooltip showing dB / Hz / nearest musical note
under the cursor.

Rendering approach:
  - Bars: one GL_TRIANGLES draw call for all bars, vertex color = per-bar
    colormap value baked into the vertex buffer each frame (position +
    color interleaved). Far cheaper than hundreds of draw calls or
    hundreds of uniform changes.
  - Envelope: GL_LINE_STRIP over the bar tops, solid white, slightly
    translucent via alpha blending.
  - Frequency labels + hover text: rasterized once per string into an
    8x8-per-glyph bitmap (numpy-generated bitmap font, no external font
    file needed) and uploaded as a texture, drawn as a textured quad.
    This is simpler and more legible than extending the 7-segment digit
    renderer (vu.py's SEGMENT_MAP) to handle letters like H/z/k/#/+/-.
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

FFT_SIZE = 4096
NUM_BARS = 320          # dense, thin bars -- close to the reference density
DB_FLOOR = -70.0
DB_CEIL = 0.0
SMOOTHING = 0.65         # bar smoothing (lower than spectrum.py -- reference looks fairly live)
ENVELOPE_SMOOTHING = 0.82  # envelope eases slower than the bars for a "peak trace" feel

MIN_FREQ_HZ = 20.0        # bottom of the displayed/labelled range
MAX_FREQ_HZ = 20000.0     # top of the displayed/labelled range

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


def make_log_bin_edges(num_bars: int, fft_size: int, min_bin: float = 1.0) -> np.ndarray:
    """
    Returns num_bars+1 log-spaced edges as FLOATS (not deduplicated
    integer bin indices). Using float edges + interpolation (see
    _update_audio) instead of integer-bin grouping is what lets NUM_BARS
    be honored exactly even when it's larger than the number of distinct
    low-frequency FFT bins available -- grouping by rounded integer bins
    collapses many adjacent log-spaced edges into the same bin at low
    frequencies and silently caps the achievable bar count well below
    what a dense reference-style display needs.
    """
    max_bin = fft_size // 2
    return np.logspace(np.log10(min_bin), np.log10(max_bin), num_bars + 1)


def colormap(level: np.ndarray) -> np.ndarray:
    """
    Vectorized version of the same dark-purple -> magenta -> orange ramp
    used in spectrogram.py, applied per-bar instead of per-texel.
    level: array shape (N,) in 0..1. Returns array shape (N, 3).
    """
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


# -------------------------------------------------------------------
# Minimal bitmap font -- just the glyphs this module actually needs.
# Each glyph is a 5x7 grid of 0/1 (classic tiny bitmap font size),
# stored as a list of row-strings (top to bottom, '1' = lit pixel).
# -------------------------------------------------------------------

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
        rows = GLYPH_MAP_LOOKUP(ch)
        x0 = i * cell_w
        for y, row in enumerate(rows):
            for x, bit in enumerate(row):
                if bit == "1":
                    canvas[y, x0 + x] = 255
    return canvas


def GLYPH_MAP_LOOKUP(ch: str):
    return GLYPH_MAP.get(ch, GLYPH_MAP[" "])


GLYPH_MAP = GLYPH_5X7


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note(freq_hz: float) -> str:
    """Nearest musical note + cents offset, e.g. 'F1 +2 Cents'. A4 = 440Hz."""
    if freq_hz <= 0:
        return "-- "
    semitones_from_a4 = 12.0 * np.log2(freq_hz / 440.0)
    nearest = round(semitones_from_a4)
    cents = int(round((semitones_from_a4 - nearest) * 100))
    midi_note = 69 + nearest  # A4 = MIDI 69
    name = NOTE_NAMES[midi_note % 12]
    octave = midi_note // 12 - 1
    sign = "+" if cents >= 0 else "-"
    return f"{name}{octave} {sign}{abs(cents)}Cents"


class TextRenderer:
    """
    Uploads one texture per unique string (cached by string) and draws it
    as a textured quad at a given NDC position. Cache is small (a handful
    of frequency labels + one hover string that changes rarely enough).
    """

    def __init__(self, program: int):
        self.program = program
        self.uv_loc = None
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

        # convert glyph-bitmap size (w,h in bitmap pixels) to NDC size
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
        self.window = glfw.create_window(self.width, self.height, "Spectrum Analyzer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
        glfw.set_cursor_enter_callback(self.window, self._on_cursor_enter)
        apply_dark_titlebar(self.window)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # --- bar program (per-vertex color) ---
        self.bar_program = link_program(BAR_VERTEX_SHADER, BAR_FRAGMENT_SHADER)
        self.bar_vao = glGenVertexArrays(1)
        glBindVertexArray(self.bar_vao)
        self.bar_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.bar_vbo)
        # 6 verts per bar (2 triangles) * NUM_BARS, 5 floats per vert (pos2+color3)
        self.max_bar_verts = NUM_BARS * 6
        glBufferData(GL_ARRAY_BUFFER, self.max_bar_verts * 5 * 4, None, GL_DYNAMIC_DRAW)
        stride = 5 * 4
        glEnableVertexAttribArray(0)  # pos
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)  # color
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # --- line program (envelope + gridlines) ---
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

        # --- text program ---
        self.text_program = link_program(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)
        self.text = TextRenderer(self.text_program)

        # --- audio / FFT state ---
        self.read_chunk_size = 256
        self.audio = AudioCapture(chunk_size=self.read_chunk_size)
        self.rolling_buffer = np.zeros(FFT_SIZE, dtype=np.float32)
        self.window_fn = np.hanning(FFT_SIZE).astype(np.float32)
        self.bin_edges = make_log_bin_edges(NUM_BARS, FFT_SIZE)  # float edges, length NUM_BARS+1
        self.actual_num_bars = NUM_BARS
        self.smoothed_db = np.full(self.actual_num_bars, DB_FLOOR, dtype=np.float32)
        self.envelope_db = np.full(self.actual_num_bars, DB_FLOOR, dtype=np.float32)
        self.sample_rate = self.audio.rate

        # per-bar center frequency (Hz), used for axis labels + hover note lookup
        bin_hz = self.sample_rate / FFT_SIZE
        bar_center_bin = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        self.bar_freqs_hz = bar_center_bin * bin_hz
        self._fft_bin_axis = np.arange(FFT_SIZE // 2 + 1, dtype=np.float64)

        # mouse / hover state
        self.mouse_x, self.mouse_y = -1.0, -1.0
        self.mouse_in_window = False

    def _on_resize(self, window, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    def _on_cursor_move(self, window, xpos, ypos):
        self.mouse_x, self.mouse_y = xpos, ypos

    def _on_cursor_enter(self, window, entered):
        self.mouse_in_window = bool(entered)

    def _update_audio(self):
        chunk = self.audio.read_chunk()
        mono = chunk.mean(axis=1).astype(np.float32)
        n = len(mono)

        self.rolling_buffer = np.roll(self.rolling_buffer, -n)
        self.rolling_buffer[-n:] = mono

        windowed = self.rolling_buffer * self.window_fn
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum) / (self.window_fn.sum() / 2)

        # sample the (linearly-interpolated) magnitude spectrum at several
        # points across each bar's [lo, hi) float bin-edge span and average
        # them -- this is the float-edge equivalent of the old integer
        # magnitude[lo:hi].mean(), but works when hi-lo < 1 (which happens
        # constantly at low frequencies once NUM_BARS exceeds the number
        # of whole FFT bins available there)
        SUBSAMPLES = 3
        lo = self.bin_edges[:-1]
        hi = np.maximum(self.bin_edges[1:], lo + 1e-6)
        t = (np.arange(SUBSAMPLES) + 0.5) / SUBSAMPLES  # shape (SUBSAMPLES,)
        sample_positions = lo[:, None] + t[None, :] * (hi - lo)[:, None]  # (num_bars, SUBSAMPLES)
        sampled = np.interp(sample_positions.ravel(), self._fft_bin_axis, magnitude)
        bar_magnitudes = sampled.reshape(self.actual_num_bars, SUBSAMPLES).mean(axis=1).astype(np.float32)

        db = 20.0 * np.log10(np.maximum(bar_magnitudes, 1e-10))
        db = np.clip(db, DB_FLOOR, DB_CEIL)

        self.smoothed_db = SMOOTHING * self.smoothed_db + (1.0 - SMOOTHING) * db
        self.envelope_db = ENVELOPE_SMOOTHING * self.envelope_db + (1.0 - ENVELOPE_SMOOTHING) * db

    def _freq_to_x_ndc(self, freq_hz: float) -> float:
        """Log-scale mapping of a frequency to NDC x, across MIN_FREQ_HZ..MAX_FREQ_HZ."""
        freq_hz = max(freq_hz, MIN_FREQ_HZ)
        t = np.log10(freq_hz / MIN_FREQ_HZ) / np.log10(MAX_FREQ_HZ / MIN_FREQ_HZ)
        t = min(max(t, 0.0), 1.0)
        return t * 2.0 - 1.0

    def _build_bar_vertices(self) -> np.ndarray:
        n = self.actual_num_bars
        level = (self.smoothed_db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)
        level = np.clip(level, 0.0, 1.0)
        colors = colormap(level)

        # x positions: log-spaced across the full width like the reference
        # (not evenly spaced by bar index -- low bars are wide, high bars thin),
        # computed directly from the float bin_edges -> Hz -> NDC
        bin_hz = self.sample_rate / FFT_SIZE
        edge_freqs = self.bin_edges * bin_hz
        x_edges = np.array([self._freq_to_x_ndc(f) for f in edge_freqs], dtype=np.float32)

        y0 = -1.0
        verts = np.zeros((n * 6, 5), dtype=np.float32)
        gap_frac = 0.12  # small gap between bars for the "many thin bars" look

        for i in range(n):
            x0, x1 = x_edges[i], x_edges[i + 1]
            width = x1 - x0
            gap = width * gap_frac
            bx0 = x0 + gap * 0.5
            bx1 = x1 - gap * 0.5
            if bx1 <= bx0:
                bx1 = bx0 + max(width, 0.0008)
            y1 = -1.0 + level[i] * 2.0
            c = colors[i]

            base = i * 6
            verts[base + 0] = [bx0, y0, c[0], c[1], c[2]]
            verts[base + 1] = [bx1, y0, c[0], c[1], c[2]]
            verts[base + 2] = [bx1, y1, c[0], c[1], c[2]]
            verts[base + 3] = [bx0, y0, c[0], c[1], c[2]]
            verts[base + 4] = [bx1, y1, c[0], c[1], c[2]]
            verts[base + 5] = [bx0, y1, c[0], c[1], c[2]]

        self._bar_x_edges = x_edges  # cached for hover lookup
        return verts.reshape(-1)

    def _build_envelope_vertices(self) -> np.ndarray:
        n = self.actual_num_bars
        level = (self.envelope_db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)
        level = np.clip(level, 0.0, 1.0)
        bin_hz = self.sample_rate / FFT_SIZE
        x_centers = np.array([self._freq_to_x_ndc(f) for f in self.bar_freqs_hz], dtype=np.float32)
        y = (-1.0 + level * 2.0).astype(np.float32)
        verts = np.empty(n * 2, dtype=np.float32)
        verts[0::2] = x_centers
        verts[1::2] = y
        return verts

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
        # keep the label from running off the right edge
        label_x = min(x_ndc, 0.5)
        self.text.draw(text, label_x, min(y_ndc + 0.05, 0.95), pixel_scale=2.5,
                        win_w=self.width, win_h=self.height,
                        color=(0.95, 0.95, 0.95), align="left")

        # vertical marker line at the hovered bar
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

    def render_frame(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        self._draw_gridlines()

        # bars
        bar_verts = self._build_bar_vertices()
        glUseProgram(self.bar_program)
        glBindVertexArray(self.bar_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.bar_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, bar_verts.nbytes, bar_verts)
        glDrawArrays(GL_TRIANGLES, 0, self.actual_num_bars * 6)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # envelope line on top
        env_verts = self._build_envelope_vertices()
        glUseProgram(self.line_program)
        glUniform4f(self.line_color_loc, 0.85, 0.88, 0.95, 0.85)
        glBindVertexArray(self.line_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, env_verts.nbytes, env_verts)
        glDrawArrays(GL_LINE_STRIP, 0, self.actual_num_bars)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self._draw_axis_labels()
        self._draw_hover()

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
    app = SpectrumAnalyzerWindow()
    app.run()