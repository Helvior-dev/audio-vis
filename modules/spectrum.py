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

sys.path.insert(0, str(Path(__file__).parent.parent))
from audio_capture import AudioCapture
from window_utils import apply_dark_titlebar

FFT_SIZE = 2048
NUM_BARS = 64
DB_FLOOR = -70.0   # anything quieter than this renders as an empty bar
DB_CEIL = 0.0
SMOOTHING = 0.75    # 0 = no smoothing (jumpy), closer to 1 = smoother but laggier

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
        apply_dark_titlebar(self.window)

        self.program = link_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.color_loc = glGetUniformLocation(self.program, "color")

        # one shared dynamic quad buffer, position rewritten per bar per frame
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.read_chunk_size = 256
        self.audio = AudioCapture(chunk_size=self.read_chunk_size)
        self.rolling_buffer = np.zeros(FFT_SIZE, dtype=np.float32)
        self.window_fn = np.hanning(FFT_SIZE).astype(np.float32)
        self.bin_edges = make_log_bin_edges(NUM_BARS, FFT_SIZE)
        self.actual_num_bars = len(self.bin_edges) - 1
        self.smoothed_db = np.full(self.actual_num_bars, DB_FLOOR, dtype=np.float32)

    def _on_resize(self, window, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    def _update_audio(self):
        chunk = self.audio.read_chunk()  # (read_chunk_size, channels)
        if chunk is None:
            return
        mono = chunk.mean(axis=1).astype(np.float32)
        n = len(mono)

        # A window resize/drag stalls the render loop for a bit (GLFW
        # blocks pumping frames during the native resize drag on
        # Windows), so several audio callbacks' worth of chunks pile up
        # in AudioCapture's queue. The next read_chunk() call then
        # returns all of them concatenated -- n can come back larger
        # than FFT_SIZE (rolling_buffer's fixed size). Only the most
        # recent FFT_SIZE samples are relevant to the rolling window
        # anyway, so trim before the roll/assign -- same fix as
        # spectrum_analyzer.py's _update_audio.
        if n > FFT_SIZE:
            mono = mono[-FFT_SIZE:]
            n = FFT_SIZE

        # slide the rolling buffer and append the new samples at the end
        self.rolling_buffer = np.roll(self.rolling_buffer, -n)
        self.rolling_buffer[-n:] = mono

        windowed = self.rolling_buffer * self.window_fn
        spectrum = np.fft.rfft(windowed)
        # normalize by the window's coherent gain (sum/2) -- raw FFT
        # magnitude scales with FFT_SIZE and the Hann window's energy
        # loss, so without this a -12dBFS signal reads as +41dB and
        # every bar clips to the ceiling regardless of actual volume
        magnitude = np.abs(spectrum) / (self.window_fn.sum() / 2)

        # group FFT bins into log-spaced bars by averaging magnitude within each range
        bar_magnitudes = np.zeros(self.actual_num_bars, dtype=np.float32)
        for i in range(self.actual_num_bars):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            hi = max(hi, lo + 1)  # ensure at least one bin per bar
            bar_magnitudes[i] = magnitude[lo:hi].mean()

        # convert to dB, avoiding log(0)
        db = 20.0 * np.log10(np.maximum(bar_magnitudes, 1e-10))
        db = np.clip(db, DB_FLOOR, DB_CEIL)

        # exponential smoothing toward the new value so bars don't jitter frame-to-frame
        self.smoothed_db = SMOOTHING * self.smoothed_db + (1.0 - SMOOTHING) * db

    def _quad_verts(self, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        return np.array(
            [
                x0, y0,  x1, y0,  x1, y1,
                x0, y0,  x1, y1,  x0, y1,
            ],
            dtype=np.float32,
        )

    def render_frame(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        n = self.actual_num_bars
        bar_width = 2.0 / n
        gap = bar_width * 0.15  # small gap between bars

        for i in range(n):
            x0 = -1.0 + i * bar_width + gap * 0.5
            x1 = -1.0 + (i + 1) * bar_width - gap * 0.5

            # normalize dB to [0, 1] then to y-range [-1, 1]
            level = (self.smoothed_db[i] - DB_FLOOR) / (DB_CEIL - DB_FLOOR)
            level = max(0.0, min(1.0, level))
            y0 = -1.0
            y1 = -1.0 + level * 2.0

            verts = self._quad_verts(x0, x1, y0, y1)
            glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

            # color gradient: blue-ish for normal levels, red only near the
            # top of the range (close to 0dB / clipping), matching how the
            # VU meter reserves red for actual overload rather than "loud".
            if level < 0.93:
                glUniform3f(self.color_loc, 0.55, 0.68, 0.85)
            else:
                glUniform3f(self.color_loc, 0.9, 0.3, 0.25)

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
    app = SpectrumWindow()
    app.run()