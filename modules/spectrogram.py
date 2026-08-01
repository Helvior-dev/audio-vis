"""
Spectrogram module: frequency (Y) vs time (X) with color = amplitude.

Unlike the bar-based Spectrum module, this renders a scrolling 2D
texture: each column is one FFT frame, and new columns push in from
the right while old ones scroll left. The texture itself is the
history buffer -- we don't need to keep separate Python-side history,
just shift GPU texture data each frame.

Color mapping approximates the reference screenshots (dark purple for
quiet, magenta/pink through the mids, orange/red for loud) using a
simple 3-stop gradient instead of a full colormap library, since this
is a from-scratch OpenGL pipeline without matplotlib available.
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
NUM_FREQ_BINS = 128     # vertical resolution of the spectrogram (log-spaced)
HISTORY_WIDTH = 512     # horizontal resolution = how many past FFT frames are visible
DB_FLOOR = -70.0
DB_CEIL = 0.0

VERTEX_SHADER = """
#version 330
in vec2 pos;
in vec2 uv;
out vec2 frag_uv;
void main() {
    frag_uv = uv;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

# color ramp: dark purple (quiet) -> magenta -> orange (loud), matching
# the reference screenshots' palette. `level` is normalized 0..1.
FRAGMENT_SHADER = """
#version 330
in vec2 frag_uv;
uniform sampler2D tex;
out vec4 out_color;

vec3 colormap(float level) {
    vec3 dark_purple = vec3(0.10, 0.02, 0.15);
    vec3 magenta     = vec3(0.75, 0.15, 0.55);
    vec3 orange      = vec3(1.00, 0.55, 0.15);

    if (level < 0.5) {
        return mix(dark_purple, magenta, level * 2.0);
    } else {
        return mix(magenta, orange, (level - 0.5) * 2.0);
    }
}

void main() {
    float level = texture(tex, frag_uv).r;
    out_color = vec4(colormap(level), 1.0);
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


def make_log_bin_edges(num_bins: int, fft_size: int, min_bin: int = 1) -> np.ndarray:
    max_bin = fft_size // 2
    edges = np.logspace(np.log10(min_bin), np.log10(max_bin), num_bins + 1)
    return np.unique(edges.astype(int))


class SpectrogramWindow:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.width, self.height = 900, 300
        self.window = glfw.create_window(self.width, self.height, "Spectrogram", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        apply_dark_titlebar(self.window)

        self.program = link_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.tex_loc = glGetUniformLocation(self.program, "tex")

        # fullscreen quad (two triangles) with UV coords, static -- only the
        # texture contents change per frame, not this geometry
        quad = np.array(
            [
                # pos.x, pos.y, uv.x, uv.y
                -1, -1, 0, 0,
                 1, -1, 1, 0,
                 1,  1, 1, 1,
                -1, -1, 0, 0,
                 1,  1, 1, 1,
                -1,  1, 0, 1,
            ],
            dtype=np.float32,
        )
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        stride = 4 * 4  # 4 floats per vertex * 4 bytes
        glEnableVertexAttribArray(0)  # pos
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)  # uv
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # the spectrogram history texture: R32F, one row per freq bin,
        # one column per past FFT frame. This IS the history buffer --
        # updated by shifting + writing one new column each frame.
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        self.bin_edges = make_log_bin_edges(NUM_FREQ_BINS, FFT_SIZE)
        self.actual_num_bins = len(self.bin_edges) - 1

        # CPU-side mirror of the texture; we edit this and re-upload each
        # frame since glCopyTexSubImage (in-GPU shift) is more complex to
        # get right than just re-uploading a numpy roll -- HISTORY_WIDTH x
        # actual_num_bins is small enough (a few hundred KB) that this is
        # not a bottleneck at 144fps.
        self.history = np.zeros((self.actual_num_bins, HISTORY_WIDTH), dtype=np.float32)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R32F, HISTORY_WIDTH, self.actual_num_bins,
            0, GL_RED, GL_FLOAT, self.history,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

        self.read_chunk_size = 256
        self.audio = AudioCapture(chunk_size=self.read_chunk_size)
        self.rolling_buffer = np.zeros(FFT_SIZE, dtype=np.float32)
        self.window_fn = np.hanning(FFT_SIZE).astype(np.float32)

    def _on_resize(self, window, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    def _update_audio(self):
        chunk = self.audio.read_chunk()
        mono = chunk.mean(axis=1).astype(np.float32)
        n = len(mono)

        self.rolling_buffer = np.roll(self.rolling_buffer, -n)
        self.rolling_buffer[-n:] = mono

        windowed = self.rolling_buffer * self.window_fn
        spectrum = np.fft.rfft(windowed)
        # same normalization fix as spectrum.py -- unnormalized FFT
        # magnitude was reading ~40dB too hot, clipping most of the
        # display to DB_CEIL regardless of real signal level
        magnitude = np.abs(spectrum) / (self.window_fn.sum() / 2)

        bar_magnitudes = np.zeros(self.actual_num_bins, dtype=np.float32)
        for i in range(self.actual_num_bins):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            hi = max(hi, lo + 1)
            bar_magnitudes[i] = magnitude[lo:hi].mean()

        db = 20.0 * np.log10(np.maximum(bar_magnitudes, 1e-10))
        db = np.clip(db, DB_FLOOR, DB_CEIL)
        level = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)  # normalize to 0..1

        # scroll history left by one column, write new column at the right edge
        self.history = np.roll(self.history, -1, axis=1)
        self.history[:, -1] = level

    def render_frame(self):
        # ONE-SHOT DIAGNOSTIC: at frame 400 (well past the ~512-frame fill
        # window), dump the actual CPU-side array content for the leftmost
        # and rightmost columns, plus the exact bytes handed to OpenGL.
        # This tells us definitively whether the problem is Python-side
        # data (left columns still literally zero) or GPU/driver-side
        # (left columns are nonzero here but don't show on screen).
        if not hasattr(self, "_diag_frame"):
            self._diag_frame = 0
        self._diag_frame += 1
        if self._diag_frame == 400:
            print(f"[DIAG] frame={self._diag_frame}")
            print(f"[DIAG] self.history.shape={self.history.shape} dtype={self.history.dtype}")
            print(f"[DIAG] self.history.flags['C_CONTIGUOUS']={self.history.flags['C_CONTIGUOUS']}")
            print(f"[DIAG] leftmost 5 cols (row 45, mid-freq): {self.history[45, 0:5]}")
            print(f"[DIAG] rightmost 5 cols (row 45, mid-freq): {self.history[45, -5:]}")
            print(f"[DIAG] leftmost col ALL rows sum: {self.history[:, 0].sum():.4f}")
            print(f"[DIAG] rightmost col ALL rows sum: {self.history[:, -1].sum():.4f}")
            print(f"[DIAG] full array min/max/mean: {self.history.min():.4f} {self.history.max():.4f} {self.history.mean():.4f}")
            uploaded = np.ascontiguousarray(self.history, dtype=np.float32)
            print(f"[DIAG] uploaded array id == self.history id: {uploaded is self.history}")
            print(f"[DIAG] uploaded leftmost col sum: {uploaded[:, 0].sum():.4f}")
            print(f"[DIAG] HISTORY_WIDTH={HISTORY_WIDTH} actual_num_bins={self.actual_num_bins}")
            print(f"[DIAG] texture id={self.texture}")
            print(f"[DIAG] glGetError after nothing yet: {glGetError()}")

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0, HISTORY_WIDTH, self.actual_num_bins,
            GL_RED, GL_FLOAT, np.ascontiguousarray(self.history, dtype=np.float32),
        )
        if self._diag_frame == 400:
            err = glGetError()
            print(f"[DIAG] glGetError after glTexSubImage2D: {err}")
            # read back what the GPU actually has stored, column 0 and -1
            readback = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
            readback = np.frombuffer(readback, dtype=np.float32).reshape(self.actual_num_bins, HISTORY_WIDTH)
            print(f"[DIAG] GPU readback leftmost col sum: {readback[:, 0].sum():.4f}")
            print(f"[DIAG] GPU readback rightmost col sum: {readback[:, -1].sum():.4f}")
            print(f"[DIAG] GPU readback matches CPU array: {np.allclose(readback, self.history)}")

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(self.tex_loc, 0)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)

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
    app = SpectrogramWindow()
    app.run()