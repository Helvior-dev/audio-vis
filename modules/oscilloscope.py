"""
Oscilloscope module: real-time waveform with correlation-based trigger.

Difference from Waveform: this module re-aligns the displayed window
every frame to whichever point best matches what was already on
screen ("Follow Pitch" in the reference app). A plain zero-crossing
trigger only stabilizes a clean single-frequency tone -- on real
music (many frequencies + noise) the "first" zero-crossing jumps to
a different point almost every frame, so the display drifts exactly
like Waveform's plain scroll. Picking the zero-crossing whose
following samples correlate best with the previous frame locks onto
the repeating shape instead, so the trace actually holds still on
program material, not just synthetic test tones.
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

BUFFER_SIZE = 4096      # how much history we keep to search for a trigger point in
DISPLAY_SAMPLES = 1024  # how many samples wide the actual displayed window is

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
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


def find_trigger_index(buf: np.ndarray, search_start: int, search_end: int,
                        reference: np.ndarray | None = None) -> int:
    """
    Finds the best trigger point for a stable display.

    Zero-crossing alone only works on a clean single-frequency tone --
    real music is a sum of many frequencies plus noise, so "first
    rising zero-crossing" jumps to a different point almost every
    frame and the waveform drifts exactly like it would with no
    trigger at all (this was the actual bug: Waveform and Oscilloscope
    looked identical because the trigger wasn't doing anything on
    program material).

    Fix: among the candidate zero-crossings, pick the one whose
    following samples correlate best with `reference` (last frame's
    displayed slice). This is a cheap stand-in for what MiniMeters'
    "Follow Pitch" does -- it locks onto whichever repeating shape was
    already on screen instead of just the nearest zero-crossing, so
    the display holds still even on complex/noisy material.
    """
    window = buf[search_start:search_end]
    signs = np.sign(window)
    crossings = np.where((signs[:-1] <= 0) & (signs[1:] > 0))[0]

    if len(crossings) == 0:
        return search_start
    if reference is None or len(crossings) == 1:
        return search_start + int(crossings[0])

    display_len = len(reference)
    compare_len = min(256, display_len)  # correlate on a short leading
                                          # segment -- cheap and enough
                                          # to distinguish good candidates
    ref_segment = reference[:compare_len]

    best_idx = int(crossings[0])
    best_score = -np.inf
    for c in crossings:
        candidate_end = search_start + c + compare_len
        if candidate_end > len(buf):
            continue
        segment = buf[search_start + c : candidate_end]
        # normalized correlation so loudness differences don't bias the pick
        denom = (np.linalg.norm(segment) * np.linalg.norm(ref_segment)) + 1e-9
        score = float(np.dot(segment, ref_segment) / denom)
        if score > best_score:
            best_score = score
            best_idx = int(c)

    return search_start + best_idx


class OscilloscopeWindow:
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
        self.window = glfw.create_window(self.width, self.height, "Oscilloscope", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        apply_dark_titlebar(self.window)

        self.program = link_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.x_step_loc = glGetUniformLocation(self.program, "x_step")

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, DISPLAY_SAMPLES * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.audio = AudioCapture(chunk_size=256)
        self.rolling_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.display_slice = np.zeros(DISPLAY_SAMPLES, dtype=np.float32)

    def _on_resize(self, window, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    def _update_audio(self):
        chunk = self.audio.read_chunk()
        mono = chunk.mean(axis=1).astype(np.float32)
        n = len(mono)

        self.rolling_buffer = np.roll(self.rolling_buffer, -n)
        self.rolling_buffer[-n:] = mono

        # search for a trigger point in the first half of the buffer so
        # there's always enough room after it to fill DISPLAY_SAMPLES
        search_end = BUFFER_SIZE - DISPLAY_SAMPLES
        trigger_idx = find_trigger_index(
            self.rolling_buffer, 0, search_end, reference=self.display_slice
        )

        self.display_slice = self.rolling_buffer[trigger_idx : trigger_idx + DISPLAY_SAMPLES]

    def render_frame(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.display_slice.nbytes, self.display_slice)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glUniform1f(self.x_step_loc, 2.0 / (DISPLAY_SAMPLES - 1))

        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINE_STRIP, 0, DISPLAY_SAMPLES)
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
    app = OscilloscopeWindow()
    app.run()