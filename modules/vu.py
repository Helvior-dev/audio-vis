"""
VU meter module: horizontal dB level bar with peak-hold indicator.

Upgrades over the plain two-color bar:
  - continuous color gradient across the whole bar (blue -> amber ->
    red), computed per-pixel in the fragment shader instead of two
    flat-colored quads
  - tick marks with dB numbers along the top, like a real meter scale
  - the bar itself eases toward the target level (short attack/release)
    instead of snapping instantly, which is what makes analog VU
    meters read as "alive" rather than jumpy
  - peak marker keeps its hold-then-decay behavior

Digits are drawn with a 7-segment bitmap (segments = quads), not an
image font -- there's no text/font library wired into this from-
scratch OpenGL pipeline, and 7-segment quads are cheap to generate
and match the meter's technical look.
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

DB_MIN = -20.0
DB_MAX = 3.0
DB_ZERO = 0.0  # where the gradient crosses into the red zone

# peak marker decay: how many dB it falls per second when not re-triggered
PEAK_DECAY_DB_PER_SEC = 12.0

# bar easing: higher = snappier, lower = smoother/laggier (per second, exponential)
BAR_ATTACK_PER_SEC = 18.0   # rising (getting louder) -- fast, like real VU ballistics
BAR_RELEASE_PER_SEC = 6.0   # falling (getting quieter) -- slower

TICK_VALUES = [-20, -15, -10, -5, 0, 1, 2, 3]  # dB values to mark on the scale

BAR_VERTEX_SHADER = """
#version 330
in vec2 pos;
uniform float x_min;
uniform float x_max;
out float frag_t;  // 0..1 across the bar's own span, for the gradient
void main() {
    frag_t = (pos.x - x_min) / max(x_max - x_min, 0.0001);
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

# gradient: blue-gray (quiet) -> amber (approaching 0dB) -> red (overload).
# frag_t is 0..1 across DB_MIN..DB_MAX, so the stops below are placed at
# the same fractions of the full scale the reference meter uses.
BAR_FRAGMENT_SHADER = """
#version 330
in float frag_t;
out vec4 out_color;

void main() {
    vec3 low    = vec3(0.35, 0.55, 0.80);  // cool blue, quiet signal
    vec3 mid    = vec3(0.95, 0.75, 0.20);  // amber, approaching 0dB
    vec3 high   = vec3(0.90, 0.20, 0.15);  // red, overload

    float mid_point = (0.0 - (-20.0)) / (3.0 - (-20.0));  // where 0dB sits in 0..1

    vec3 color;
    if (frag_t < mid_point) {
        color = mix(low, mid, frag_t / mid_point);
    } else {
        color = mix(mid, high, (frag_t - mid_point) / (1.0 - mid_point));
    }
    out_color = vec4(color, 1.0);
}
"""

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


def db_to_x(db: float) -> float:
    """Map a dB value to NDC x-coordinate (-1..1) across the DB_MIN..DB_MAX range."""
    t = (db - DB_MIN) / (DB_MAX - DB_MIN)
    t = max(0.0, min(1.0, t))
    return t * 2.0 - 1.0


# 7-segment layout, each digit is 7 booleans: top, top-right, bottom-right,
# bottom, bottom-left, top-left, middle
SEGMENT_MAP = {
    "0": (1, 1, 1, 1, 1, 1, 0),
    "1": (0, 1, 1, 0, 0, 0, 0),
    "2": (1, 1, 0, 1, 1, 0, 1),
    "3": (1, 1, 1, 1, 0, 0, 1),
    "4": (0, 1, 1, 0, 0, 1, 1),
    "5": (1, 0, 1, 1, 0, 1, 1),
    "6": (1, 0, 1, 1, 1, 1, 1),
    "7": (1, 1, 1, 0, 0, 0, 0),
    "8": (1, 1, 1, 1, 1, 1, 1),
    "9": (1, 1, 1, 1, 0, 1, 1),
    "-": (0, 0, 0, 0, 0, 0, 1),
}


def digit_segment_quads(cx: float, cy: float, w: float, h: float, digit: str) -> list:
    """
    Returns a list of (x0, y0, x1, y1) quads (in the same coordinate
    space as cx/cy/w/h -- caller picks NDC-ish units) for the lit
    segments of one character. Layout is the standard 7-segment
    arrangement, thickness scaled from h.
    """
    segs = SEGMENT_MAP.get(digit)
    if segs is None:
        return []

    thickness = h * 0.16
    half_w = w * 0.5
    half_h = h * 0.5

    quads = []
    top, top_r, bot_r, bot, bot_l, top_l, mid = segs

    if top:
        quads.append((cx - half_w, cy + half_h - thickness, cx + half_w, cy + half_h))
    if bot:
        quads.append((cx - half_w, cy - half_h, cx + half_w, cy - half_h + thickness))
    if mid:
        quads.append((cx - half_w, cy - thickness / 2, cx + half_w, cy + thickness / 2))
    if top_l:
        quads.append((cx - half_w, cy, cx - half_w + thickness, cy + half_h))
    if bot_l:
        quads.append((cx - half_w, cy - half_h, cx - half_w + thickness, cy))
    if top_r:
        quads.append((cx + half_w - thickness, cy, cx + half_w, cy + half_h))
    if bot_r:
        quads.append((cx + half_w - thickness, cy - half_h, cx + half_w, cy))

    return quads


class VUWindow:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.width, self.height = 800, 140
        self.window = glfw.create_window(self.width, self.height, "VU Meter", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        apply_dark_titlebar(self.window)

        # gradient program: used for the main bar fill
        self.bar_program = link_program(BAR_VERTEX_SHADER, BAR_FRAGMENT_SHADER)
        self.bar_xmin_loc = glGetUniformLocation(self.bar_program, "x_min")
        self.bar_xmax_loc = glGetUniformLocation(self.bar_program, "x_max")

        # solid-color program: used for ticks, peak marker, digits
        self.solid_program = link_program(SOLID_VERTEX_SHADER, SOLID_FRAGMENT_SHADER)
        self.solid_color_loc = glGetUniformLocation(self.solid_program, "color")

        # main bar VAO/VBO (gradient shader)
        self.bar_vao = glGenVertexArrays(1)
        glBindVertexArray(self.bar_vao)
        self.bar_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.bar_vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # generic solid-quad VAO/VBO, reused for peak marker, ticks, digits,
        # and background track -- one shared dynamic buffer, rewritten per draw
        self.solid_vao = glGenVertexArrays(1)
        glBindVertexArray(self.solid_vao)
        self.solid_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.solid_vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.audio = AudioCapture(chunk_size=512)
        self.current_db = DB_MIN       # raw measured level this frame
        self.displayed_db = DB_MIN     # eased value actually drawn
        self.peak_db = DB_MIN
        self.last_time = glfw.get_time()

    def _on_resize(self, window, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    def _update_audio(self):
        chunk = self.audio.read_chunk()
        mono = chunk.mean(axis=1)
        rms = float(np.sqrt(np.mean(mono**2)))
        db = 20.0 * np.log10(rms) if rms > 1e-6 else DB_MIN
        db = max(DB_MIN, min(DB_MAX, db))
        self.current_db = db

        now = glfw.get_time()
        dt = now - self.last_time
        self.last_time = now

        # ease the displayed bar toward current_db -- attack (rising) is
        # faster than release (falling), matching real VU ballistics so
        # the needle "punches" on transients but settles gently after
        rate = BAR_ATTACK_PER_SEC if db > self.displayed_db else BAR_RELEASE_PER_SEC
        max_step = rate * dt
        diff = db - self.displayed_db
        if abs(diff) <= max_step:
            self.displayed_db = db
        else:
            self.displayed_db += max_step if diff > 0 else -max_step

        if db > self.peak_db:
            self.peak_db = db
        else:
            self.peak_db = max(DB_MIN, self.peak_db - PEAK_DECAY_DB_PER_SEC * dt)

    def _quad_verts(self, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        return np.array(
            [
                x0, y0,  x1, y0,  x1, y1,
                x0, y0,  x1, y1,  x0, y1,
            ],
            dtype=np.float32,
        )

    def _draw_solid_quad(self, x0, y0, x1, y1, color):
        glBindBuffer(GL_ARRAY_BUFFER, self.solid_vbo)
        verts = self._quad_verts(x0, x1, y0, y1)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glUniform3f(self.solid_color_loc, *color)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def _draw_number(self, value: int, cx: float, cy: float, digit_w: float, digit_h: float,
                      gap: float, color) -> None:
        """Draws an integer (may be negative) centered at (cx, cy)."""
        text = str(value)
        total_w = len(text) * digit_w + (len(text) - 1) * gap
        start_x = cx - total_w / 2 + digit_w / 2

        for i, ch in enumerate(text):
            dx = start_x + i * (digit_w + gap)
            for (x0, y0, x1, y1) in digit_segment_quads(dx, cy, digit_w * 0.7, digit_h, ch):
                self._draw_solid_quad(x0, y0, x1, y1, color)

    def render_frame(self):
        glClearColor(0.02, 0.02, 0.02, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        bar_y0, bar_y1 = -0.35, 0.35
        x_start = db_to_x(DB_MIN)
        x_end = db_to_x(DB_MAX)
        x_current = db_to_x(self.displayed_db)

        # background track (dim, shows the full scale even at low levels)
        glBindVertexArray(self.solid_vao)
        glUseProgram(self.solid_program)
        self._draw_solid_quad(x_start, bar_y0, x_end, bar_y1, (0.12, 0.12, 0.14))
        glBindVertexArray(0)

        # filled portion: gradient shader, anchored to the full DB_MIN..DB_MAX
        # span so the color at a given x always matches that dB value,
        # regardless of how much of the bar is currently filled
        if x_current > x_start:
            glBindVertexArray(self.bar_vao)
            glUseProgram(self.bar_program)
            glUniform1f(self.bar_xmin_loc, x_start)
            glUniform1f(self.bar_xmax_loc, x_end)
            glBindBuffer(GL_ARRAY_BUFFER, self.bar_vbo)
            verts = self._quad_verts(x_start, x_current, bar_y0, bar_y1)
            glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)

        glUseProgram(self.solid_program)
        glBindVertexArray(self.solid_vao)

        # tick marks + dB numbers along the top of the bar
        tick_h = 0.08
        for db_val in TICK_VALUES:
            x = db_to_x(db_val)
            tick_color = (0.85, 0.25, 0.2) if db_val > 0 else (0.55, 0.6, 0.68)
            self._draw_solid_quad(x - 0.0025, bar_y1, x + 0.0025, bar_y1 + tick_h, tick_color)
            self._draw_number(
                db_val, x, bar_y1 + tick_h + 0.14,
                digit_w=0.022, digit_h=0.09, gap=0.008, color=tick_color,
            )

        # peak marker: thin vertical bar at peak_db position
        peak_color = (0.95, 0.95, 0.95) if self.peak_db <= DB_ZERO else (0.95, 0.3, 0.25)
        x_peak = db_to_x(self.peak_db)
        peak_half_width = 0.005
        self._draw_solid_quad(
            x_peak - peak_half_width, bar_y0 - 0.05, x_peak + peak_half_width, bar_y1 + 0.05,
            peak_color,
        )

        # current numeric readout, bottom center
        readout_db = int(round(self.displayed_db))
        readout_color = (0.95, 0.95, 0.95) if readout_db <= 0 else (0.95, 0.3, 0.25)
        self._draw_number(
            readout_db, 0.0, bar_y0 - 0.28,
            digit_w=0.03, digit_h=0.14, gap=0.012, color=readout_color,
        )

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
    app = VUWindow()
    app.run()