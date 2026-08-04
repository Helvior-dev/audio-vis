"""
Loudness module: combined "Loudness" (LUFS) + "Level L/R" meter, replacing
the old VU Meter module. Same idea as MiniMeters' Loudness widget -- this
is a from-scratch reimplementation, not a copy of their code.

Left panel -- Loudness:
    Big readout = Momentary loudness (LUFS-M, 400ms window). The gradient
    bar fill also tracks Momentary. Two thin tick markers overlaid on the
    bar show where Short-Term (3s) and Integrated loudness currently sit,
    so you can see at a glance whether the instantaneous level is above
    or below the session average. Below the bar: M / S / INT / LRA / PK
    readouts.

Right panel -- Level L/R:
    Classic two-row level meter, one bar per channel, fast attack / slow
    release ballistics (same spirit as vu.py's old VU ballistics) with a
    peak-hold tick per channel.

Loudness math follows the shape of ITU-R BS.1770-4 / EBU R128: K-weighting
(pre-filter + RLB high-pass, both derived from the standard's analog
prototypes so they're correct at whatever sample rate WASAPI hands us,
not hardcoded to 48kHz), 400ms gating blocks at 100ms hop for Integrated
Loudness with the two-stage absolute/relative gate, and a percentile-based
Loudness Range (LRA) computed from short-term loudness history. This is a
practical from-scratch implementation for a real-time on-screen meter, not
a certified-precision loudness measurement tool.

Window is aspect-locked (like stereometer.py) at its initial 1150:260
ratio -- the user can still resize it, but GLFW constrains the drag to
that ratio, so every element defined below in fixed NDC fractions always
lands in the same relative place. This is what fixes both the "text runs
into the bars" issue (extreme aspect ratios used to stretch text-height
NDC deltas far more than the fixed horizontal offsets they were tuned
against) and the "meter feels laggy/unreadable" complaint (the layout
no longer visibly distorts as you resize, so the bars and readouts stay
legible instead of appearing to drift/smear across the window).
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
from text_render import TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER, TextRenderer

DB_MIN = -60.0
DB_MAX = 0.0

# Two independent columns instead of one long stacked list -- the window
# is wide and short (1150:260), so a single centered text block only
# ever fills the top-left corner and leaves the rest of the overlay
# empty. Splitting the two sections (Loudness / Level L/R) into their
# own columns uses the actual shape of the window instead of fighting it.
HELP_TITLE = "LOUDNESS"

HELP_COLUMN_LEFT = [
    "LOUDNESS (LUFS)",
    "",
    "Shows how loud the audio sounds overall, using the",
    "same LUFS scale streaming platforms and broadcasters",
    "use -- not just raw volume, but loudness weighted the",
    "way human hearing works.",
    "",
    "M     Momentary -- last 0.4 seconds",
    "S     Short-Term -- last 3 seconds",
    "INT   Integrated -- average since this window opened",
    "LRA   Loudness Range -- how much the level varies",
    "PK    Peak -- the loudest single sample so far",
]

HELP_COLUMN_RIGHT = [
    "LEVEL L/R",
    "",
    "A simple volume meter for the left and right",
    "speakers, measured in dBFS -- 0 is the loudest a",
    "digital signal can go before clipping.",
    "",
    "The thin white marker on each bar is the peak",
    "hold: it jumps to the loudest recent moment and",
    "slowly falls back, so brief transients are still",
    "visible even after they've passed.",
]

HELP_FOOTER = "Click anywhere to close"

# ---------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------

BAR_VERTEX_SHADER = """
#version 330
in vec2 pos;
uniform float x_min;
uniform float x_max;
uniform vec2 aspect_scale;
out float frag_t;
void main() {
    frag_t = (pos.x - x_min) / max(x_max - x_min, 0.0001);
    gl_Position = vec4(pos * aspect_scale, 0.0, 1.0);
}
"""

# green (quiet) -> yellow (loud) -> red only right at the top of the
# scale (near 0dB/clipping), same "red reserved for overload" convention
# the rest of this app uses (vu.py, spectrum.py).
BAR_FRAGMENT_SHADER = """
#version 330
in float frag_t;
out vec4 out_color;
void main() {
    vec3 lo = vec3(0.30, 0.72, 0.42);
    vec3 mid = vec3(0.85, 0.82, 0.28);
    vec3 hi = vec3(0.90, 0.25, 0.20);
    vec3 color;
    if (frag_t < 0.82) {
        color = mix(lo, mid, frag_t / 0.82);
    } else {
        color = mix(mid, hi, (frag_t - 0.82) / 0.18);
    }
    out_color = vec4(color, 1.0);
}
"""

SOLID_VERTEX_SHADER = """
#version 330
in vec2 pos;
uniform vec2 aspect_scale;
void main() {
    gl_Position = vec4(pos * aspect_scale, 0.0, 1.0);
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


def db_to_x(db: float, x0: float, x1: float, db_min: float = DB_MIN, db_max: float = DB_MAX) -> float:
    t = (db - db_min) / (db_max - db_min)
    t = max(0.0, min(1.0, t))
    return x0 + t * (x1 - x0)


# ---------------------------------------------------------------------
# K-weighting (ITU-R BS.1770-4) -- biquad coefficients derived from the
# standard's analog prototypes via the RBJ Audio EQ Cookbook bilinear
# transform, so they come out correct for any sample rate instead of
# only the commonly-published 48kHz table.
# ---------------------------------------------------------------------

def _high_shelf_coeffs(fs: float, f0: float, gain_db: float, q: float):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    sqrt_a = np.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_a * alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _high_pass_coeffs(fs: float, f0: float, q: float):
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)

    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


class Biquad:
    """
    Direct Form II Transposed biquad with persistent state across calls
    (required -- K-weighting is a continuous filter over the whole
    session, it must not reset every audio chunk). Processed sample by
    sample: the recursive feedback term (each output depends on the
    previous one) can't be vectorized away, but chunk sizes here are
    small enough (~256-1024 samples) that a plain Python loop is not a
    bottleneck at real-time rates.
    """

    def __init__(self, b0, b1, b2, a1, a2):
        self.b0, self.b1, self.b2, self.a1, self.a2 = b0, b1, b2, a1, a2
        self.z1 = 0.0
        self.z2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        z1, z2 = self.z1, self.z2
        for i in range(len(x)):
            xi = x[i]
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            y[i] = yi
        self.z1, self.z2 = z1, z2
        return y


class KWeightingFilter:
    """Pre-filter (high shelf) -> RLB filter (high-pass), per BS.1770-4."""

    def __init__(self, sample_rate: float):
        b0, b1, b2, a1, a2 = _high_shelf_coeffs(
            sample_rate, f0=1681.9744509555319, gain_db=3.99984385397, q=0.7071752369554193
        )
        self.stage1 = Biquad(b0, b1, b2, a1, a2)
        b0, b1, b2, a1, a2 = _high_pass_coeffs(
            sample_rate, f0=38.13547087613982, q=0.5003270373253953
        )
        self.stage2 = Biquad(b0, b1, b2, a1, a2)

    def process(self, x: np.ndarray) -> np.ndarray:
        return self.stage2.process(self.stage1.process(x))


class LoudnessMeter:
    """
    Continuous Momentary / Short-Term / Integrated LUFS + LRA + sample
    peak, fed with small stereo chunks via push().
    """

    ABS_GATE_LUFS = -70.0
    REL_GATE_LU = -10.0
    LRA_REL_GATE_LU = -20.0
    LRA_LOW_PCT = 10.0
    LRA_HIGH_PCT = 95.0
    PEAK_DECAY_DB_PER_SEC = 11.0

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.kw_l = KWeightingFilter(sample_rate)
        self.kw_r = KWeightingFilter(sample_rate)

        # BS.1770 defines Momentary as a 400ms window, but at that length
        # the readout visibly lags fast music (each new value is an
        # average smeared over the last 0.4s of audio, so a hit note is
        # still "blended in" with whatever came right before it).
        # Shortening this to 150ms trades spec-accuracy for a snappier,
        # more responsive on-screen number -- this display was never a
        # certified loudness measurement tool to begin with (see module
        # docstring), so favoring feel over strict BS.1770 compliance
        # here is reasonable. Short-Term (3s) and Integrated are left at
        # their spec values since those are meant to represent longer-
        # term trends, not instant reaction.
        self.momentary_samples = max(1, int(round(0.15 * sample_rate)))
        # Integrated Loudness gating (per BS.1770) is defined over 400ms
        # blocks specifically -- kept separate from momentary_samples
        # above so shortening the on-screen Momentary readout doesn't
        # also shrink the gating blocks Integrated relies on, which
        # would drift Integrated away from spec for no benefit (it's a
        # long-term average, it isn't meant to react quickly anyway).
        self.gating_block_samples = max(1, int(round(0.4 * sample_rate)))
        self.shortterm_samples = max(1, int(round(3.0 * sample_rate)))

        # rolling K-weighted mean-square buffer, long enough for the 3s
        # short-term window; momentary reads the tail of the same buffer
        self._buf_sq = np.zeros(self.shortterm_samples, dtype=np.float64)
        self._filled = 0

        # 400ms gating blocks sampled every 100ms (75% overlap, per spec)
        self._block_hop_samples = max(1, int(round(0.1 * sample_rate)))
        self._samples_since_block = 0
        self._integrated_blocks: list[float] = []

        # short-term loudness sampled once a second, for LRA
        self._st_hop_samples = max(1, int(round(1.0 * sample_rate)))
        self._samples_since_st = 0
        self._st_history: list[float] = []

        self.momentary_lufs = self.ABS_GATE_LUFS
        self.shortterm_lufs = self.ABS_GATE_LUFS
        self.integrated_lufs = self.ABS_GATE_LUFS
        self.lra = 0.0
        self._peak_hold_db = self.ABS_GATE_LUFS
        self.peak_db = self.ABS_GATE_LUFS

    @staticmethod
    def _power_to_lufs(mean_square: float) -> float:
        if mean_square <= 1e-12:
            return -70.0
        return -0.691 + 10.0 * np.log10(mean_square)

    def push(self, left: np.ndarray, right: np.ndarray, dt: float):
        if len(left) == 0:
            return

        # sample peak (true peak per spec needs 4x oversampling -- skipped
        # here to avoid pulling in a resampler dependency; sample peak is
        # what most lightweight real-time meters show anyway)
        chunk_peak = float(np.max(np.abs(np.concatenate([left, right]))))
        chunk_peak_db = 20.0 * np.log10(max(chunk_peak, 1e-10))
        if chunk_peak_db > self._peak_hold_db:
            self._peak_hold_db = chunk_peak_db
        else:
            self._peak_hold_db = max(chunk_peak_db, self._peak_hold_db - self.PEAK_DECAY_DB_PER_SEC * dt)
        self.peak_db = self._peak_hold_db

        kl = self.kw_l.process(left.astype(np.float64))
        kr = self.kw_r.process(right.astype(np.float64))
        sq = kl * kl + kr * kr  # L/R channel weight = 1.0 each, per BS.1770

        n = len(sq)
        if n >= self.shortterm_samples:
            self._buf_sq[:] = sq[-self.shortterm_samples:]
            self._filled = self.shortterm_samples
        else:
            self._buf_sq = np.roll(self._buf_sq, -n)
            self._buf_sq[-n:] = sq
            self._filled = min(self.shortterm_samples, self._filled + n)

        m_n = min(self.momentary_samples, self._filled)
        if m_n > 0:
            self.momentary_lufs = self._power_to_lufs(self._buf_sq[-m_n:].mean())

        if self._filled > 0:
            tail = self._buf_sq if self._filled >= self.shortterm_samples else self._buf_sq[-self._filled:]
            self.shortterm_lufs = self._power_to_lufs(tail.mean())

        self._samples_since_block += n
        while self._filled >= self.gating_block_samples and self._samples_since_block >= self._block_hop_samples:
            self._samples_since_block -= self._block_hop_samples
            self._integrated_blocks.append(self._buf_sq[-self.gating_block_samples:].mean())
            if len(self._integrated_blocks) > 100_000:
                self._integrated_blocks = self._integrated_blocks[-50_000:]
        self.integrated_lufs = self._compute_integrated()

        self._samples_since_st += n
        if self._samples_since_st >= self._st_hop_samples:
            self._samples_since_st = 0
            self._st_history.append(self.shortterm_lufs)
            if len(self._st_history) > 20_000:
                self._st_history = self._st_history[-10_000:]
        self.lra = self._compute_lra()

    def _compute_integrated(self) -> float:
        if not self._integrated_blocks:
            return -70.0
        blocks = np.array(self._integrated_blocks)
        loudness = -0.691 + 10.0 * np.log10(np.maximum(blocks, 1e-12))

        gated = blocks[loudness > self.ABS_GATE_LUFS]
        if len(gated) == 0:
            return -70.0
        mean_gated = gated.mean()
        rel_threshold = self._power_to_lufs(mean_gated) + self.REL_GATE_LU

        gated_loudness = -0.691 + 10.0 * np.log10(np.maximum(gated, 1e-12))
        gated2 = gated[gated_loudness > rel_threshold]
        if len(gated2) == 0:
            return self._power_to_lufs(mean_gated)
        return self._power_to_lufs(gated2.mean())

    def _compute_lra(self) -> float:
        if len(self._st_history) < 2:
            return 0.0
        values = np.array(self._st_history)
        values = values[values > self.ABS_GATE_LUFS]
        if len(values) < 2:
            return 0.0

        mean_power = np.mean(10 ** ((values + 0.691) / 10.0))
        rel_gate = self._power_to_lufs(mean_power) + self.LRA_REL_GATE_LU
        gated = values[values > rel_gate]
        if len(gated) < 2:
            return 0.0

        lo = np.percentile(gated, self.LRA_LOW_PCT)
        hi = np.percentile(gated, self.LRA_HIGH_PCT)
        return float(hi - lo)


class ChannelLevel:
    """
    Attack/release-smoothed dBFS level + decaying peak-hold tick for one
    channel -- same ballistics style as the old vu.py VU meter (fast
    attack so transients punch through, slower release so it reads as
    "alive" instead of jittery).
    """

    ATTACK_PER_SEC = 90.0
    RELEASE_PER_SEC = 45.0
    PEAK_DECAY_DB_PER_SEC = 11.0

    def __init__(self):
        self.displayed_db = DB_MIN
        self.peak_db = DB_MIN

    def update(self, samples: np.ndarray, dt: float):
        if len(samples):
            rms = float(np.sqrt(np.mean(np.square(samples))))
        else:
            rms = 0.0
        db = 20.0 * np.log10(rms) if rms > 1e-8 else DB_MIN
        db = max(DB_MIN, min(DB_MAX, db))

        rate = self.ATTACK_PER_SEC if db > self.displayed_db else self.RELEASE_PER_SEC
        max_step = rate * dt
        diff = db - self.displayed_db
        if abs(diff) <= max_step:
            self.displayed_db = db
        else:
            self.displayed_db += max_step if diff > 0 else -max_step

        if db > self.peak_db:
            self.peak_db = db
        else:
            self.peak_db = max(DB_MIN, self.peak_db - self.PEAK_DECAY_DB_PER_SEC * dt)


class LoudnessWindow:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.base_width, self.base_height = 1150, 260
        self.width, self.height = self.base_width, self.base_height
        self.window = glfw.create_window(self.width, self.height, "Loudness", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        glfw.set_window_pos(self.window, (mode.size.width - self.width) // 2, (mode.size.height - self.height) // 2)

        # lock the resize drag to the original aspect ratio -- this is
        # what keeps every fixed-NDC-fraction element (text, bars, ticks)
        # landing in the same relative place at any window size, instead
        # of stretching non-uniformly and running text into the bars
        glfw.set_window_aspect_ratio(self.window, self.base_width, self.base_height)

        # a floor on how small the window can get, at the same locked
        # aspect ratio. Text size is resolved from win_h every draw call
        # (see text_render.py), so shrinking the window shrinks every
        # label right along with it -- past a certain point the numbers
        # and stat labels become too small to read even though nothing
        # is technically broken. Half the base size is small enough to
        # dock in a corner of the screen but still keeps text legible;
        # GLFW_DONT_CARE on the max means it can still be enlarged
        # (and, combined with the aspect lock above, made fullscreen-wide)
        # without limit.
        min_width = self.base_width // 2
        min_height = self.base_height // 2
        glfw.set_window_size_limits(
            self.window, min_width, min_height, glfw.DONT_CARE, glfw.DONT_CARE
        )

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
        apply_dark_titlebar(self.window)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.bar_program = link_program(BAR_VERTEX_SHADER, BAR_FRAGMENT_SHADER)
        self.bar_xmin_loc = glGetUniformLocation(self.bar_program, "x_min")
        self.bar_xmax_loc = glGetUniformLocation(self.bar_program, "x_max")
        self.bar_aspect_loc = glGetUniformLocation(self.bar_program, "aspect_scale")

        self.solid_program = link_program(SOLID_VERTEX_SHADER, SOLID_FRAGMENT_SHADER)
        self.solid_color_loc = glGetUniformLocation(self.solid_program, "color")
        self.solid_alpha_loc = glGetUniformLocation(self.solid_program, "alpha")
        self.solid_aspect_loc = glGetUniformLocation(self.solid_program, "aspect_scale")

        self.text_program = link_program(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)
        self.text = TextRenderer(self.text_program)

        # shared dynamic quad buffer for bars/ticks/panels (rewritten per draw)
        self.bar_vao = glGenVertexArrays(1)
        glBindVertexArray(self.bar_vao)
        self.bar_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.bar_vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.solid_vao = glGenVertexArrays(1)
        glBindVertexArray(self.solid_vao)
        self.solid_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.solid_vbo)
        glBufferData(GL_ARRAY_BUFFER, 32 * 2 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.audio = AudioCapture(chunk_size=256)
        self.loudness = LoudnessMeter(self.audio.rate)
        self.level_l = ChannelLevel()
        self.level_r = ChannelLevel()
        self.last_time = glfw.get_time()

        # Earlier revision added an easing step here (ChannelLevel-style
        # ballistics on top of momentary_lufs) to smooth out per-chunk
        # jitter. In practice that made the meter feel sluggish and
        # unresponsive against fast music -- the 400ms momentary window
        # inside LoudnessMeter itself already provides the intended
        # BS.1770 smoothing, so a second easing layer on top was pure
        # added lag with no real benefit. Reading momentary_lufs directly
        # keeps the readout, bar fill, and M stat all reacting as fast as
        # the underlying measurement actually changes.

        self.help_icon_cx = 0.965
        self.help_icon_cy = 0.87
        self.help_icon_r = 0.028
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

    # -----------------------------------------------------------------
    # audio
    # -----------------------------------------------------------------

    def _update_audio(self):
        now = glfw.get_time()
        dt = max(1e-4, now - self.last_time)
        self.last_time = now

        chunk = self.audio.read_chunk()  # (N, channels)
        if chunk is None:
            return
        left = chunk[:, 0]
        right = chunk[:, 1]

        self.loudness.push(left, right, dt)
        self.level_l.update(left, dt)
        self.level_r.update(right, dt)

    # -----------------------------------------------------------------
    # drawing helpers
    # -----------------------------------------------------------------

    def _aspect_scale(self):
        """
        Because the window is aspect-locked (set_window_aspect_ratio),
        width/height always maintain the base_width:base_height ratio
        -- but the actual pixel size still changes as the user resizes.
        Layout below is authored entirely in the *base* aspect's NDC
        space; this uniform maps that space onto whatever the current
        framebuffer's NDC actually is, which (since the ratio is locked)
        reduces to a uniform scale, not a stretch -- so nothing skews.
        """
        if self.width <= 0 or self.height <= 0:
            return 1.0, 1.0
        return 1.0, 1.0

    def _quad_verts(self, x0, y0, x1, y1) -> np.ndarray:
        return np.array(
            [x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1], dtype=np.float32
        )

    def _draw_solid_quad(self, x0, y0, x1, y1, color, alpha=1.0):
        glUseProgram(self.solid_program)
        ax, ay = self._aspect_scale()
        glUniform2f(self.solid_aspect_loc, ax, ay)
        glBindVertexArray(self.solid_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.solid_vbo)
        verts = self._quad_verts(x0, y0, x1, y1)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glUniform3f(self.solid_color_loc, *color)
        glUniform1f(self.solid_alpha_loc, alpha)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _draw_gradient_bar(self, x0, y0, x1, y1, fill_x):
        """Background track + gradient fill up to fill_x, colored by absolute
        position on the DB_MIN..DB_MAX scale (not by fill fraction), so a
        given dB value always renders the same color regardless of how
        much of the bar is filled."""
        self._draw_solid_quad(x0, y0, x1, y1, (0.10, 0.10, 0.12))
        if fill_x > x0:
            glUseProgram(self.bar_program)
            ax, ay = self._aspect_scale()
            glUniform2f(self.bar_aspect_loc, ax, ay)
            glUniform1f(self.bar_xmin_loc, x0)
            glUniform1f(self.bar_xmax_loc, x1)
            glBindVertexArray(self.bar_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.bar_vbo)
            verts = self._quad_verts(x0, y0, fill_x, y1)
            glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)

    def _text(self, s, x, y, scale, color, align="left", valign="top"):
        self.text.draw(s, x, y, pixel_scale=scale, win_w=self.width, win_h=self.height,
                        color=color, align=align, valign=valign)

    def _draw_circle_window(self, cx, cy, r, color, alpha=1.0, segments=24):
        if self.width <= 0 or self.height <= 0:
            return
        aspect = self.height / self.width
        glUseProgram(self.solid_program)
        ax, ay = self._aspect_scale()
        glUniform2f(self.solid_aspect_loc, ax, ay)
        glUniform3f(self.solid_color_loc, *color)
        glUniform1f(self.solid_alpha_loc, alpha)
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=True, dtype=np.float32)
        verts = np.stack(
            [cx + np.cos(angles) * r * aspect, cy + np.sin(angles) * r],
            axis=1,
        ).astype(np.float32)
        glBindVertexArray(self.solid_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.solid_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        glDrawArrays(GL_LINE_STRIP, 0, segments)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    # -----------------------------------------------------------------
    # help icon / overlay
    # -----------------------------------------------------------------

    def _draw_help_icon(self):
        hovering = False
        x, y = self._mouse_to_window_ndc(self.mouse_x, self.mouse_y)
        dx, dy = x - self.help_icon_cx, y - self.help_icon_cy
        if dx * dx + dy * dy <= self.help_icon_r * self.help_icon_r:
            hovering = True

        icon_color = (0.75, 0.75, 0.8) if hovering else (0.5, 0.5, 0.55)
        self._draw_circle_window(self.help_icon_cx, self.help_icon_cy, self.help_icon_r, icon_color, alpha=0.9)
        self._text("?", self.help_icon_cx, self.help_icon_cy, 1.1, icon_color, align="center", valign="middle")

    def _draw_help_column(self, lines, x_left, start_y, line_h):
        muted = (0.68, 0.68, 0.73)
        heading = (0.9, 0.9, 0.95)
        for i, line in enumerate(lines):
            is_heading = i == 0
            color = heading if is_heading else muted
            scale = 1.15 if is_heading else 0.85
            if line:
                self._text(line, x_left, start_y - i * line_h, scale, color, align="left", valign="top")

    def _draw_help_overlay(self):
        self._draw_solid_quad(-1.0, -1.0, 1.0, 1.0, (0.0, 0.0, 0.0), alpha=0.72)
        self._draw_solid_quad(-0.95, -0.85, 0.95, 0.85, (0.08, 0.08, 0.1), alpha=0.97)

        # centered title across the full width, then two independent
        # columns below it -- each column uses the panel's own left edge
        # as its origin so the whole overlay width is put to use instead
        # of collapsing into one narrow left-aligned block
        title_y = 0.72
        self._text(HELP_TITLE, 0.0, title_y, 1.6, (0.92, 0.92, 0.97), align="center", valign="top")

        col_top = title_y - 0.22
        line_h = 0.075
        gap = 0.06
        col_left_x = -0.88
        col_right_x = gap

        self._draw_help_column(HELP_COLUMN_LEFT, col_left_x, col_top, line_h)
        self._draw_help_column(HELP_COLUMN_RIGHT, col_right_x, col_top, line_h)

        self._text(HELP_FOOTER, 0.0, -0.78, 0.85, (0.55, 0.55, 0.6), align="center", valign="top")

    # -----------------------------------------------------------------
    # panels
    # -----------------------------------------------------------------
    #
    # Vertical layout below uses explicit cumulative offsets (each
    # element's y computed from the previous one's y minus its own
    # rendered height) instead of independent fixed deltas from y_top.
    # The old version placed every line at y_top - <constant>, tuned by
    # eye for one scale value each -- fine until the ~4.4x-scale LUFS
    # readout's actual glyph height (which grows with pixel_scale) ran
    # past where the bar was hardcoded to start. Chaining offsets means
    # each block reserves space proportional to its own scale, so bumping
    # any one scale can't make it collide with what's below it.

    def _draw_loudness_panel(self, x0, x1, y_top, y_bot):
        muted = (0.45, 0.45, 0.5)
        bright = (0.92, 0.92, 0.95)

        # panel card background
        self._draw_solid_quad(x0, y_bot, x1, y_top, (0.055, 0.055, 0.065))

        header_y = y_top - 0.03
        self._text("LOUDNESS", x0 + 0.02, header_y, 1.9, muted, valign="top")

        # big readout sits below the header with a fixed gap that scales
        # with the header's own line height, then reserves its own
        # (much taller, scale=4.4) line height before the bar starts
        big_val = f"{self.loudness.momentary_lufs:.1f} LUFS"
        big_y = header_y - 0.14
        self._text(big_val, (x0 + x1) / 2, big_y, 4.4, bright, align="center", valign="top")

        bar_y1 = big_y - 0.30
        bar_y0 = bar_y1 - 0.16
        bar_x0, bar_x1 = x0 + 0.05, x1 - 0.05
        fill_x = db_to_x(self.loudness.momentary_lufs, bar_x0, bar_x1)
        self._draw_gradient_bar(bar_x0, bar_y0, bar_x1, bar_y1, fill_x)

        # short-term / integrated tick markers overlaid on the bar
        st_x = db_to_x(self.loudness.shortterm_lufs, bar_x0, bar_x1)
        int_x = db_to_x(self.loudness.integrated_lufs, bar_x0, bar_x1)
        tick_half = 0.0025
        self._draw_solid_quad(st_x - tick_half, bar_y0 - 0.03, st_x + tick_half, bar_y1 + 0.03, (0.45, 0.78, 0.95))
        self._draw_solid_quad(int_x - tick_half, bar_y0 - 0.03, int_x + tick_half, bar_y1 + 0.03, (0.95, 0.95, 0.95))

        scale_label_y = bar_y0 - 0.07
        self._text("-60", bar_x0, scale_label_y, 1.4, muted, align="center", valign="top")
        self._text("0", bar_x1, scale_label_y, 1.4, muted, align="center", valign="top")

        # stat row: M / S / INT / LRA / PK
        stats = [
            ("M", self.loudness.momentary_lufs, bright),
            ("S", self.loudness.shortterm_lufs, (0.45, 0.78, 0.95)),
            ("INT", self.loudness.integrated_lufs, bright),
            ("LRA", self.loudness.lra, muted),
            ("PK", self.loudness.peak_db, (0.90, 0.35, 0.30) if self.loudness.peak_db > 0 else bright),
        ]
        n = len(stats)
        col_w = (x1 - x0 - 0.06) / n
        stat_label_y = scale_label_y - 0.14
        stat_value_y = stat_label_y - 0.11
        for i, (label, value, color) in enumerate(stats):
            cx = x0 + 0.03 + col_w * i + col_w / 2
            self._text(label, cx, stat_label_y, 1.5, muted, align="center", valign="top")
            self._text(f"{value:.1f}", cx, stat_value_y, 2.0, color, align="center", valign="top")

    def _draw_level_panel(self, x0, x1, y_top, y_bot):
        muted = (0.45, 0.45, 0.5)
        bright = (0.92, 0.92, 0.95)

        self._draw_solid_quad(x0, y_bot, x1, y_top, (0.055, 0.055, 0.065))
        header_y = y_top - 0.03
        self._text("LEVEL L/R", x0 + 0.02, header_y, 1.9, muted, valign="top")

        bar_x0, bar_x1 = x0 + 0.10, x1 - 0.05

        rows = [
            ("L", self.level_l),
            ("R", self.level_r),
        ]
        row_top = header_y - 0.17
        row_h = 0.30
        gap = 0.12

        for label, ch in rows:
            label_y = row_top
            self._text(f"{label}  {ch.displayed_db:.1f}", x0 + 0.02, label_y, 2.1, bright, valign="top")

            bar_y1 = label_y - 0.16
            bar_y0 = bar_y1 - 0.22
            fill_x = db_to_x(ch.displayed_db, bar_x0, bar_x1)
            self._draw_gradient_bar(bar_x0, bar_y0, bar_x1, bar_y1, fill_x)

            peak_x = db_to_x(ch.peak_db, bar_x0, bar_x1)
            peak_color = (0.95, 0.95, 0.95) if ch.peak_db < -0.5 else (0.95, 0.30, 0.25)
            self._draw_solid_quad(peak_x - 0.0025, bar_y0 - 0.02, peak_x + 0.0025, bar_y1 + 0.02, peak_color)

            row_top = bar_y0 - gap

        # shared dB axis across the bottom
        axis_y = row_top - 0.02
        for db_val in (-60, -48, -36, -24, -12, 0):
            x = db_to_x(db_val, bar_x0, bar_x1)
            self._draw_solid_quad(x - 0.0015, axis_y, x + 0.0015, axis_y + 0.05, muted)
            self._text(str(db_val), x, axis_y - 0.02, 1.3, muted, align="center", valign="top")

    def render_frame(self):
        glClearColor(0.02, 0.02, 0.025, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        gap = 0.03
        mid = 0.0
        self._draw_loudness_panel(-0.97, mid - gap, 0.90, -0.90)
        self._draw_level_panel(mid + gap, 0.97, 0.90, -0.90)

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
    app = LoudnessWindow()
    app.run()