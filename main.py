"""
Launcher window: one button per visualization module.
Each module runs as a separate process (not thread) so a crash/freeze
in one visualizer (e.g. Spectrum Analyzer) never takes down the others
or the launcher itself.

Styled to match the dark, card-based look of the visualizer windows
themselves, instead of stock Tk widgets.
"""

import base64
import subprocess
import sys
import tkinter as tk
import webbrowser
from pathlib import Path

from modules.window_utils import apply_dark_titlebar_tk, set_window_icon_tk
from modules.audio_capture import AudioSource, list_audio_sources
from modules.audio_source_config import save_selected_source, load_selected_source

MODULES_DIR = Path(__file__).parent / "modules"
ICON_PATH = MODULES_DIR / "assets" / "icon.png"

REPO_URL = "https://github.com/Helvior-dev/audio-vis"

# GitHub octocat mark, embedded as base64 PNG (Octicons "mark-github",
# 16x16 viewBox, MIT-licensed -- https://primer.style/foundations/icons),
# rendered at 64x64 for a crisp small icon. Two pre-rendered color states
# (idle/hover) instead of colorizing at runtime, since tk.PhotoImage has
# no built-in tint operation.
GITHUB_ICON_IDLE_B64 = (
    "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"
)

GITHUB_ICON_HOVER_B64 = (
    "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"
    "SUVORK5CYII="
)


# display name -> (script filename, icon key)
MODULES = [
    ("Loudness",          "loudness.py",          "bars"),
    ("Oscilloscope",           "oscilloscope.py",          "wave"),
    ("Spectrum",            "spectrum.py",          "spectrum"),
    ("Stereometer",       "stereometer.py",       "dots"),
    ("Spectrum Analyzer", "spectrum_analyzer.py", "spectrum_dense"),
]

# ---------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------
BG            = "#0b0b0d"
CARD_BG       = "#17171b"
CARD_BG_HOVER = "#1f1f25"
CARD_BORDER   = "#26262c"
TEXT_PRIMARY  = "#e9e9ee"
TEXT_MUTED    = "#6f6f79"
ACCENT        = "#3ddc84"   # running-state dot
DOT_IDLE      = "#3a3a42"
ICON_COLOR    = "#9b9ba6"


def round_rect(canvas: tk.Canvas, x0, y0, x1, y1, radius=12, **kwargs):
    points = [
        x0 + radius, y0,
        x1 - radius, y0,
        x1, y0,
        x1, y0 + radius,
        x1, y1 - radius,
        x1, y1,
        x1 - radius, y1,
        x0 + radius, y1,
        x0, y1,
        x0, y1 - radius,
        x0, y0 + radius,
        x0, y0,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class ModuleButton(tk.Canvas):
    """A single rounded, dark card representing one visualizer module."""

    WIDTH, HEIGHT = 320, 58

    def __init__(self, parent, label: str, icon_key: str, on_click):
        super().__init__(
            parent, width=self.WIDTH, height=self.HEIGHT,
            bg=BG, highlightthickness=0, bd=0, cursor="hand2",
        )
        self.on_click = on_click
        self.running = False
        self._hover = False
        self.label = label
        self.icon_key = icon_key

        self._draw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", lambda _e: self.on_click())

    def _draw(self):
        self.delete("all")
        bg = CARD_BG_HOVER if self._hover else CARD_BG
        round_rect(self, 1, 1, self.WIDTH - 1, self.HEIGHT - 1, radius=12,
                   fill=bg, outline=CARD_BORDER, width=1)

        self._draw_icon(self.icon_key, 30, self.HEIGHT / 2)

        self.create_text(
            58, self.HEIGHT / 2, text=self.label, anchor="w",
            fill=TEXT_PRIMARY, font=("Segoe UI", 11),
        )

        dot_color = ACCENT if self.running else DOT_IDLE
        d = 7
        self.create_oval(
            self.WIDTH - 22 - d, self.HEIGHT / 2 - d / 2,
            self.WIDTH - 22, self.HEIGHT / 2 + d / 2,
            fill=dot_color, outline="",
        )

    def _draw_icon(self, key: str, cx: float, cy: float):
        c = ICON_COLOR
        if key == "bars":
            for i, h in enumerate([7, 13, 9, 15]):
                x = cx - 9 + i * 5
                self.create_rectangle(x, cy + 8 - h, x + 3, cy + 8, fill=c, outline="")
        elif key == "wave":
            self.create_line(
                cx - 9, cy, cx - 5, cy - 8, cx - 1, cy + 8, cx + 3, cy - 8, cx + 7, cy + 8, cx + 10, cy,
                fill=c, width=2, smooth=True, capstyle="round", joinstyle="round",
            )
        elif key == "spectrum":
            for i, h in enumerate([6, 12, 16, 10, 14, 8]):
                x = cx - 9 + i * 3.4
                self.create_rectangle(x, cy + 8 - h, x + 2, cy + 8, fill=c, outline="")
        elif key == "spectrum_dense":
            for i, h in enumerate([4, 9, 13, 8, 15, 6, 11, 5]):
                x = cx - 9 + i * 2.4
                self.create_rectangle(x, cy + 8 - h, x + 1.4, cy + 8, fill=c, outline="")
        elif key == "dots":
            for dx, dy in [(-6, -2), (-2, 5), (3, -6), (6, 2), (0, 0), (-4, 6), (5, -4)]:
                self.create_oval(cx + dx - 1.3, cy + dy - 1.3, cx + dx + 1.3, cy + dy + 1.3,
                                  fill=c, outline="")

    def set_running(self, running: bool):
        self.running = running
        self._draw()

    def _on_enter(self, _event):
        self._hover = True
        self._draw()

    def _on_leave(self, _event):
        self._hover = False
        self._draw()


class GitHubFooter(tk.Canvas):
    """
    Small clickable row at the bottom of the launcher: GitHub octocat
    icon + "by Helvior" text. Opens REPO_URL in the system's default
    browser on click -- webbrowser.open() shells out to the OS handler
    (start/open/xdg-open under the hood), so there's no extra dependency
    and no need to know which browser the user has.

    The icon itself is the real Octicons "mark-github" glyph, pre-
    rendered to two small PNGs (idle/hover color) and embedded as
    base64 (GITHUB_ICON_IDLE_B64 / GITHUB_ICON_HOVER_B64) -- drawing a
    recognizable octocat out of tk.Canvas primitives (ovals/polygons)
    at this size doesn't hold up, it reads as an unrecognizable blob
    rather than the GitHub mark, so this uses the actual icon instead.
    """

    WIDTH, HEIGHT = 320, 28
    HOVER_COLOR = "#e9e9ee"
    ICON_SIZE = 18  # on-screen size in px; source PNGs are 64x64, downscaled here

    def __init__(self, parent, url: str, label: str = "by Helvior"):
        super().__init__(
            parent, width=self.WIDTH, height=self.HEIGHT,
            bg=BG, highlightthickness=0, bd=0, cursor="hand2",
        )
        self.url = url
        self.label = label
        self._hover = False

        # tk.PhotoImage needs its source bytes kept alive for as long as
        # the image is in use -- storing on self (not a local var) so
        # they aren't garbage-collected after __init__ returns, which
        # would blank the icon
        self._icon_idle = self._load_icon(GITHUB_ICON_IDLE_B64)
        self._icon_hover = self._load_icon(GITHUB_ICON_HOVER_B64)

        self._draw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", lambda _e: webbrowser.open(self.url))

    def _load_icon(self, b64_data: str) -> tk.PhotoImage | None:
        raw = base64.b64decode(b64_data)
        try:
            img = tk.PhotoImage(data=raw)
        except tk.TclError:
            # Tk < 8.6 has no built-in PNG decoder in PhotoImage -- fall
            # back to no icon (a plain dot is drawn instead, see _draw())
            # rather than crashing the whole launcher over a footer icon
            return None
        # source PNG is rendered at 64x64 for crispness; subsample down
        # to the actual on-screen size instead of drawing it oversized
        factor = max(1, img.width() // self.ICON_SIZE)
        if factor > 1:
            img = img.subsample(factor, factor)
        return img

    def _draw(self):
        self.delete("all")
        color = self.HOVER_COLOR if self._hover else TEXT_MUTED
        icon = self._icon_hover if self._hover else self._icon_idle
        cy = self.HEIGHT / 2
        if icon is not None:
            self.create_image(14, cy, image=icon, anchor="center")
        else:
            self.create_oval(8, cy - 6, 20, cy + 6, fill=color, outline="")
        self.create_text(
            30, cy, text=self.label, anchor="w",
            fill=color, font=("Segoe UI", 9),
        )

    def _on_enter(self, _event):
        self._hover = True
        self._draw()

    def _on_leave(self, _event):
        self._hover = False
        self._draw()


class SourceSelectorButton(tk.Canvas):
    """
    Header control showing the currently selected audio source
    ("Default system audio", a specific device name, "Default
    microphone", ...). Clicking it opens a small popup menu grouped
    into "System audio" and "Microphone" sections; picking an entry
    saves it to the shared config file, which every already-open
    visualizer window picks up on its own within a frame or two (see
    SourceWatcher in audio_source_config.py) -- nothing here talks to
    the running subprocesses directly.
    """

    WIDTH, HEIGHT = 320, 34
    ICON_COLOR = "#7fb8e0"

    def __init__(self, parent, on_change=None):
        super().__init__(
            parent, width=self.WIDTH, height=self.HEIGHT,
            bg=BG, highlightthickness=0, bd=0, cursor="hand2",
        )
        self.on_change = on_change
        self._hover = False
        self.current: AudioSource = load_selected_source()
        self._popup: tk.Toplevel | None = None

        self._draw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", lambda _e: self._open_popup())

    def _draw(self):
        self.delete("all")
        bg = CARD_BG_HOVER if self._hover else CARD_BG
        round_rect(self, 1, 1, self.WIDTH - 1, self.HEIGHT - 1, radius=9,
                   fill=bg, outline=CARD_BORDER, width=1)

        # small mic/speaker glyph so the control reads as audio-source-y
        # at a glance, not just a generic settings button
        cx, cy = 18, self.HEIGHT / 2
        if self.current.kind == "microphone":
            self.create_oval(cx - 4, cy - 7, cx + 4, cy + 2, outline=self.ICON_COLOR, width=1.4)
            self.create_line(cx, cy + 2, cx, cy + 7, fill=self.ICON_COLOR, width=1.4)
            self.create_line(cx - 5, cy + 7, cx + 5, cy + 7, fill=self.ICON_COLOR, width=1.4)
        else:
            self.create_rectangle(cx - 6, cy - 4, cx - 2, cy + 4, outline=self.ICON_COLOR, width=1.4)
            self.create_polygon(cx - 2, cy - 4, cx + 5, cy - 8, cx + 5, cy + 8, cx - 2, cy + 4,
                                 outline=self.ICON_COLOR, fill="", width=1.4)

        label = self.current.name
        if len(label) > 30:
            label = label[:29] + "\u2026"
        self.create_text(
            32, self.HEIGHT / 2, text=label, anchor="w",
            fill=TEXT_PRIMARY, font=("Segoe UI", 9),
        )

        # small chevron on the right, hinting this opens something
        chevron_x = self.WIDTH - 16
        self.create_line(chevron_x - 4, self.HEIGHT / 2 - 3, chevron_x, self.HEIGHT / 2 + 2,
                          chevron_x + 4, self.HEIGHT / 2 - 3, fill=TEXT_MUTED, width=1.4,
                          joinstyle="round", capstyle="round")

    def _on_enter(self, _event):
        self._hover = True
        self._draw()

    def _on_leave(self, _event):
        self._hover = False
        self._draw()

    def _open_popup(self):
        if self._popup is not None and tk.Toplevel.winfo_exists(self._popup):
            self._popup.destroy()
            self._popup = None
            return

        loopback_sources, mic_sources = list_audio_sources()

        popup = tk.Toplevel(self)
        popup.overrideredirect(True)
        popup.configure(bg=CARD_BG, highlightthickness=1, highlightbackground=CARD_BORDER)
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.HEIGHT + 4
        popup.geometry(f"+{x}+{y}")
        self._popup = popup

        def add_section(title, sources):
            tk.Label(
                popup, text=title, bg=CARD_BG, fg=TEXT_MUTED,
                font=("Segoe UI", 8, "bold"), anchor="w",
            ).pack(fill="x", padx=10, pady=(8, 2))
            for src in sources:
                is_current = (src.kind == self.current.kind and src.device_index == self.current.device_index)
                row = tk.Label(
                    popup, text=("\u2713  " if is_current else "    ") + src.name,
                    bg=CARD_BG, fg=(TEXT_PRIMARY if is_current else TEXT_MUTED),
                    font=("Segoe UI", 9), anchor="w", cursor="hand2",
                )
                row.pack(fill="x", padx=10, pady=1)

                def choose(_e, s=src):
                    self._select(s)
                    popup.destroy()
                    self._popup = None

                row.bind("<Button-1>", choose)
                row.bind("<Enter>", lambda _e, w=row: w.configure(bg=CARD_BG_HOVER))
                row.bind("<Leave>", lambda _e, w=row: w.configure(bg=CARD_BG))

        add_section("SYSTEM AUDIO", loopback_sources)
        add_section("MICROPHONE", mic_sources)
        tk.Frame(popup, bg=CARD_BG, height=8).pack(fill="x")

        # close the popup if the user clicks anywhere else
        def on_focus_out(_e=None):
            if self._popup is not None:
                self._popup.destroy()
                self._popup = None

        popup.bind("<FocusOut>", on_focus_out)
        popup.after(50, popup.focus_force)

    def _select(self, source: AudioSource):
        self.current = source
        save_selected_source(source)
        self._draw()
        if self.on_change:
            self.on_change(source)


class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Audio Visualizer")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        apply_dark_titlebar_tk(self.root)
        # keep the PhotoImage alive on self -- Tk only holds a weak
        # reference internally, so a purely local variable here would
        # get garbage-collected and the icon would vanish later
        self.icon_img = set_window_icon_tk(self.root, ICON_PATH)

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=26, pady=(24, 6))
        tk.Label(
            header, text="AUDIO VISUALIZER", bg=BG, fg=TEXT_PRIMARY,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header, text="real-time audio meters", bg=BG, fg=TEXT_MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))

        self.source_selector = SourceSelectorButton(header)
        self.source_selector.pack(anchor="w", pady=(10, 0))

        divider = tk.Frame(root, bg=CARD_BORDER, height=1)
        divider.pack(fill="x", padx=26, pady=(14, 4))

        body = tk.Frame(root, bg=BG)
        body.pack(padx=22, pady=(10, 22))

        # track running subprocess per module so we can kill it on toggle
        self.processes: dict[str, subprocess.Popen | None] = {}
        self.buttons: dict[str, ModuleButton] = {}
        self._scripts: dict[str, str] = {}

        for name, script, icon in MODULES:
            self.processes[name] = None
            self._scripts[name] = script
            btn = ModuleButton(body, name, icon, on_click=lambda n=name: self.toggle(n))
            btn.pack(pady=4)
            self.buttons[name] = btn

        footer_divider = tk.Frame(root, bg=CARD_BORDER, height=1)
        footer_divider.pack(fill="x", padx=26, pady=(0, 8))

        footer = GitHubFooter(root, REPO_URL)
        footer.pack(anchor="w", padx=26, pady=(0, 16))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        # poll every 500ms to catch modules the user closed manually (window X button)
        self.root.after(500, self.poll_processes)

    def toggle(self, name: str):
        proc = self.processes[name]
        if proc is not None and proc.poll() is None:
            # running -> stop it
            proc.terminate()
            self.processes[name] = None
            self.buttons[name].set_running(False)
        else:
            # not running -> start it
            script = MODULES_DIR / self._scripts[name]
            new_proc = subprocess.Popen([sys.executable, str(script)])
            self.processes[name] = new_proc
            self.buttons[name].set_running(True)

    def poll_processes(self):
        # if a module window was closed by the user directly, reset its dot
        for name, proc in self.processes.items():
            if proc is not None and proc.poll() is not None:
                self.processes[name] = None
                self.buttons[name].set_running(False)
        self.root.after(500, self.poll_processes)

    def on_close(self):
        for proc in self.processes.values():
            if proc is not None and proc.poll() is None:
                proc.terminate()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = Launcher(root)
    root.mainloop()