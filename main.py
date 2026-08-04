"""
Launcher window: one button per visualization module.
Each module runs as a separate process (not thread) so a crash/freeze
in one visualizer (e.g. Spectrum Analyzer) never takes down the others
or the launcher itself.

Styled to match the dark, card-based look of the visualizer windows
themselves, instead of stock Tk widgets.
"""

import subprocess
import sys
import tkinter as tk
from pathlib import Path

from modules.window_utils import apply_dark_titlebar_tk
from modules.audio_capture import AudioSource, list_audio_sources
from modules.audio_source_config import save_selected_source, load_selected_source

MODULES_DIR = Path(__file__).parent / "modules"

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