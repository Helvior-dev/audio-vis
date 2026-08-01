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

MODULES_DIR = Path(__file__).parent / "modules"

# display name -> (script filename, icon key)
MODULES = [
    ("Loudness",          "loudness.py",          "bars"),
    ("Waveform",           "waveform.py",          "wave"),
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