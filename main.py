"""
Launcher window: one button per visualization module.
Each module runs as a separate process (not thread) so a crash/freeze
in one visualizer (e.g. Spectrogram) never takes down the others or
the launcher itself.
"""

import subprocess
import sys
import tkinter as tk
from pathlib import Path

MODULES_DIR = Path(__file__).parent / "modules"

# name shown on button -> script filename
MODULES = {
    "VU Meter": "vu.py",
    "Waveform": "waveform.py",
    "Spectrum": "spectrum.py",
    "Spectrogram": "spectrogram.py",
    "Stereometer": "stereometer.py",
    "Spectrum Analyzer": "spectrum_analyzer.py",
}


class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Audio Visualizer")
        self.root.configure(bg="#111111")
        self.root.resizable(False, False)

        # track running subprocess per module so we can kill it on toggle
        self.processes: dict[str, subprocess.Popen | None] = {
            name: None for name in MODULES
        }
        self.buttons: dict[str, tk.Button] = {}

        for name in MODULES:
            btn = tk.Button(
                self.root,
                text=name,
                width=20,
                height=2,
                bg="#222222",
                fg="#dddddd",
                activebackground="#333333",
                relief="flat",
                command=lambda n=name: self.toggle(n),
            )
            btn.pack(padx=10, pady=5)
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
            self.buttons[name].configure(bg="#222222")
        else:
            # not running -> start it
            script = MODULES_DIR / MODULES[name]
            new_proc = subprocess.Popen([sys.executable, str(script)])
            self.processes[name] = new_proc
            self.buttons[name].configure(bg="#2e7d32")

    def poll_processes(self):
        # if a module window was closed by the user directly, reset its button color
        for name, proc in self.processes.items():
            if proc is not None and proc.poll() is not None:
                self.processes[name] = None
                self.buttons[name].configure(bg="#222222")
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