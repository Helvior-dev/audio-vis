# Audio Visualizer (MiniMeters-inspired Fan Project)

A real-time, GPU-accelerated audio visualization suite for Windows written in Python. Inspired by modern audio meters like MiniMeters, this project provides a set of modular visualizers for monitoring audio playback directly from system output (WASAPI Loopback).

> **Disclaimer & Project Purpose**
> This is a non-commercial, fan-made project created for educational purposes and personal audio monitoring. 
> It was built with the assistance of AI coding tools to experiment with Python DSP (Digital Signal Processing), PyOpenGL, and real-time audio capture.

---

## Features

- **System Audio Capture:** Captures native Windows audio using WASAPI Loopback (no virtual cables required).
- **Multi-Process Architecture:** Every visualizer window runs in its own independent process. If one window is resized, restarted, or closed, the rest keep running without latency drops.
- **GPU-Accelerated Rendering:** Smooth, high-FPS visuals built on PyOpenGL and GLFW.
- **Dark Windows Integration:** Custom dark title bars matching the UI design.
- **Modular Control Launcher:** A lightweight launcher GUI to toggle individual meters on and off.

---

## Modules Included

| Module | Description |
| :--- | :--- |
| **Loudness Meter** | Momentary (M), Short-Term (S), Integrated (INT), LRA, Peak levels, and stereo L/R meters adhering to EBU R128 / ITU-R BS.1770 concepts. |
| **Oscilloscope** | Real-time waveform display featuring correlation-based triggering for a stable trace even with complex musical content. |
| **Spectrum** | Fast Fourier Transform (FFT) spectrum analyzer with logarithmic frequency scaling and temporal smoothing. |
| **Spectrum Analyzer** | Detailed spectrum view with frequency markers, musical note detection, and interactive hover inspection. |
| **Vectorscope / Stereometer** | Mid/Side phase display, Lissajous stereo field visualizer, and correlation meters. |

---

## Requirements & Tech Stack

- **OS:** Windows 10 / 11 (WASAPI Loopback requirement)
- **Language:** Python 3.10+ (Tested up to Python 3.14)
- **Dependencies:**
  - `PyOpenGL` & `PyOpenGL_accelerate`
  - `glfw`
  - `numpy`
  - `pyaudiowpatch`
  - `tkinter` (Standard Python library)

---

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Helvior-dev/audio-vis.git
   cd audio-vis
   ```

2. **Install dependencies:**
   ```bash
   pip install PyOpenGL PyOpenGL_accelerate glfw numpy pyaudiowpatch
   ```

3. **Launch the app:**
   ```bash
   python main.py
   ```

---

## How It Works

- **Audio Pipeline:** `pyaudiowpatch` captures system output via WASAPI Loopback and broadcasts raw audio buffers to active modules.
- **Isolation:** Launcher spawns each visualizer module via `subprocess`.
- **Text & UI Rendering:** Custom bitmap text rendering pipeline optimized for raw OpenGL calls without heavy external GUI framework overhead.

---

## AI & Development Note

This repository was created using an iterative AI-assisted workflow. Coding assistants were used to prototype DSP algorithms, generate initial OpenGL shaders, and handle boilerplate code, while debugging, architecture design, profiling, and final refinements were done manually.

---

## License

MIT License. Feel free to fork, modify, and experiment!
