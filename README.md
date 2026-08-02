# Audio Visualizer

A real-time audio visualization application for Windows written in Python.

The project captures system audio using Windows WASAPI Loopback and renders multiple GPU-accelerated visualizations with OpenGL. Every visualizer runs as an independent process, allowing modules to be launched, restarted, or closed without affecting the rest of the application.

> **AI-assisted project**
>
> This project is developed using modern AI coding assistants.
>
> AI is used as a development tool for implementation, explanations, and rapid prototyping. All features, architecture, debugging, testing, profiling, and iterative improvements are directed by the project author. Every generated change is reviewed, tested, and refined before becoming part of the project.

---

# Features

* Real-time system audio capture (Windows WASAPI Loopback)
* Hardware-accelerated rendering using OpenGL
* Multiple independent visualization modules
* Separate process for every visualization window
* Native dark Windows title bars
* Resizable windows
* VSync rendering
* Low CPU usage
* Modular architecture

---

# Available Visualizers

## Loudness

Professional loudness and peak metering.

Features:

* Momentary (M)
* Short-Term (S)
* Integrated (INT)
* Loudness Range (LRA)
* Peak Level
* Stereo L/R level meters
* Smooth attack/release animation
* Peak hold indicators
* GPU-rendered gradients

---

## Oscilloscope

Real-time waveform display with correlation-based triggering.

Unlike a traditional zero-crossing oscilloscope, this implementation searches for the trigger position that best matches the previous frame, producing a much more stable display for complex music.

Features:

* Stable triggering
* Smooth rendering
* GPU acceleration
* VSync rendering

---

## Spectrum

FFT spectrum analyzer.

Features:

* Real-time FFT
* Logarithmic frequency bins
* dB scale
* Temporal smoothing
* GPU-rendered bars

---

## Spectrum Analyzer

Advanced spectrum visualization.

Features:

* Dense logarithmic FFT display
* Envelope tracking
* Frequency labels
* Musical note detection
* Hover inspection
* GPU-accelerated rendering

---

## Vectorscope

Stereo image visualization using a Lissajous display.

Features:

* Mid/Side visualization
* Stereo correlation meter
* Phase monitoring
* Stereo field analysis

---

# Technologies

* Python 3.14
* PyOpenGL
* GLFW
* NumPy
* PyAudioWPatch
* Tkinter

---

# Installation

```bash
git clone https://github.com/yourusername/audio-visualizer.git

cd audio-visualizer

pip install PyOpenGL PyOpenGL_accelerate glfw numpy pyaudiowpatch
```

---

# Running

```bash
python main.py
```

The launcher allows every visualization module to be started independently.

Each visualizer runs in its own process, so a crash or restart of one module does not affect the launcher or the remaining windows. This architecture also allows modules to be profiled and developed independently.

---

# Project Structure

```text
.
├── README.md
├── main.py
├── modules/
│   ├── audio_capture.py
│   ├── loudness.py
│   ├── oscilloscope.py
│   ├── spectrum.py
│   ├── spectrum_analyzer.py
│   ├── stereometer.py
│   ├── text_render.py
│   └── window_utils.py
```

---

# Development

This project follows an iterative AI-assisted development workflow.

Typical development cycle:

1. Design the feature.
2. Research the underlying DSP or rendering technique.
3. Generate an initial implementation using AI.
4. Review the generated code.
5. Debug and profile performance.
6. Improve architecture and rendering quality.
7. Repeat until the desired result is achieved.

The objective is not simply generating code, but understanding, validating, and refining every implementation.

---

# Technical Highlights

Current implementation includes:

* Windows WASAPI Loopback audio capture.
* Independent visualization processes.
* GPU rendering through raw OpenGL.
* Logarithmic FFT analysis.
* Correlation-based oscilloscope triggering.
* Loudness measurements inspired by ITU-R BS.1770 / EBU R128 concepts.
* Shared bitmap text rendering pipeline for OpenGL modules.
* Native Windows dark title bars for all windows.

---

# Platform

Supported operating systems:

* Windows 10
* Windows 11

System audio capture relies on Windows WASAPI Loopback and is therefore currently Windows-only.

---

# Future Plans

Potential future additions include:

* Recording support
* Frame export
* Theme customization
* Additional visualization modules
* Configuration system
* Performance overlay
* Cross-platform support where technically possible

---

# Acknowledgements

This project is developed with the assistance of modern AI coding tools.

Rather than replacing the development process, AI is used to accelerate implementation and experimentation while all architectural decisions, validation, debugging, testing, and iterative refinement remain under the author's control.

---

# License

Released under the MIT License.
