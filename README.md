# Audio Visualizer

A real-time audio visualization application written in **Python**, using **OpenGL**, **GLFW**, and **NumPy**.

## The project captures **system audio** through **Windows WASAPI Loopback** and renders multiple visualization windows in real time. Each visualizer runs in its own process, allowing them to be launched and closed independently without affecting the launcher.

## Features

* Real-time system audio capture (Windows WASAPI Loopback)
* GPU rendering with OpenGL
* Multiple independent visualization modules
* Separate process for every visualization
* Dark native Windows title bars
* Resizable windows
* VSync rendering for smooth animation

---

## Available Visualizers

### VU Meter

Classic horizontal VU meter featuring:

* Smooth attack/release animation
* Peak hold indicator
* dB scale
* GPU-generated color gradient
* 7-segment numeric labels

---

### Waveform

Displays raw audio samples as a scrolling waveform.

Features:

* Real-time waveform
* GPU rendering
* VSync rendering
* Low CPU usage

---

### Oscilloscope

A stabilized waveform display using correlation-based triggering.

Unlike a traditional zero-crossing oscilloscope, this implementation keeps complex music visually stable by selecting the trigger point that best matches the previous frame.

---

### Spectrum

Real-time FFT spectrum analyzer.

Features:

* FFT analysis
* Logarithmic frequency bins
* dB scale
* Temporal smoothing
* GPU-rendered bars

---

### Spectrogram

Scrolling time-frequency spectrogram.

Features:

* Continuous FFT history
* GPU texture rendering
* Logarithmic frequency scale
* Custom color gradient

---

### Stereometer

Stereo image visualization using a Lissajous display.

Includes:

* Mid/Side visualization
* Stereo correlation meter
* Phase monitoring

---

## Technologies

* Python 3.14
* PyOpenGL
* GLFW
* NumPy
* pyaudiowpatch
* Tkinter

---

## Installation

```bash
git clone https://github.com/yourusername/audio-visualizer.git

cd audio-visualizer

pip install PyOpenGL PyOpenGL_accelerate glfw numpy pyaudiowpatch
```

---

## Running

```bash
python main.py
```

The launcher allows every visualization to be started or stopped independently. Each module runs in its own process, so closing one window does not affect the others.

---

## Project Structure

```text
.
├── README.md
├── main.py
├── modules/
│   ├── audio_capture.py
│   ├── oscilloscope.py
│   ├── spectrogram.py
│   ├── spectrum.py
│   ├── stereometer.py
│   ├── vu.py
│   ├── waveform.py
│   └── window_utils.py
```

---

## Platform

Currently supported:

* Windows 10
* Windows 11

System audio capture relies on **WASAPI Loopback**, making Windows the only supported platform at the moment.

---

## License

This project is available under the MIT License.
