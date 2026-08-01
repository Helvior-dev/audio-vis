"""
System audio loopback capture (Windows WASAPI via pyaudiowpatch).

This captures what's actually playing on your speakers/headphones
(loopback), not microphone input. Each visualization module will
import AudioCapture and read the latest samples from it.

Standalone test: run this file directly, it prints RMS level to
console every ~50ms so you can verify audio is being captured
before wiring up any visuals.
"""

import numpy as np
import pyaudiowpatch as pyaudio


class AudioCapture:
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.channels = 2
        self.rate = 44100
        self._open_loopback_stream()

    def _open_loopback_stream(self):
        # find default WASAPI speaker device, then its loopback counterpart
        try:
            wasapi_info = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            raise RuntimeError(
                "WASAPI not available. pyaudiowpatch requires Windows."
            )

        default_speakers = self.pa.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        if not default_speakers.get("isLoopbackDevice", False):
            # need to find the loopback-flagged version of this device
            for loopback in self.pa.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                raise RuntimeError(
                    f"Could not find loopback device for: {default_speakers['name']}"
                )

        self.channels = int(default_speakers["maxInputChannels"])
        self.rate = int(default_speakers["defaultSampleRate"])

        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            frames_per_buffer=self.chunk_size,
            input=True,
            input_device_index=default_speakers["index"],
        )

    def read_chunk(self) -> np.ndarray:
        """Returns shape (chunk_size, channels) float32 array, range [-1, 1]."""
        raw = self.stream.read(self.chunk_size, exception_on_overflow=False)
        data = np.frombuffer(raw, dtype=np.float32)
        return data.reshape(-1, self.channels)

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()


if __name__ == "__main__":
    import time

    print("Opening loopback stream... play some audio now.")
    cap = AudioCapture(chunk_size=256)
    print(f"Capturing: {cap.channels} channels @ {cap.rate} Hz, chunk_size=256")
    print("Measuring real read_chunk() timing for 5 seconds...")
    print("(expected ~5.3ms per call at 48000Hz if not blocking excessively)")
    print()

    times = []
    start = time.time()
    try:
        while time.time() - start < 5.0:
            t0 = time.perf_counter()
            chunk = cap.read_chunk()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()

    times_ms = [t * 1000 for t in times]
    print(f"Total calls in 5s: {len(times_ms)}")
    print(f"Expected calls at 5.3ms/call: ~{int(5000/5.3)}")
    print(f"Min: {min(times_ms):.2f}ms  Max: {max(times_ms):.2f}ms  Avg: {sum(times_ms)/len(times_ms):.2f}ms")
    print(f"Calls over 20ms (2x expected): {sum(1 for t in times_ms if t > 20)}")
    print(f"Calls over 100ms (very slow): {sum(1 for t in times_ms if t > 100)}")