"""
System audio loopback capture (Windows WASAPI via pyaudiowpatch).

This captures what's actually playing on your speakers/headphones
(loopback), not microphone input. Each visualization module will
import AudioCapture and read the latest samples from it.

Uses PyAudio's callback mode instead of blocking stream.read(). In
blocking mode, read_chunk() calls stream.read() directly on the GUI
thread -- if WASAPI stops delivering packets (e.g. the output device
goes fully idle with nothing playing, which some drivers do instead of
streaming silence), that call can block indefinitely and the whole
window freezes ("Not Responding"). In callback mode, PortAudio runs
audio I/O on its own thread and pushes chunks into a buffer; read_chunk()
just drains that buffer and returns immediately (with zeros if nothing
new has arrived yet), so the render loop never stalls waiting on audio.

Standalone test: run this file directly, it prints RMS level to
console every ~50ms so you can verify audio is being captured
before wiring up any visuals.
"""

import queue
import threading

import numpy as np
import pyaudiowpatch as pyaudio


class AudioCapture:
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.channels = 2
        self.rate = 44100

        # bounded queue of raw chunks from the callback thread; bounded
        # so a stalled consumer can't leak memory if it ever falls behind
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._last_chunk = None  # returned when the queue is empty (silence/underrun)

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
        self._last_chunk = np.zeros((self.chunk_size, self.channels), dtype=np.float32)

        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            frames_per_buffer=self.chunk_size,
            input=True,
            input_device_index=default_speakers["index"],
            stream_callback=self._on_audio,
        )
        self.stream.start_stream()

    def _on_audio(self, in_data, frame_count, time_info, status):
        # runs on PortAudio's own thread, never the GUI thread -- must
        # not block here either, so the queue is fire-and-forget: if
        # it's full (consumer stalled), drop the oldest chunk rather
        # than blocking the audio thread itself
        data = np.frombuffer(in_data, dtype=np.float32).reshape(-1, self.channels)
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                pass
        return (None, pyaudio.paContinue)

    def read_chunk(self) -> np.ndarray:
        """
        Returns shape (chunk_size, channels) float32 array, range [-1, 1].

        Non-blocking: if no new chunk has arrived since the last call
        (e.g. the output device is fully idle and WASAPI isn't
        delivering packets), returns the last chunk received instead of
        blocking the caller -- callers already treat near-silent input
        the same as silence, so repeating the last real chunk for a
        frame or two is visually indistinguishable from underrun and,
        critically, never freezes the render loop.
        """
        try:
            chunk = self._queue.get_nowait()
            self._last_chunk = chunk
            return chunk
        except queue.Empty:
            return self._last_chunk

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
            time.sleep(0.005)  # callback mode doesn't pace read_chunk() itself anymore
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()

    times_ms = [t * 1000 for t in times]
    print(f"Total calls in 5s: {len(times_ms)}")
    print(f"Min: {min(times_ms):.3f}ms  Max: {max(times_ms):.3f}ms  Avg: {sum(times_ms)/len(times_ms):.3f}ms")
    print(f"Calls over 20ms: {sum(1 for t in times_ms if t > 20)}")
    print(f"Calls over 100ms: {sum(1 for t in times_ms if t > 100)}")