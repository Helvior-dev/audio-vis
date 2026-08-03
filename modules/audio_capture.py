"""
System audio loopback capture (Windows WASAPI via pyaudiowpatch).

This captures what's actually playing on your speakers/headphones
(loopback), not microphone input. Each visualization module will
import AudioCapture and read the latest samples from it.

Uses PyAudio's callback mode instead of blocking stream.read(). In
blocking mode, read_chunk() calls stream.read() directly on the GUI
thread -- if WASAPI stops delivering packets (e.g. the output device
goes fully idle, which happens on pause), that call can block
indefinitely and the whole window freezes ("Not Responding"). In
callback mode, PortAudio runs audio I/O on its own thread and pushes
chunks into a queue; read_chunk() drains that queue and returns
immediately, so the render loop never stalls waiting on audio.

read_chunk() drains and concatenates *every* chunk currently queued,
not just one. The audio callback fires roughly every chunk_size/rate
seconds (a few ms), which is faster than most render loops poll --
taking only one queued chunk per read_chunk() call means the queue
keeps growing behind the reader, and every module ends up visualizing
audio that's increasingly behind what's actually playing (the backlog
was audible as a growing lag, worse the longer playback ran). Draining
the whole queue each call keeps the reader caught up to the writer, so
there's no accumulating delay -- callers already handle variable-length
chunks (they index by len(chunk), not a fixed chunk_size).

When nothing is queued (e.g. playback is paused and WASAPI isn't
delivering new packets), read_chunk() returns actual silence instead of
replaying the last real chunk -- repeating old audio would freeze the
visualization on a stale frame instead of honestly showing "nothing is
playing right now".

Standalone test: run this file directly, it prints RMS level to
console every ~50ms so you can verify audio is being captured
before wiring up any visuals.
"""

import queue

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
        # so a stalled consumer can't leak memory, though normal use
        # drains it every read_chunk() call so it rarely holds more
        # than one or two chunks at a time
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._silence = None  # a single silent chunk, returned on underrun

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
        self._silence = np.zeros((self.chunk_size, self.channels), dtype=np.float32)

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
        Returns shape (N, channels) float32 array, range [-1, 1]. N is
        normally close to chunk_size but not guaranteed to be exactly
        chunk_size -- it's every sample that arrived since the last call,
        concatenated, so callers (which already index by len(chunk))
        stay in sync with real time instead of falling behind a growing
        queue.

        Non-blocking: if nothing has arrived since the last call (e.g.
        playback is paused and WASAPI isn't delivering packets), returns
        a chunk of silence rather than either blocking or replaying old
        audio -- so the visualization honestly goes quiet instead of
        freezing on the last frame that had real signal.
        """
        chunks = []
        try:
            while True:
                chunks.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        if not chunks:
            return self._silence
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=0)

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
    print("Measuring real read_chunk() timing and returned chunk sizes for 5 seconds...")
    print()

    times = []
    sizes = []
    start = time.time()
    try:
        while time.time() - start < 5.0:
            t0 = time.perf_counter()
            chunk = cap.read_chunk()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            sizes.append(len(chunk))
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()

    times_ms = [t * 1000 for t in times]
    print(f"Total calls in 5s: {len(times_ms)}")
    print(f"Call time -- Min: {min(times_ms):.3f}ms  Max: {max(times_ms):.3f}ms  Avg: {sum(times_ms)/len(times_ms):.3f}ms")
    print(f"Chunk size -- Min: {min(sizes)}  Max: {max(sizes)}  Avg: {sum(sizes)/len(sizes):.1f}")