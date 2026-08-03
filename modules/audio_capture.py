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

read_chunk() drains and concatenates every chunk currently queued (not
just one), so a render loop that's briefly slower than the audio
callback doesn't fall permanently behind -- see the module-level notes
in the previous revision for why taking only one chunk per call causes
a growing lag.

Two different "nothing to report" cases are handled differently:
  - No new chunk since the last call, but audio was flowing recently
    (the render loop just polled a bit faster than the callback fired,
    which is normal and happens on most frames): read_chunk() returns
    None. Callers should treat this as "nothing changed, skip this
    frame's audio update" -- NOT as silence. An earlier revision
    returned an explicit silent chunk here, which every polling module
    dutifully rolled into its history buffer; since this happens on a
    large fraction of frames even during normal playback, a block of
    real zeros kept getting woven into otherwise-live audio and showed
    up as a visible flat line sweeping across the display.
  - No new chunk for a sustained period (SILENCE_TIMEOUT_SEC), e.g.
    playback is actually paused and WASAPI has stopped delivering
    packets entirely: read_chunk() then returns real silence, so the
    visualization honestly goes quiet instead of freezing forever on
    the last frame that had signal.

Standalone test: run this file directly, it prints RMS level to
console every ~50ms so you can verify audio is being captured
before wiring up any visuals.
"""

import queue
import time

import numpy as np
import pyaudiowpatch as pyaudio

# how long to wait with no new audio before treating it as real silence
# (as opposed to the render loop just being a bit faster than the next
# callback) -- short enough that pausing playback still reads as silence
# quickly, long enough that normal poll-vs-callback timing jitter never
# triggers it
SILENCE_TIMEOUT_SEC = 0.15


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
        self._silence = None  # a single silent chunk, returned on sustained underrun
        self._last_data_time = time.monotonic()

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

    def read_chunk(self):
        """
        Returns a (N, channels) float32 array, range [-1, 1], or None.

        - New audio arrived since the last call: returns it (every
          queued chunk since the last call, concatenated -- see module
          docstring for why this matters for latency).
        - No new audio, but some arrived within SILENCE_TIMEOUT_SEC:
          returns None. Callers should skip updating on None rather
          than treat it as silence -- this is the common case of the
          render loop polling faster than audio arrives, not an actual
          gap in playback.
        - No new audio for longer than SILENCE_TIMEOUT_SEC (playback is
          genuinely paused/stopped): returns a real silent chunk, so
          the visualization settles to quiet instead of either freezing
          on stale data or leaving every caller to implement its own
          timeout.
        """
        chunks = []
        try:
            while True:
                chunks.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        if chunks:
            self._last_data_time = time.monotonic()
            if len(chunks) == 1:
                return chunks[0]
            return np.concatenate(chunks, axis=0)

        if time.monotonic() - self._last_data_time >= SILENCE_TIMEOUT_SEC:
            return self._silence
        return None

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()


if __name__ == "__main__":
    print("Opening loopback stream... play some audio now.")
    cap = AudioCapture(chunk_size=256)
    print(f"Capturing: {cap.channels} channels @ {cap.rate} Hz, chunk_size=256")
    print("Measuring read_chunk() return sizes for 5 seconds (None counted separately)...")
    print()

    sizes = []
    none_count = 0
    start = time.time()
    try:
        while time.time() - start < 5.0:
            chunk = cap.read_chunk()
            if chunk is None:
                none_count += 1
            else:
                sizes.append(len(chunk))
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()

    print(f"Calls returning data: {len(sizes)}  Calls returning None: {none_count}")
    if sizes:
        print(f"Chunk size -- Min: {min(sizes)}  Max: {max(sizes)}  Avg: {sum(sizes)/len(sizes):.1f}")