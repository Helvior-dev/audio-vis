"""
Audio capture (Windows WASAPI via pyaudiowpatch), with a selectable
source: either system loopback (what's playing on an output device) or
a real microphone/line-in.

Each visualization module imports AudioCapture and reads the latest
samples from it. The active source can be changed at any time via
reopen(source) -- this tears down and recreates the underlying
PortAudio stream in place, without needing a new AudioCapture instance,
which is what lets already-open visualizer windows pick up a source
change made from the launcher without restarting.

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
from dataclasses import dataclass

import numpy as np
import pyaudiowpatch as pyaudio

# how long to wait with no new audio before treating it as real silence
# (as opposed to the render loop just being a bit faster than the next
# callback) -- short enough that pausing playback still reads as silence
# quickly, long enough that normal poll-vs-callback timing jitter never
# triggers it
SILENCE_TIMEOUT_SEC = 0.15

KIND_LOOPBACK = "loopback"     # system audio: capture what an output device is playing
KIND_MICROPHONE = "microphone"  # a real input device (mic, line-in)


@dataclass(frozen=True)
class AudioSource:
    """
    Identifies where to capture from. `kind` picks loopback vs
    microphone; `device_index` is a specific PyAudio device index, or
    None to mean "whatever WASAPI currently considers the default for
    that kind" (so e.g. "default microphone" keeps tracking the OS
    default even if the user changes it in Windows settings later).

    `name` is display-only, carried along so callers/config files don't
    need a separate device lookup just to show what's selected.
    """
    kind: str = KIND_LOOPBACK
    device_index: int | None = None
    name: str = "Default system audio"

    def as_dict(self) -> dict:
        return {"kind": self.kind, "device_index": self.device_index, "name": self.name}

    @staticmethod
    def from_dict(d: dict) -> "AudioSource":
        return AudioSource(
            kind=d.get("kind", KIND_LOOPBACK),
            device_index=d.get("device_index"),
            name=d.get("name", "Default system audio"),
        )


def list_audio_sources() -> tuple[list[AudioSource], list[AudioSource]]:
    """
    Enumerates available sources. Returns (loopback_sources, microphone_sources).

    loopback_sources: one entry per output device that has (or can be
    mapped to) a WASAPI loopback counterpart -- selecting one captures
    whatever that device is currently playing.

    microphone_sources: one entry per WASAPI input device that is NOT a
    loopback device -- i.e. an actual microphone / line-in.

    Each list's first entry is a "Default ..." pseudo-source
    (device_index=None) that always tracks the OS default at open time,
    so it stays correct if the user changes their default device later
    without needing to reselect anything here.
    """
    pa = pyaudio.PyAudio()
    try:
        loopback_sources = [AudioSource(KIND_LOOPBACK, None, "Default system audio")]
        mic_sources = [AudioSource(KIND_MICROPHONE, None, "Default microphone")]

        seen_loopback_names = set()
        for dev in pa.get_loopback_device_info_generator():
            name = dev["name"]
            if name in seen_loopback_names:
                continue
            seen_loopback_names.add(name)
            loopback_sources.append(AudioSource(KIND_LOOPBACK, dev["index"], name))

        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            wasapi_info = None

        if wasapi_info is not None:
            for i in range(pa.get_device_count()):
                dev = pa.get_device_info_by_index(i)
                if dev.get("hostApi") != wasapi_info["index"]:
                    continue
                if dev.get("isLoopbackDevice", False):
                    continue
                if int(dev.get("maxInputChannels", 0)) <= 0:
                    continue
                mic_sources.append(AudioSource(KIND_MICROPHONE, dev["index"], dev["name"]))

        return loopback_sources, mic_sources
    finally:
        pa.terminate()


class AudioCapture:
    def __init__(self, chunk_size: int = 1024, source: AudioSource | None = None):
        self.chunk_size = chunk_size
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.channels = 2
        self.rate = 44100
        self.source = source or AudioSource()
        self.last_error = None

        # bounded queue of raw chunks from the callback thread; bounded
        # so a stalled consumer can't leak memory, though normal use
        # drains it every read_chunk() call so it rarely holds more
        # than one or two chunks at a time
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._silence = None  # a single silent chunk, returned on sustained underrun
        self._last_data_time = time.monotonic()

        self._open_stream(self.source)

    def _resolve_device(self, source: AudioSource) -> dict:
        """Returns the PyAudio device-info dict to open for the given source."""
        if source.device_index is not None:
            return self.pa.get_device_info_by_index(source.device_index)

        try:
            wasapi_info = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            raise RuntimeError(
                "WASAPI not available. pyaudiowpatch requires Windows."
            )

        if source.kind == KIND_MICROPHONE:
            return self.pa.get_device_info_by_index(wasapi_info["defaultInputDevice"])

        # default loopback: default *output* device, mapped to its loopback twin
        default_speakers = self.pa.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )
        if not default_speakers.get("isLoopbackDevice", False):
            for loopback in self.pa.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    return loopback
            raise RuntimeError(
                f"Could not find loopback device for: {default_speakers['name']}"
            )
        return default_speakers

    def _open_stream(self, source: AudioSource):
        device = self._resolve_device(source)

        # a loopback capture device already reports maxInputChannels for
        # what it can hand back (usually 2); a real input device (mic)
        # reports its own channel count the same way -- either way this
        # is "how many channels will frames_per_buffer worth of data
        # contain", so no branching needed between the two kinds here
        self.channels = max(1, int(device.get("maxInputChannels", 2)))
        self.rate = int(device["defaultSampleRate"])
        self._silence = np.zeros((self.chunk_size, self.channels), dtype=np.float32)

        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            frames_per_buffer=self.chunk_size,
            input=True,
            input_device_index=device["index"],
            stream_callback=self._on_audio,
        )
        self.stream.start_stream()

    def reopen(self, source: AudioSource):
        """
        Switches the capture to a new source in place. Safe to call at
        any time from the render loop (not from the audio callback
        thread) -- e.g. every frame after checking whether the user's
        selection changed. On failure, the previous stream stays closed
        and reading returns silence rather than crashing the
        visualizer; last_error is set so callers can surface it if they
        want to.
        """
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except OSError:
                pass
            self.stream = None

        # drop anything queued from the old source so the next
        # read_chunk() doesn't hand back a mix of old-source and
        # new-source audio
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        self.last_error = None
        try:
            self._open_stream(source)
            self.source = source
        except (RuntimeError, OSError) as exc:
            self.last_error = str(exc)
            self.stream = None
        self._last_data_time = time.monotonic()

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
          genuinely paused/stopped, or reopen() failed and there is no
          stream): returns a real silent chunk, so the visualization
          settles to quiet instead of either freezing on stale data or
          leaving every caller to implement its own timeout.
        """
        if self.stream is None:
            return self._silence

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
    loopback_sources, mic_sources = list_audio_sources()
    print("Loopback (system audio) sources:")
    for s in loopback_sources:
        print(f"  [{s.device_index}] {s.name}")
    print("Microphone sources:")
    for s in mic_sources:
        print(f"  [{s.device_index}] {s.name}")
    print()

    print("Opening default loopback stream... play some audio now.")
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