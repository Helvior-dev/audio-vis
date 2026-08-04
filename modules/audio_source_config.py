"""
Shared audio-source selection, persisted to a small JSON file next to
the other modules so every visualizer process (each one is its own
subprocess -- see main.py) can agree on "what should I be capturing".

Two things live here:
  - save_selected_source() / load_selected_source(): read/write the
    JSON file. Used by the launcher when the user picks something from
    the dropdown, and by each visualizer window at startup.
  - SourceWatcher: a tiny poll-based helper a visualizer's render loop
    calls once per frame. It stats the config file's mtime (a cheap
    filesystem call, safe to do every frame at 60-144Hz) and only
    re-reads/parses the JSON when the mtime actually changed, so an
    already-open window picks up a source change made from the
    launcher without needing to restart -- this is the "apply live to
    already-open windows" behavior.

No file locking: writes are a single atomic os.replace() of a fully-
formed temp file, so a reader never sees a half-written file, and the
worst case on a genuinely concurrent write/read is the reader getting
the old or the new complete value, never a torn one.
"""

import json
import os
import tempfile
from pathlib import Path

try:
    # when modules/ is imported as a package (e.g. main.py does
    # `from modules.audio_source_config import ...`), relative import
    # is required -- a bare `from audio_capture import AudioSource`
    # fails here because modules/ itself is never put on sys.path in
    # that case, only its parent directory is
    from .audio_capture import AudioSource
except ImportError:
    # when this file's own directory has been put on sys.path directly
    # (each visualizer script does sys.path.insert(0, its own folder)
    # so it can also be run standalone, not just via the launcher),
    # there is no enclosing package for a relative import to resolve
    # against, so fall back to the bare/absolute form
    from audio_capture import AudioSource

CONFIG_PATH = Path(__file__).parent / "audio_source.json"


def save_selected_source(source: AudioSource) -> None:
    """Atomically writes the selected source to the shared config file."""
    data = source.as_dict()
    fd, tmp_path = tempfile.mkstemp(
        dir=str(CONFIG_PATH.parent), prefix=".audio_source_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, CONFIG_PATH)
    except OSError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_selected_source() -> AudioSource:
    """
    Reads the shared config file, or returns the default source
    (default system audio loopback) if the file doesn't exist yet or
    is unreadable/corrupt -- a module should never fail to start just
    because the config file is missing or briefly mid-write.
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AudioSource.from_dict(data)
    except (OSError, json.JSONDecodeError, ValueError):
        return AudioSource()


class SourceWatcher:
    """
    Call check() once per frame from a visualizer's render loop. Returns
    the new AudioSource if the config file changed since the last check
    (or since construction, for the first call), otherwise None.

    Usage in a module's _update_audio():
        new_source = self.source_watcher.check()
        if new_source is not None:
            self.audio.reopen(new_source)
    """

    def __init__(self):
        self._last_mtime = self._stat_mtime()

    def _stat_mtime(self):
        try:
            return CONFIG_PATH.stat().st_mtime_ns
        except OSError:
            return None

    def check(self) -> AudioSource | None:
        mtime = self._stat_mtime()
        if mtime == self._last_mtime:
            return None
        self._last_mtime = mtime
        return load_selected_source()