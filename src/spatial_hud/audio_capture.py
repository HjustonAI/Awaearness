from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

try:
    import soundcard as sc
except ImportError:  # pragma: no cover - optional dependency for live capture
    sc = None
else:  # pragma: no cover - runtime compatibility patch for NumPy>=2
    try:
        import soundcard.mediafoundation as _sc_mf

        _sc_mf.numpy.fromstring = np.frombuffer  # type: ignore[attr-defined]
    except Exception:
        pass

if TYPE_CHECKING:  # pragma: no cover - typing only
    from soundcard import Recorder
else:
    Recorder = Any


logger = logging.getLogger(__name__)


class LoopbackCapture:
    """Capture system playback audio via WASAPI loopback."""

    def __init__(
        self,
        samplerate: int = 48_000,
        channels: int = 2,
        blocksize: int = 1024,
        ring_size: int = 32,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=ring_size)
        self._lock = threading.Lock()
        self._recorder: Optional[Recorder] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

    def __enter__(self) -> "LoopbackCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            logger.debug("LoopbackCapture.start called but thread already running")
            return
        if sc is None:
            raise RuntimeError(
                "soundcard package is required for loopback capture. Install it with 'pip install soundcard' or run with --mock."
            )

        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError("No default speaker found for loopback capture.")
        logger.info("Selected default speaker '%s' for loopback capture", speaker.name)

        microphone = sc.get_microphone(speaker.name, include_loopback=True)
        if microphone is None:
            raise RuntimeError(
                "Loopback device not available. Enable 'Stereo Mix' or check audio driver settings."
            )
        logger.debug("Using loopback endpoint '%s'", microphone.name)

        self._recorder = microphone.recorder(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=self.channels,
        )
        self._recorder.__enter__()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()
        logger.info(
            "Loopback capture thread started (samplerate=%s, blocksize=%s, channels=%s)",
            self.samplerate,
            self.blocksize,
            self.channels,
        )

    def stop(self) -> None:
        with self._lock:
            if self._thread is None:
                logger.debug("LoopbackCapture.stop called but no thread running")
                return
            assert self._stop_event is not None
            self._stop_event.set()
            self._thread.join(timeout=1.5)
            self._thread = None
            self._stop_event = None
            if self._recorder is not None:
                self._recorder.__exit__(None, None, None)
                self._recorder = None
            with self._queue.mutex:
                self._queue.queue.clear()
            logger.info("Loopback capture stopped and buffers cleared")

    def frames(self) -> Iterator[np.ndarray]:
        """Yield PCM frames as numpy arrays shaped (blocksize, channels)."""
        while True:
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                stop_event = self._stop_event
                if self._thread is None or (stop_event is not None and stop_event.is_set()):
                    break
                continue
            yield frame

    def _capture_worker(self) -> None:
        assert self._recorder is not None
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                data = self._recorder.record(self.blocksize)
            except Exception as exc:  # pragma: no cover - hardware failure
                logger.exception("Recorder error while capturing audio: %s", exc)
                break
            if data is None:
                continue
            if data.shape[1] > self.channels:
                frame = data[:, : self.channels]
            elif data.shape[1] < self.channels:
                pad = self.channels - data.shape[1]
                frame = np.pad(data, ((0, 0), (0, pad)))
            else:
                frame = data
            frame = frame.astype(np.float32, copy=False)
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    _ = self._queue.get_nowait()
                    logger.warning("Loopback capture queue full; dropping oldest frame")
                except queue.Empty:
                    logger.warning("Loopback capture queue full but empty on readback; continuing")
                self._queue.put_nowait(frame)
        logger.debug("Capture thread exiting")
