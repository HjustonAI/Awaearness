from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

from .audio_capture import LoopbackCapture
from .event_classifier import EventClassifier
from .hud import HudLoop
from .models import Event, HudState
from .signal_processing import feature_stream
from .simulation import offline_feature_stream


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    samplerate: int = 48_000
    blocksize: int = 1024
    history_ms: int = 1500


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.state_queue: "queue.Queue[HudState]" = queue.Queue(maxsize=8)
        self.capture = LoopbackCapture(
            samplerate=config.samplerate,
            blocksize=config.blocksize,
            channels=2,
        )
        self.classifier = EventClassifier()
        self.events: deque[tuple[Event, float]] = deque()
        self._threads: list[threading.Thread] = []
        self._running = threading.Event()
        self._running.set()
        self._hud_thread: HudLoop | None = None

    def start(self, use_mock: bool = False) -> None:
        logger.info("Starting pipeline (use_mock=%s)", use_mock)
        hud_thread = HudLoop(self.state_queue)
        hud_thread.start()
        self._threads.append(hud_thread)
        self._hud_thread = hud_thread

        if use_mock:
            logger.info("Using offline feature stream for mock mode")
            feature_iterable = offline_feature_stream()
        else:
            self.capture.start()
            feature_iterable = feature_stream(self.capture.frames(), self.capture.samplerate)
            logger.info(
                "Live capture initialized (samplerate=%s, blocksize=%s)",
                self.capture.samplerate,
                self.capture.blocksize,
            )

        processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(feature_iterable,),
            daemon=True,
        )
        processing_thread.start()
        self._threads.append(processing_thread)
        logger.debug("Processing thread started")

    def stop(self) -> None:
        logger.info("Stopping pipeline")
        self._running.clear()
        if self._hud_thread is not None:
            self._hud_thread.stop()
        self.capture.stop()
        while not self.state_queue.empty():
            try:
                self.state_queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("State queue drained")

    def join(self) -> None:
        for thread in self._threads:
            logger.debug("Joining thread %s", thread.name)
            thread.join(timeout=2.0)
        logger.info("Pipeline threads joined")
        self._hud_thread = None

    def _processing_loop(self, feature_iterable) -> None:
        for feature in feature_iterable:
            if not self._running.is_set():
                break
            event = self.classifier.classify(feature)
            if event.kind == "ambient":
                self._expire_events()
                self._publish_state()
                continue

            now = time.time() * 1000
            self.events.append((event, now))
            self._expire_events()
            self._publish_state()
            logger.debug(
                "Event detected: kind=%s azimuth=%.1f distance=%s confidence=%.2f", 
                event.kind,
                event.azimuth_deg,
                event.distance_bucket.value,
                event.confidence,
            )

    def _publish_state(self) -> None:
        state = HudState(events=[evt for evt, _ in self.events])
        logger.debug("Publishing HUD state with %s events", len(state.events))
        try:
            self.state_queue.put_nowait(state)
        except queue.Full:
            try:
                _ = self.state_queue.get_nowait()
            except queue.Empty:
                pass
            self.state_queue.put_nowait(state)
            logger.warning("State queue full; dropping oldest HUD state")

    def _expire_events(self) -> None:
        now = time.time() * 1000
        while self.events and now - self.events[0][1] > self.events[0][0].ttl_ms:
            expired = self.events.popleft()
            logger.debug(
                "Expiring event %s after %.0f ms",
                expired[0].kind,
                now - expired[1],
            )


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    invalid = False
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        invalid = True
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    if invalid:
        logging.getLogger(__name__).warning("Invalid log level '%s'; defaulting to INFO", level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial HUD prototype")
    parser.add_argument("--mock", action="store_true", help="Use prerecorded mock features instead of live audio")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def run_pipeline(use_mock: bool = False) -> None:
    if not logging.getLogger().hasHandlers():
        configure_logging("INFO")
    config = PipelineConfig()
    pipeline = Pipeline(config)
    try:
        pipeline.start(use_mock=use_mock)
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping pipeline...")
    finally:
        pipeline.stop()
        pipeline.join()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    run_pipeline(use_mock=args.mock)
