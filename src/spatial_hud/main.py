from __future__ import annotations

import argparse
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

    def start(self, use_mock: bool = False) -> None:
        hud_thread = HudLoop(self.state_queue)
        hud_thread.start()
        self._threads.append(hud_thread)

        if use_mock:
            feature_iterable = offline_feature_stream()
        else:
            self.capture.start()
            feature_iterable = feature_stream(self.capture.frames(), self.capture.samplerate)

        processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(feature_iterable,),
            daemon=True,
        )
        processing_thread.start()
        self._threads.append(processing_thread)

    def stop(self) -> None:
        self._running.clear()
        self.capture.stop()
        while not self.state_queue.empty():
            try:
                self.state_queue.get_nowait()
            except queue.Empty:
                break

    def join(self) -> None:
        for thread in self._threads:
            thread.join(timeout=2.0)

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

    def _publish_state(self) -> None:
        state = HudState(events=[evt for evt, _ in self.events])
        try:
            self.state_queue.put_nowait(state)
        except queue.Full:
            try:
                _ = self.state_queue.get_nowait()
            except queue.Empty:
                pass
            self.state_queue.put_nowait(state)

    def _expire_events(self) -> None:
        now = time.time() * 1000
        while self.events and now - self.events[0][1] > self.events[0][0].ttl_ms:
            self.events.popleft()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial HUD prototype")
    parser.add_argument("--mock", action="store_true", help="Use prerecorded mock features instead of live audio")
    return parser.parse_args()


def run_pipeline(use_mock: bool = False) -> None:
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
    run_pipeline(use_mock=args.mock)
