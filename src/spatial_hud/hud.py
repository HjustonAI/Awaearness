from __future__ import annotations

import logging
import math
import queue
import threading
from typing import Iterable

import pygame

from .models import Event, HudState

WINDOW_SIZE = (400, 400)
BACKGROUND_COLOR = (10, 10, 10, 0)
FOREGROUND_COLOR = (40, 200, 255)


logger = logging.getLogger(__name__)


class HudRenderer:
    def __init__(self, refresh_rate: float = 60.0) -> None:
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Spatial HUD")
        pygame.display.set_mode(WINDOW_SIZE, pygame.NOFRAME | pygame.SRCALPHA)
        self.screen = pygame.display.get_surface()
        self.clock = pygame.time.Clock()
        self.refresh_rate = refresh_rate
        self.running = False
        self._hwnd: int | None = None
        self._click_through = False
        self._dragging = False
        self._configure_window()
        logger.info("HUD renderer initialized (refresh_rate=%s)", self.refresh_rate)

    def draw_compass(self) -> None:
        radius = min(WINDOW_SIZE) // 2 - 20
        center = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)
        pygame.draw.circle(self.screen, FOREGROUND_COLOR, center, radius, width=2)
        for angle in range(0, 360, 45):
            radians = math.radians(angle)
            inner = (
                int(center[0] + math.cos(radians) * (radius - 10)),
                int(center[1] + math.sin(radians) * (radius - 10)),
            )
            outer = (
                int(center[0] + math.cos(radians) * radius),
                int(center[1] + math.sin(radians) * radius),
            )
            pygame.draw.line(self.screen, FOREGROUND_COLOR, inner, outer, width=1)

    def draw_event(self, event: Event) -> None:
        radius = min(WINDOW_SIZE) // 2 - 40
        center = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)
        angle = math.radians(event.azimuth_deg - 90)
        distance_scale = {"near": 0.6, "mid": 0.8, "far": 1.0}[event.distance_bucket.value]
        pos = (
            int(center[0] + math.cos(angle) * radius * distance_scale),
            int(center[1] + math.sin(angle) * radius * distance_scale),
        )
        color_map = {
            "footstep": (255, 200, 40),
            "vehicle": (40, 255, 120),
            "gunfire": (255, 80, 80),
        }
        color = color_map.get(event.kind, FOREGROUND_COLOR)
        size = max(8, int(12 * event.confidence))
        pygame.draw.circle(self.screen, color, pos, size)
        font = pygame.font.SysFont("Segoe UI", 16)
        label = font.render(event.kind.upper(), True, color)
        self.screen.blit(label, (pos[0] - label.get_width() // 2, pos[1] - label.get_height() // 2 - size - 4))

    def render(self, state: HudState) -> None:
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_compass()
        for event in state.events:
            self.draw_event(event)
        pygame.display.update()

    def _configure_window(self) -> None:
        try:
            import win32con
            import win32gui

            hwnd = pygame.display.get_wm_info()["window"]
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            style |= win32con.WS_EX_LAYERED
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)
            win32gui.SetLayeredWindowAttributes(hwnd, 0, 235, win32con.LWA_ALPHA)
            self._hwnd = hwnd
            logger.debug("HUD window configured with transparency")
        except Exception:
            # PyWin32 not available or running on non-Windows platforms.
            logger.debug("PyWin32 not available; skipping HUD transparency setup", exc_info=True)
            pass

    def _set_click_through(self, value: bool) -> None:
        if self._hwnd is None:
            logger.debug("Click-through requested but HWND not available")
            return
        try:
            import win32con
            import win32gui

            style = win32gui.GetWindowLong(self._hwnd, win32con.GWL_EXSTYLE)
            if value:
                style |= win32con.WS_EX_TRANSPARENT
            else:
                style &= ~win32con.WS_EX_TRANSPARENT
            win32gui.SetWindowLong(self._hwnd, win32con.GWL_EXSTYLE, style)
            logger.debug("Click-through style applied: %s", value)
        except Exception:
            logger.warning("Failed to update click-through style", exc_info=True)
            pass

    def _move_window(self, dx: int, dy: int) -> None:
        if self._hwnd is None:
            logger.debug("Window move requested but HWND not available")
            return
        try:
            import win32con
            import win32gui

            left, top, right, bottom = win32gui.GetWindowRect(self._hwnd)
            win32gui.SetWindowPos(
                self._hwnd,
                None,
                left + dx,
                top + dy,
                0,
                0,
                win32con.SWP_NOZORDER | win32con.SWP_NOSIZE,
            )
            logger.debug("Window position set to (%s, %s)", left + dx, top + dy)
        except Exception:
            logger.warning("Failed to reposition HUD window", exc_info=True)
            pass

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_t:
                self._click_through = not self._click_through
                self._set_click_through(self._click_through)
                logger.info("Click-through toggled to %s", self._click_through)
        if not self._click_through:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._dragging = False
            elif event.type == pygame.MOUSEMOTION and self._dragging:
                dx, dy = event.rel
                self._move_window(dx, dy)
                logger.debug("HUD moved by (%s, %s)", dx, dy)
        return True

    def run(self, state_stream: Iterable[HudState]) -> None:
        self.running = True
        stream_iter = iter(state_stream)
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                try:
                    state = next(stream_iter)
                except StopIteration:
                    self.running = False
                    break
                self.render(state)
                self.clock.tick(self.refresh_rate)
        finally:
            pygame.quit()


class HudLoop(threading.Thread):
    def __init__(self, state_queue: "queue.Queue[HudState]") -> None:
        super().__init__(daemon=True)
        self.state_queue = state_queue
        self._running = threading.Event()
        self._running.set()
        self.renderer: HudRenderer | None = None

    def run(self) -> None:
        self.renderer = HudRenderer()
        logger.info("HUD loop thread started")
        try:
            while self._running.is_set():
                for event in pygame.event.get():
                    if not self.renderer.handle_event(event):
                        self._running.clear()
                        break
                try:
                    state = self.state_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                self.renderer.render(state)
                self.renderer.clock.tick(self.renderer.refresh_rate)
        finally:
            pygame.quit()
            self.renderer = None
            logger.info("HUD loop thread exiting")

    def stop(self) -> None:
        self._running.clear()
        if self.renderer is not None:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            logger.debug("QUIT event posted to HUD loop")
