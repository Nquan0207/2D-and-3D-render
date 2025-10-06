from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import glfw
import numpy as np

from .libs.transform import Trackball


@dataclass
class MouseState:
    pos: tuple[float, float] = (0.0, 0.0)
    left_pressed: bool = False
    right_pressed: bool = False
    drag_blocked: bool = False


class TrackballController:
    def __init__(self, *, distance: float = 4.0) -> None:
        self._distance = max(distance, 0.001)
        self.trackball = Trackball(distance=self._distance)
        self.mouse = MouseState()
        self.window_size: Tuple[int, int] = (1, 1)

    def resize(self, width: int, height: int) -> None:
        self.window_size = (max(width, 1), max(height, 1))

    def view_matrix(self) -> np.ndarray:
        return self.trackball.view_matrix()

    def camera_position(self) -> np.ndarray:
        view = self.view_matrix().astype(np.float32)
        inv = np.linalg.inv(view)
        pos = inv @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return pos[:3]

    def projection_matrix(self, winsize: Tuple[int, int]) -> np.ndarray:
        return self.trackball.projection_matrix(winsize)

    def on_cursor(self, xpos: float, ypos: float) -> None:
        old = self.mouse.pos
        inv_y = self.window_size[1] - ypos
        self.mouse.pos = (xpos, inv_y)
        if self.mouse.left_pressed and not self.mouse.drag_blocked:
            self.trackball.drag(old, self.mouse.pos, self.window_size)
        if self.mouse.right_pressed and not self.mouse.drag_blocked:
            self.trackball.pan(old, self.mouse.pos)

    def on_scroll(self, dy: float) -> None:
        self.trackball.zoom(dy, max(self.window_size[1], 1))

    def on_mouse_button(self, button: int, pressed: bool) -> None:
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse.left_pressed = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse.right_pressed = pressed
        if not pressed:
            self.mouse.drag_blocked = False

    def block_drag_this_frame(self) -> None:
        self.mouse.drag_blocked = True

    def reset(self, *, distance: float | None = None) -> None:
        if distance is not None:
            self._distance = max(distance, 0.001)
        self.trackball = Trackball(distance=self._distance)
        self.mouse = MouseState()
