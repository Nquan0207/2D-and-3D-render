from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import OpenGL.GL as GL

from .libs import transform as T

from .threeD.axes import LineAxes
from .mouse import TrackballController
from .rendering import RenderMode


@dataclass
class TransformState:
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation_deg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    scale: float = 1.0


class BaseScene3D:
    def __init__(self, renderer, *, axes_scale: float = 1.2) -> None:
        self.renderer = renderer
        self.transform = TransformState()
        self.mode: RenderMode = RenderMode.GOURAUD

        self.projection = T.perspective(45.0, 1.0, 0.1, 100.0)
        self.camera = TrackballController(distance=4.0)
        self.axes = LineAxes()
        self.show_axes = True
        self.axes_scale = axes_scale

        self.flat_color = np.array([0.9, 0.5, 0.3], dtype=np.float32)
        self.view_pos = self.camera.camera_position()

    # ------------------------------------------------------------------ basic properties
    def get_translation(self) -> np.ndarray:
        return self.transform.translation.copy()

    def set_translation(self, values: list[float]) -> None:
        self.transform.translation = np.array(values, dtype=np.float32)

    def get_rotation(self) -> np.ndarray:
        return self.transform.rotation_deg.copy()

    def set_rotation(self, values: list[float]) -> None:
        self.transform.rotation_deg = np.array(values, dtype=np.float32)

    def get_scale(self) -> float:
        return float(self.transform.scale)

    def set_scale(self, value: float) -> None:
        self.transform.scale = max(0.1, float(value))

    def reset_transform(self) -> None:
        self.transform = TransformState()

    def reset_view(self) -> None:
        self.camera.reset(distance=4.0)
        self.view_pos = self.camera.camera_position()

    def resize(self, width: int, height: int) -> None:
        self.camera.resize(width, height)

    def set_axes_settings(
        self,
        *,
        show: bool | None = None,
        scale: float | None = None,
    ) -> None:
        if show is not None:
            self.show_axes = bool(show)
        if scale is not None:
            self.axes_scale = max(0.1, float(scale))

    def set_flat_color(self, color: np.ndarray) -> None:
        self.flat_color = color.astype(np.float32)

    def get_flat_color(self) -> np.ndarray:
        return self.flat_color

    # ------------------------------------------------------------------ rendering helpers
    def set_mode(self, mode: RenderMode) -> None:
        self.mode = mode

    def on_window_resize(
        self,
        window_width: int,
        window_height: int,
        *,
        framebuffer_width: Optional[int] = None,
        framebuffer_height: Optional[int] = None,
    ) -> None:
        if framebuffer_width is None:
            framebuffer_width = window_width
        if framebuffer_height is None:
            framebuffer_height = window_height
        aspect = max(framebuffer_width / float(max(framebuffer_height, 1)), 1e-4)
        self.projection = T.perspective(45.0, aspect, 0.1, 100.0)
        self.camera.resize(window_width, window_height)

    def draw(self) -> None:
        GL.glEnable(GL.GL_DEPTH_TEST)
        projection = self.projection.astype(np.float32)
        view = self.camera.view_matrix().astype(np.float32)
        self.view_pos = self.camera.camera_position()
        self._draw_object(projection, view)
        GL.glDisable(GL.GL_DEPTH_TEST)
        if self.show_axes:
            axes_model = self._axes_model_matrix()
            self.axes.draw(projection, view, axes_model, line_width=2.0)

    # ------------------------------------------------------------------ utilities for subclasses
    def _axes_model_matrix(self) -> np.ndarray:
        return (
            T.scale(self.axes_scale)
        ).astype(np.float32)

    def _model_matrix(self) -> np.ndarray:
        translate = T.translate(*self.transform.translation)
        rot_x = T.rotate((1.0, 0.0, 0.0), float(self.transform.rotation_deg[0]))
        rot_y = T.rotate((0.0, 1.0, 0.0), float(self.transform.rotation_deg[1]))
        rot_z = T.rotate((0.0, 0.0, 1.0), float(self.transform.rotation_deg[2]))
        scale_mat = T.scale(self.transform.scale)
        return (translate @ rot_z @ rot_y @ rot_x @ scale_mat).astype(np.float32)

    # ------------------------------------------------------------------ input handling
    def on_cursor(self, xpos: float, ypos: float) -> None:
        self.camera.on_cursor(xpos, ypos)

    def on_scroll(self, dy: float) -> None:
        self.camera.on_scroll(dy)

    def on_mouse_button(self, button: int, pressed: bool) -> None:
        self.camera.on_mouse_button(button, pressed)

    def block_drag_this_frame(self) -> None:
        self.camera.block_drag_this_frame()

    # ------------------------------------------------------------------ common transform helpers
    def translate(self, dx: float, dy: float, dz: float = 0.0) -> None:
        delta = np.array([dx, dy, dz], dtype=np.float32)
        self.transform.translation += delta

    def rotate(self, delta_x: float = 0.0, delta_y: float = 0.0, delta_z: float = 0.0) -> None:
        self.transform.rotation_deg += np.array([delta_x, delta_y, delta_z], dtype=np.float32)

    def scale(self, factor: float) -> None:
        self.transform.scale = max(0.1, min(5.0, self.transform.scale * float(factor)))

    def reset(self) -> None:
        self.reset_transform()

    def _draw_object(self, projection: np.ndarray, view: np.ndarray) -> None:
        raise NotImplementedError
