from __future__ import annotations

import glfw
import numpy as np
import OpenGL.GL as GL

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D


class TrapezoidScene(BaseScene3D):
    def __init__(self, renderer: Renderer) -> None:
        super().__init__(renderer, axes_scale=1.2)

        # Isosceles trapezoid: wider bottom, narrower top
        x_left_bot, y_bot = -0.9, -0.5
        x_right_bot = 0.9
        x_left_top, y_top = -0.5, 0.5
        x_right_top = 0.5

        # CCW corners: bottom-left, bottom-right, top-right, top-left
        bl = np.array([x_left_bot, y_bot, 0.0], dtype=np.float32)
        br = np.array([x_right_bot, y_bot, 0.0], dtype=np.float32)
        tr = np.array([x_right_top, y_top, 0.0], dtype=np.float32)
        tl = np.array([x_left_top, y_top, 0.0], dtype=np.float32)
        # Two triangles (bl, br, tr) and (bl, tr, tl)
        positions = np.array([
            bl, br, tr,
            bl, tr, tl,
        ], dtype=np.float32)

        # Gradient colors from position
        uv = (positions[:, :2] + 1.0) * 0.5
        u, v = uv[:, 0], uv[:, 1]
        colors = np.stack([u, v, 1.0 - 0.5 * (u + v)], axis=1).astype(np.float32)
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (positions.shape[0], 1))
        texcoords = (positions[:, :2] + 1.0) * 0.5
        self.mesh = Mesh(positions, colors, normals, texcoords)

        # Lighting/material
        self.light_pos = np.array([2.5, 2.5, 2.5], dtype=np.float32)
        self.shininess = 48.0
        self.light_ambient = np.array([0.35, 0.35, 0.35], dtype=np.float32)
        self.light_diffuse = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.light_specular = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.mat_ambient = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.mat_diffuse = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        self.mat_specular = np.array([0.6, 0.6, 0.6], dtype=np.float32)

    def _draw_object(self, projection: np.ndarray, view: np.ndarray) -> None:
        model = self._model_matrix()
        normal_matrix = np.linalg.inv(model[:3, :3]).T.astype(np.float32)
        specular_light = self.light_specular if self.mode == RenderMode.PHONG else np.zeros_like(self.light_specular)
        specular_mat = self.mat_specular if self.mode == RenderMode.PHONG else np.zeros_like(self.mat_specular)

        settings = RenderSettings(
            mode=self.mode,
            flat_color=self.flat_color,
            light_pos=self.light_pos,
            shininess=self.shininess,
            light_ambient=self.light_ambient,
            light_diffuse=self.light_diffuse,
            light_specular=specular_light,
            mat_ambient=self.mat_ambient,
            mat_diffuse=self.mat_diffuse,
            mat_specular=specular_mat,
        )
        self.renderer.draw(self.mesh, projection, view, model, normal_matrix, settings)
