from __future__ import annotations

import glfw
import numpy as np
import OpenGL.GL as GL

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D


class RectangleScene(BaseScene3D):
    def __init__(self, renderer: Renderer) -> None:
        super().__init__(renderer, axes_scale=1.2)

        # Axis-aligned rectangle centered at origin as a triangle fan
        x0, y0 = -0.8, -0.5
        x1, y1 = 0.8, 0.5
        center = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        # Order corners CCW starting from bottom-left
        ring = np.array(
            [
                [x0, y0, 0.0],
                [x1, y0, 0.0],
                [x1, y1, 0.0],
                [x0, y1, 0.0],
            ],
            dtype=np.float32,
        )
        # Close the fan by repeating the first corner
        positions = np.vstack([center, ring, ring[:1]]).astype(np.float32)
        # Continuous gradient colors derived from position (u,v in [0,1])
        uv = (positions[:, :2] + 1.0) * 0.5
        u, v = uv[:, 0], uv[:, 1]
        colors = np.stack([u, v, 1.0 - 0.5 * (u + v)], axis=1).astype(np.float32)
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (positions.shape[0], 1))
        # Map x,y in [-1,1] to [0,1]
        texcoords = (positions[:, :2] + 1.0) * 0.5
        self.mesh = Mesh(positions, colors, normals, texcoords, primitive=GL.GL_TRIANGLE_FAN)

        self.light_pos = np.array([2.5, 2.5, 2.5], dtype=np.float32)
        self.shininess = 48.0
        self.light_ambient = np.array([0.35, 0.35, 0.35], dtype=np.float32)
        self.light_diffuse = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.light_specular = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.mat_ambient = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.mat_diffuse = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        self.mat_specular = np.array([0.6, 0.6, 0.6], dtype=np.float32)

    # ------------------------------------------------------------------ drawing
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

    # Transform helpers now provided by BaseScene3D
