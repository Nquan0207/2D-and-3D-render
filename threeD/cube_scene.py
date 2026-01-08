from __future__ import annotations

import numpy as np

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D


class CubeScene(BaseScene3D):
    def __init__(self, renderer: Renderer) -> None:
        super().__init__(renderer, axes_scale=1.6)
        # self.camera.resize(width, height)

        self.mesh = self._create_mesh()

        self.flat_color = np.array([0.45, 0.6, 0.85], dtype=np.float32)
        self.light_pos = np.array([3.5, 3.5, 3.5], dtype=np.float32)
        self.light_ambient = np.array([0.35, 0.35, 0.35], dtype=np.float32)
        self.light_diffuse = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.light_specular = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.mat_ambient = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.mat_diffuse = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        self.mat_specular = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        self.shininess = 48.0

        self.transform.rotation_deg = np.array([25.0, 35.0, 0.0], dtype=np.float32)
        self.follow_camera_light = False

    # ------------------------------------------------------------------ drawing
    def _draw_object(self, projection: np.ndarray, view: np.ndarray) -> None:
        model = self._model_matrix()
        normal_matrix = np.linalg.inv(model[:3, :3]).T.astype(np.float32)
        light_pos = self.view_pos if self.follow_camera_light else self.light_pos

        specular_light = self.light_specular if self.mode == RenderMode.PHONG else np.zeros_like(self.light_specular)
        specular_mat = self.mat_specular if self.mode == RenderMode.PHONG else np.zeros_like(self.mat_specular)

        settings = RenderSettings(
            mode=self.mode,
            flat_color=self.flat_color,
            light_pos=light_pos.astype(np.float32),
            shininess=self.shininess,
            light_ambient=self.light_ambient,
            light_diffuse=self.light_diffuse,
            light_specular=specular_light,
            mat_ambient=self.mat_ambient,
            mat_diffuse=self.mat_diffuse,
            mat_specular=specular_mat,
        )
        self.renderer.draw(self.mesh, projection.astype(np.float32), view.astype(np.float32), model, normal_matrix, settings)

    # ------------------------------------------------------------------ internal helpers
    def _create_mesh(self) -> Mesh:
        s = 0.5
        faces = [
            (
                np.array([[+s, -s, +s], [+s, +s, +s], [-s, +s, +s], [-s, -s, +s]], dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
                np.array([0.95, 0.25, 0.25], dtype=np.float32),
                np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
            ),
            (
                np.array([[+s, -s, -s], [-s, -s, -s], [-s, +s, -s], [+s, +s, -s]], dtype=np.float32),
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
                np.array([0.30, 0.55, 0.95], dtype=np.float32),
                np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            ),
            (
                np.array([[-s, -s, +s], [-s, -s, -s], [-s, +s, -s], [-s, +s, +s]], dtype=np.float32),
                np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.40, 0.85, 0.40], dtype=np.float32),
                np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            ),
            (
                np.array([[+s, -s, -s], [+s, -s, +s], [+s, +s, +s], [+s, +s, -s]], dtype=np.float32),
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.95, 0.55, 0.25], dtype=np.float32),
                np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            ),
            (
                np.array([[-s, +s, +s], [-s, +s, -s], [+s, +s, -s], [+s, +s, +s]], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.95, 0.60, 0.80], dtype=np.float32),
                np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            ),
            (
                np.array([[-s, -s, -s], [-s, -s, +s], [+s, -s, +s], [+s, -s, -s]], dtype=np.float32),
                np.array([0.0, -1.0, 0.0], dtype=np.float32),
                np.array([0.95, 0.85, 0.30], dtype=np.float32),
                np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            ),
        ]

        positions = []
        colors = []
        normals = []
        texcoords = []
        indices = []
        for verts, normal, color, uvs in faces:
            base = len(positions)
            positions.extend(verts)
            colors.extend([color] * 4)
            normals.extend([normal] * 4)
            texcoords.extend(uvs)
            indices.extend([
                base + 0,
                base + 1,
                base + 2,
                base + 0,
                base + 2,
                base + 3,
            ])

        return Mesh(
            np.array(positions, dtype=np.float32),
            np.array(colors, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(texcoords, dtype=np.float32),
            np.array(indices, dtype=np.uint32),
        )
