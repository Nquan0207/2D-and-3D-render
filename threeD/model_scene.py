from __future__ import annotations

import numpy as np

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D
from ..utils import meshio


class ModelScene(BaseScene3D):
    def __init__(self, renderer: Renderer, model_path: str) -> None:
        super().__init__(renderer, axes_scale=1.2)

        positions, colors, normals, indices = meshio.load_mesh(model_path)
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        center = (mins + maxs) * 0.5
        extent = maxs - mins
        max_dim = float(np.max(extent))
        scale = 1.0 / max_dim if max_dim > 1e-6 else 1.0
        positions = (positions - center) * scale
        if normals.shape[0] == positions.shape[0] and normals.size:
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normals = normals / norms

        # Safe texcoords placeholder
        texcoords = np.zeros((positions.shape[0], 2), dtype=np.float32)
        self.mesh = Mesh(positions, colors, normals, texcoords, indices)

        # Lighting/material defaults similar to other 3D scenes
        self.light_pos = np.array([2.5, 2.5, 2.5], dtype=np.float32)
        self.shininess = 48.0
        self.light_ambient = np.array([0.35, 0.35, 0.35], dtype=np.float32)
        self.light_diffuse = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.light_specular = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.mat_ambient = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.mat_diffuse = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        self.mat_specular = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        self.mode = RenderMode.PHONG

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
