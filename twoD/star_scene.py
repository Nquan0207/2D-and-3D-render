from __future__ import annotations

import glfw
import numpy as np
import OpenGL.GL as GL

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D


def _star_triangles(points: int = 5, r_outer: float = 0.85, r_inner: float = 0.4, start_angle: float = -np.pi/2) -> np.ndarray:
    # Build alternating outer/inner ring around the center, CCW
    angles = start_angle + np.arange(0, points * 2) * (np.pi / points)
    radii = np.where(np.arange(0, points * 2) % 2 == 0, r_outer, r_inner)
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    ring = np.stack([xs, ys, np.zeros_like(xs)], axis=1).astype(np.float32)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    # Triangulate as (center, ring[i], ring[i+1]) using GL_TRIANGLES (not fan primitive)
    tris = []
    n = ring.shape[0]
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        tris.extend([center, a, b])
    return np.array(tris, dtype=np.float32)


class StarScene(BaseScene3D):
    def __init__(self, renderer: Renderer) -> None:
        super().__init__(renderer, axes_scale=1.2)

        # Flip orientation by starting at +pi/2 (points downward versus upward)
        positions = _star_triangles(points=5, r_outer=0.85, r_inner=0.38, start_angle=np.pi/2)
        uv = (positions[:, :2] + 1.0) * 0.5
        u, v = uv[:, 0], uv[:, 1]
        colors = np.stack([u, v, 1.0 - 0.5 * (u + v)], axis=1).astype(np.float32)
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (positions.shape[0], 1))
        texcoords = (positions[:, :2] + 1.0) * 0.5
        self.mesh = Mesh(positions, colors, normals, texcoords)

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
            mat_specular=self.mat_specular if self.mode == RenderMode.PHONG else np.zeros_like(self.mat_specular),
        )
        self.renderer.draw(self.mesh, projection, view, model, normal_matrix, settings)
