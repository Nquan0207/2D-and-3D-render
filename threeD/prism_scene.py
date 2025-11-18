from __future__ import annotations

import numpy as np

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D
import OpenGL.GL as GL
def _build_triangular_prism():
    # Build prism using face-based triangles with flat normals and vcolor
    B0 = np.array([ 1.0, -1.0,  0.0], dtype=np.float32)
    B1 = np.array([-0.5, -1.0,  1.0], dtype=np.float32)
    B2 = np.array([-0.5, -1.0, -1.0], dtype=np.float32)
    T0 = np.array([ 1.0,  1.0,  0.0], dtype=np.float32)
    T1 = np.array([-0.5,  1.0,  1.0], dtype=np.float32)
    T2 = np.array([-0.5,  1.0, -1.0], dtype=np.float32)

    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    texcoords: list[list[float]] = []

    def _vcolor(p: np.ndarray) -> np.ndarray:
        d = p / (np.linalg.norm(p) + 1e-8)
        return (d * 0.5 + 0.5).astype(np.float32)

    def add_face(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        n = np.cross(b - a, c - a)
        n = n / (np.linalg.norm(n) + 1e-8)
        positions.extend([a, b, c])
        normals.extend([n, n, n])
        colors.extend([_vcolor(a), _vcolor(b), _vcolor(c)])
        texcoords.extend([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

    # Caps
    add_face(B0, B2, B1)
    add_face(T0, T1, T2)
    # Sides (3 quads -> 6 tris)
    add_face(B0, B1, T1)
    add_face(B0, T1, T0)
    add_face(B1, B2, T2)
    add_face(B1, T2, T1)
    add_face(B2, B0, T0)
    add_face(B2, T0, T2)

    p = np.asarray(positions, dtype=np.float32)
    c = np.asarray(colors, dtype=np.float32)
    n = np.asarray(normals, dtype=np.float32)
    t = np.asarray(texcoords, dtype=np.float32)
    idx = np.arange(p.shape[0], dtype=np.uint32)
    return p, c, n, t, idx


class PrismScene(BaseScene3D):
    def __init__(self, renderer: Renderer) -> None:
        super().__init__(renderer, axes_scale=1.6)
        p, c, n, t, indices = _build_triangular_prism()
        self.mesh = Mesh(p, c, n, t, indices)

        # Lighting/material
        self.light_pos = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        self.shininess = 64.0
        self.light_ambient = np.array([0.35, 0.35, 0.35], dtype=np.float32)
        self.light_diffuse = np.array([0.95, 0.95, 0.95], dtype=np.float32)
        self.light_specular = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.mat_ambient = np.array([0.45, 0.45, 0.45], dtype=np.float32)
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
