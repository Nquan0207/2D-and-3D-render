from __future__ import annotations

import numpy as np

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D


def _regular_tetrahedron(scale: float = 1.0):
    # Regular tetrahedron vertices (centered), using coordinates with symmetry
    v = np.array([
        [1.0,  1.0,  1.0],
        [-1.0, -1.0,  1.0],
        [-1.0,  1.0, -1.0],
        [1.0, -1.0, -1.0],
    ], dtype=np.float32)
    # Normalize so the circumscribed radius is ~1, then scale
    v = v / np.linalg.norm([1.0, 1.0, 1.0]) * scale

    faces = [
        (0, 1, 2),
        (0, 3, 1),
        (0, 2, 3),
        (1, 3, 2),
    ]
    positions = []
    normals = []
    colors = []
    texcoords = []
    indices = []
    # Assign a distinct color per face
    face_colors = [
        [0.95, 0.35, 0.35],
        [0.35, 0.80, 0.45],
        [0.35, 0.55, 0.95],
        [0.95, 0.85, 0.35],
    ]
    for fi, (a, b, c) in enumerate(faces):
        pa, pb, pc = v[a], v[b], v[c]
        # Face normal (flat shading), normalized
        n = np.cross(pb - pa, pc - pa)
        n = n / (np.linalg.norm(n) + 1e-8)
        base = len(positions)
        positions.extend([pa, pb, pc])
        normals.extend([n, n, n])
        col = np.array(face_colors[fi], dtype=np.float32)
        colors.extend([col, col, col])
        # Simple planar UVs (placeholder)
        texcoords.extend([ [0.0, 0.0], [1.0, 0.0], [0.5, 1.0] ])
        indices.extend([base + 0, base + 1, base + 2])

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(colors, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(texcoords, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


class TetrahedronScene(BaseScene3D):
    def __init__(self, renderer: Renderer) -> None:
        super().__init__(renderer, axes_scale=1.6)

        p, c, n, t, indices = _regular_tetrahedron(scale=0.9)
        self.mesh = Mesh(p, c, n, t, indices)

        # Lighting/material defaults (Phong looks nice with flat normals)
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

