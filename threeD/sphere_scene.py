from __future__ import annotations

import numpy as np

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D


def _build_uv_sphere(stacks: int = 32, slices: int = 64, radius: float = 1.0):
    stacks = max(3, int(stacks))
    slices = max(3, int(slices))

    # Angles: phi in [0, pi], theta in [0, 2pi)
    phi = np.linspace(0.0, np.pi, stacks + 1, dtype=np.float32)
    theta = np.linspace(0.0, 2.0 * np.pi, slices + 1, dtype=np.float32)
    # Meshgrid with indexing='ij' so first dimension is phi (stacks)
    PHI, THETA = np.meshgrid(phi, theta, indexing="ij")

    x = radius * np.sin(PHI) * np.cos(THETA)
    y = radius * np.cos(PHI)
    z = radius * np.sin(PHI) * np.sin(THETA)
    positions = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Normals from positions (unit length)
    normals = positions / radius

    # UVs: u = theta / (2pi), v = phi / pi
    u = (THETA / (2.0 * np.pi)).astype(np.float32)
    v = (PHI / np.pi).astype(np.float32)
    texcoords = np.stack([u, v], axis=-1).astype(np.float32)

    colors = (normals * 0.5 + 0.5).astype(np.float32)

    # Indices for triangle grid
    idx = []
    cols = slices + 1
    rows = stacks + 1
    for i in range(rows - 1):
        for j in range(cols - 1):
            i0 = i * cols + j
            i1 = i * cols + (j + 1)
            i2 = (i + 1) * cols + j
            i3 = (i + 1) * cols + (j + 1)
            # Two triangles (i0, i1, i3) and (i0, i3, i2)
            idx.extend([i0, i1, i3, i0, i3, i2])

    positions = positions.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    normals = normals.reshape(-1, 3)
    texcoords = texcoords.reshape(-1, 2)
    indices = np.array(idx, dtype=np.uint32)
    return positions, colors, normals, texcoords, indices


class SphereScene(BaseScene3D):
    def __init__(self, renderer: Renderer, stacks: int = 32, slices: int = 64, radius: float = 1.0) -> None:
        super().__init__(renderer, axes_scale=1.6)

        p, c, n, t, indices = _build_uv_sphere(stacks=stacks, slices=slices, radius=radius)
        self.mesh = Mesh(p, c, n, t, indices)

        # Lighting/material defaults similar to other 3D scenes
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
