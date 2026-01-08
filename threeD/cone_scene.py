from __future__ import annotations

import numpy as np

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D


def _build_cone(slices: int = 64, radius: float = 1.0, height: float = 2.0, caps: bool = True):
    slices = max(3, int(slices))
    r = float(radius)
    h = float(height)

    theta = np.linspace(0.0, 2.0 * np.pi, slices, endpoint=False, dtype=np.float32)
    # Geometry
    yb = -0.5 * h
    yt = 0.5 * h
    apex = np.array([[0.0, yt, 0.0]], dtype=np.float32)
    ring = np.stack([r * np.cos(theta), np.full_like(theta, yb), r * np.sin(theta)], axis=-1).astype(np.float32)

    positions = []
    normals = []
    colors = []
    texcoords = []
    indices = []

    # Side normals: radial with slope along Y. Tangent slope s = r/h.
    s = r / max(h, 1e-8)
    nx = np.cos(theta)
    nz = np.sin(theta)
    side_nrm_ring = np.stack([nx, np.full_like(nx, s), nz], axis=-1)
    side_nrm_ring = side_nrm_ring / (np.linalg.norm(side_nrm_ring, axis=-1, keepdims=True) + 1e-8)
    # Apex normal approximate as average of all side normals (unit)
    apex_n = side_nrm_ring.mean(axis=0)
    apex_n = apex_n / (np.linalg.norm(apex_n) + 1e-8)

    def add_tri(a, na, b, nb, c, nc):
        base = len(positions)
        positions.extend([a, b, c])
        normals.extend([na, nb, nc])
        # Seamless vertex colors from position (normalized to [0,1])
        def vcol(p):
            d = p / (np.linalg.norm(p) + 1e-8)
            return (d * 0.5 + 0.5).astype(np.float32)
        colors.extend([vcol(a), vcol(b), vcol(c)])
        texcoords.extend([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        indices.extend([base, base + 1, base + 2])

    # Sides: (apex, ring[j], ring[jn]) with corresponding normals
    for j in range(slices):
        jn = (j + 1) % slices
        add_tri(apex[0], apex_n, ring[j], side_nrm_ring[j], ring[jn], side_nrm_ring[jn])

    if caps:
        # Bottom cap (flat): (center, ring[jn], ring[j]) clockwise to face downward
        nb = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        cb_center = np.array([0.0, yb, 0.0], dtype=np.float32)
        for j in range(slices):
            jn = (j + 1) % slices
            add_tri(cb_center, nb, ring[jn], nb, ring[j], nb)

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(colors, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(texcoords, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


class ConeScene(BaseScene3D):
    def __init__(self, renderer: Renderer, slices: int = 48, radius: float = 1.0, height: float = 2.0, caps: bool = True) -> None:
        super().__init__(renderer, axes_scale=1.6)
        p, c, n, t, idx = _build_cone(slices=slices, radius=radius, height=height, caps=caps)
        self.mesh = Mesh(p, c, n, t, idx)

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

