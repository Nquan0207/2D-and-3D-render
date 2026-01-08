from __future__ import annotations

import numpy as np

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D


def _build_frustum(slices: int = 64, r_bottom: float = 1.0, r_top: float = 0.5, height: float = 2.0, caps: bool = True):
    slices = max(3, int(slices))
    rb = float(r_bottom)
    rt = float(r_top)
    h = float(height)

    theta = np.linspace(0.0, 2.0 * np.pi, slices, endpoint=False, dtype=np.float32)
    yb = -0.5 * h
    yt = 0.5 * h
    ring_b = np.stack([rb * np.cos(theta), np.repeat(yb, theta.size), rb * np.sin(theta)], axis=-1).astype(np.float32)
    ring_t = np.stack([rt * np.cos(theta), np.repeat(yt, theta.size), rt * np.sin(theta)], axis=-1).astype(np.float32)

    positions = []
    normals = []
    colors = []
    texcoords = []
    indices = []

    # Side normals: radial with slope s = (rb-rt)/h
    s = (rb - rt) / max(h, 1e-8)
    nx = np.cos(theta)
    nz = np.sin(theta)
    n_side = np.stack([nx, np.full_like(nx, s), nz], axis=-1)
    n_side = n_side / (np.linalg.norm(n_side, axis=-1, keepdims=True) + 1e-8)

    def vcol(p):
        d = p / (np.linalg.norm(p) + 1e-8)
        return (d * 0.5 + 0.5).astype(np.float32)

    def add_tri(a, na, b, nb, c, nc):
        base = len(positions)
        positions.extend([a, b, c])
        normals.extend([na, nb, nc])
        colors.extend([vcol(a), vcol(b), vcol(c)])
        texcoords.extend([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        indices.extend([base, base + 1, base + 2])

    # Sides (two triangles per slice)
    for j in range(slices):
        jn = (j + 1) % slices
        b0, b1 = ring_b[j], ring_b[jn]
        t0, t1 = ring_t[j], ring_t[jn]
        n0, n1 = n_side[j], n_side[jn]
        # (b0, b1, t1)
        add_tri(b0, n0, b1, n1, t1, n1)
        # (b0, t1, t0)
        add_tri(b0, n0, t1, n1, t0, n0)

    if caps:
        nb = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        tb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        center_b = np.array([0.0, yb, 0.0], dtype=np.float32)
        center_t = np.array([0.0, yt, 0.0], dtype=np.float32)
        for j in range(slices):
            jn = (j + 1) % slices
            # bottom (center, ring_b[jn], ring_b[j])
            add_tri(center_b, nb, ring_b[jn], nb, ring_b[j], nb)
            # top (center, ring_t[j], ring_t[jn])
            add_tri(center_t, tb, ring_t[j], tb, ring_t[jn], tb)

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(colors, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(texcoords, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


class FrustumScene(BaseScene3D):
    def __init__(self, renderer: Renderer, slices: int = 48, r_bottom: float = 1.0, r_top: float = 0.5, height: float = 2.0, caps: bool = True) -> None:
        super().__init__(renderer, axes_scale=1.6)
        p, c, n, t, idx = _build_frustum(slices=slices, r_bottom=r_bottom, r_top=r_top, height=height, caps=caps)
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

