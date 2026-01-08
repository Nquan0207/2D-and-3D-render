from __future__ import annotations

import numpy as np

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D


def _build_cylinder(slices: int = 32, radius: float = 1.0, height: float = 2.0, caps: bool = True):
    r = float(radius)
    h = float(height)
    yb, yt = -0.5 * h, 0.5 * h
    theta = np.linspace(0.0, 2.0 * np.pi, slices, endpoint=False, dtype=np.float32)
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    ring_bot = np.stack([x, np.full_like(theta, yb), z], axis=-1).astype(np.float32)
    ring_top = np.stack([x, np.full_like(theta, yt), z], axis=-1).astype(np.float32)
    center_bot = np.array([0.0, yb, 0.0], dtype=np.float32)
    center_top = np.array([0.0, yt, 0.0], dtype=np.float32)
    ring_bot_cap = ring_bot.copy()  # (N,3)
    ring_top_cap = ring_top.copy()  # (N,3)\
    positions = np.vstack([ring_bot, ring_top, center_bot, center_top, ring_bot_cap, ring_top_cap]).astype(np.float32)
    
    nr = np.stack([x, np.zeros_like(x), z], axis=-1).astype(np.float32)  # (N,3)
    # Cap normals
    nb = np.array([[0.0, -1.0, 0.0]], dtype=np.float32)
    nt = np.array([[0.0,  1.0, 0.0]], dtype=np.float32)

    normals_side = np.vstack([nr, nr])  # bottom ring (N,3) + top ring (N,3)
    normals_centers = np.vstack([nb, nt])  # centers (2,3)
    normals_caps = np.vstack([np.repeat(nb, slices, axis=0),  
                              np.repeat(nt, slices, axis=0)]) 
    normals = np.vstack([normals_side, normals_centers, normals_caps]).astype(np.float32)

    colors = (normals * 0.5 + 0.5).astype(np.float32)
        # --- UVs
    # Side UVs: u wraps by angle, v along height (0 bottom, 1 top)
    u_side = (theta / (2.0 * np.pi)).astype(np.float32) 
    v_bot = np.zeros_like(u_side, dtype=np.float32)      # bottom ring v=0
    v_top = np.ones_like(u_side, dtype=np.float32)       # top ring v=1
    uv_bot_side = np.stack([u_side, v_bot], axis=-1).astype(np.float32)  # (N,2)
    uv_top_side = np.stack([u_side, v_top], axis=-1).astype(np.float32)  # (N,2)

    # Centers UV for caps (mapped to disk center)
    uv_center_b = np.array([[0.5, 0.5]], dtype=np.float32)
    uv_center_t = np.array([[0.5, 0.5]], dtype=np.float32)

    # Cap ring UVs (map unit circle to [0,1]^2)
    uv_ring_cap = 0.5 + 0.5 * np.stack([x, z], axis=-1).astype(np.float32)  # (N,2)
    uv_bot_cap = uv_ring_cap.copy()
    uv_top_cap = uv_ring_cap.copy()

    # Assemble UVs in the same order as positions
    uvs = np.vstack([uv_bot_side,      # bottom ring (side)
                     uv_top_side,      # top ring   (side)
                     uv_center_b,      # center bottom (cap)
                     uv_center_t,      # center top    (cap)
                     uv_bot_cap,       # bottom ring   (cap dup)
                     uv_top_cap]).astype(np.float32)

    # --- Indices
    idx = []

    for j in range(slices):
        jn = (j + 1) % slices
        b0 = j
        t0 = slices + j
        b1 = jn
        t1 = slices + jn
        idx.extend([b0, t1, t0, b0, b1, t1])
    if caps:
                # Bottom cap: center index = 2N, ring start (dup) = 2N + 2
        cb = 2 * slices
        bot_ring_cap_start = 2 * slices + 2
        for j in range(slices):
            jn = (j + 1) % slices
            idx.extend([cb, bot_ring_cap_start + jn, bot_ring_cap_start + j])

                # Top cap: center index = 2N + 1, ring start (dup) = 2N + 2 + N
        ct_idx = 2 * slices + 1
        top_ring_cap_start = 2 * slices + 2 + slices
        for j in range(slices):
            jn = (j + 1) % slices
            idx.extend([ct_idx, top_ring_cap_start + j, top_ring_cap_start + jn])
        
    indices = np.asarray(idx, dtype=np.uint32)
    return positions, colors, normals, uvs, indices

class CylinderScene(BaseScene3D):
    def __init__(self, renderer: Renderer, slices: int = 32, radius: float = 1.0, height: float = 2.0, caps: bool = True) -> None:
        super().__init__(renderer, axes_scale=1.6)

        p, c, n, t, indices = _build_cylinder(slices=slices, radius=radius, height=height, caps=caps)
        self.mesh = Mesh(p, c, n, t, indices)

        # Lighting/material similar to Sphere
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
