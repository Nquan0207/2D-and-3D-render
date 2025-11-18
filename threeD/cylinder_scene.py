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

    # side_pos = np.stack([xs, Y, zs], axis=-1).astype(np.float32)
    # Radial normals on side
    # nsx = np.cos(TH)
    # nsz = np.sin(TH)
    # side_nrm = np.stack([nsx, np.zeros_like(nsx), nsz], axis=-1).astype(np.float32)
    # # UVs for side (u wrap, v along height)
    # u = (TH / (2.0 * np.pi)).astype(np.float32)
    # v = ((Y + 0.5 * h) / max(h, 1e-8)).astype(np.float32)
    # side_uv = np.stack([u, v], axis=-1).astype(np.float32)
    # # Colors from normals
    # side_col = (side_nrm * 0.5 + 0.5).astype(np.float32)

    # # Indices for side
    # idx_side = []
    # cols = slices
    # # exactly two rows (bottom/top)
    # rows = side_pos.shape[0]
    # for i in range(rows - 1):
    #     for j in range(cols):
    #         jn = (j + 1) % cols
    #         i0 = i * cols + j
    #         i1 = i * cols + jn
    #         i2 = (i + 1) * cols + j
    #         i3 = (i + 1) * cols + jn
    #         idx_side.extend([i0, i1, i3, i0, i3, i2])

    # # Caps (top and bottom)
    # cap_positions = []
    # cap_normals = []
    # cap_colors = []
    # cap_uvs = []
    # cap_indices = []
    # base_index = side_pos.reshape(-1, 3).shape[0]
    # if caps:
    #     # Bottom cap (y = -h/2)
    #     center_bottom = np.array([0.0, -0.5 * h, 0.0], dtype=np.float32)
    #     ring_bottom = np.stack([r * np.cos(theta), np.repeat(-0.5 * h, theta.size), r * np.sin(theta)], axis=-1).astype(np.float32)
    #     cap_positions.append(center_bottom)
    #     cap_positions.append(ring_bottom)
    #     nb = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    #     cap_normals.append(nb)
    #     cap_normals.append(np.tile(nb, (slices, 1)))
    #     # Color from normal
    #     cb = (nb * 0.5 + 0.5)
    #     cap_colors.append(cb)
    #     cap_colors.append(np.tile(cb, (slices, 1)))
    #     # UVs map circle to [0,1]^2 for demonstration
    #     uv_center_b = np.array([0.5, 0.5], dtype=np.float32)
    #     uv_ring_b = 0.5 + 0.5 * np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)
    #     cap_uvs.append(uv_center_b)
    #     cap_uvs.append(uv_ring_b)

    #     # Indices bottom (fan)
    #     start = base_index
    #     # center
    #     cap_indices.extend([])
    #     # Flatten after building both caps to compute indices properly

    #     # Top cap (y = +h/2)
    #     center_top = np.array([0.0, 0.5 * h, 0.0], dtype=np.float32)
    #     ring_top = np.stack([r * np.cos(theta), np.repeat(0.5 * h, theta.size), r * np.sin(theta)], axis=-1).astype(np.float32)
    #     cap_positions.append(center_top)
    #     cap_positions.append(ring_top)
    #     nt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    #     cap_normals.append(nt)
    #     cap_normals.append(np.tile(nt, (slices, 1)))
    #     ct = (nt * 0.5 + 0.5)
    #     cap_colors.append(ct)
    #     cap_colors.append(np.tile(ct, (slices, 1)))
    #     uv_center_t = np.array([0.5, 0.5], dtype=np.float32)
    #     uv_ring_t = 0.5 + 0.5 * np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)
    #     cap_uvs.append(uv_center_t)
    #     cap_uvs.append(uv_ring_t)

    # # Flatten all arrays
    # side_pos_f = side_pos.reshape(-1, 3)
    # side_nrm_f = side_nrm.reshape(-1, 3)
    # side_col_f = side_col.reshape(-1, 3)
    # side_uv_f = side_uv.reshape(-1, 2)

    # positions = [side_pos_f]
    # normals = [side_nrm_f]
    # colors = [side_col_f]
    # uvs = [side_uv_f]
    # indices = [np.array(idx_side, dtype=np.uint32)]

    # if caps:
    #     # Compute indices for bottom and top after knowing base indices
    #     curr = side_pos_f.shape[0]
    #     # Bottom
    #     cb_center_idx = curr
    #     cb_ring_start = curr + 1
    #     positions.append(cap_positions[0].reshape(1, 3))
    #     positions.append(cap_positions[1])
    #     normals.append(cap_normals[0].reshape(1, 3))
    #     normals.append(cap_normals[1])
    #     colors.append(cap_colors[0].reshape(1, 3))
    #     colors.append(cap_colors[1])
    #     uvs.append(cap_uvs[0].reshape(1, 2))
    #     uvs.append(cap_uvs[1])
    #     idx_b = []
    #     for j in range(slices):
    #         jn = (j + 1) % slices
    #         idx_b.extend([cb_center_idx, cb_ring_start + jn, cb_ring_start + j])
    #     indices.append(np.array(idx_b, dtype=np.uint32))
    #     curr = cb_ring_start + slices
    #     # Top
    #     ct_center_idx = curr
    #     ct_ring_start = curr + 1
    #     positions.append(cap_positions[2].reshape(1, 3))
    #     positions.append(cap_positions[3])
    #     normals.append(cap_normals[2].reshape(1, 3))
    #     normals.append(cap_normals[3])
    #     colors.append(cap_colors[2].reshape(1, 3))
    #     colors.append(cap_colors[3])
    #     uvs.append(cap_uvs[2].reshape(1, 2))
    #     uvs.append(cap_uvs[3])
    #     idx_t = []
    #     for j in range(slices):
    #         jn = (j + 1) % slices
    #         idx_t.extend([ct_center_idx, ct_ring_start + j, ct_ring_start + jn])
    #     indices.append(np.array(idx_t, dtype=np.uint32))

    # positions_np = np.vstack(positions).astype(np.float32)
    # normals_np = np.vstack(normals).astype(np.float32)
    # colors_np = np.vstack(colors).astype(np.float32)
    # uvs_np = np.vstack(uvs).astype(np.float32)
    # indices_np = np.concatenate(indices).astype(np.uint32)
    # return positions_np, colors_np, normals_np, uvs_np, indices_np


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
