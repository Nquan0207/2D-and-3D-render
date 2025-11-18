from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..libs import transform as T
from .sphere_scene import _build_uv_sphere
from ..scene_base import BaseScene3D
import OpenGL.GL as GL
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath

def _safe_eval_func(expr: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    allowed = {
        # numpy and common symbols
        'np': np,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'asin': np.arcsin,
        'acos': np.arccos,
        'atan': np.arctan,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'exp': np.exp,
        'log': np.log,
        'log10': np.log10,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
        'e': np.e,
    }

    code = compile(expr, '<expr>', 'eval')

    def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(eval(code, {"__builtins__": {}}, {**allowed, 'x': x, 'y': y}), dtype=np.float32)

    return f


@dataclass
class SurfaceSpec:
    expr: str = "x+y"
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    resolution: int = 64


class SurfaceScene(BaseScene3D):
    def __init__(self, renderer: Renderer) -> None:
        super().__init__(renderer, axes_scale=1.2)
        self.spec = SurfaceSpec()
        self.mesh = self._build_mesh()
        # Contour rendering state
        self.show_contours: bool = False
        self.contour_mesh: Mesh | None = None
        self.contour_levels: int = 10
        self.contour_cmap: str = "turbo"
        self.mode = RenderMode.PHONG
        # Rolling ball state
        self.show_ball: bool = False
        self.ball_radius: float = 0.15  # in surface units
        self.ball_speed: float = 0.6    # legacy param; unused after removing parametric path
        self._ball_mesh: Mesh | None = None
        self._ball_last_time: float | None = None
        # Rolling spin state (updated per-frame)
        self._ball_rot: np.ndarray = np.eye(4, dtype=np.float32)
        # Ball render mode (independent of surface mode)
        self.ball_mode = RenderMode.PHONG
        # Dynamic rolling physics state
        self.ball_dyn: bool = False
        self.ball_gravity: float = 9.8
        self.ball_damping: float = 0.2
        cx = 0.5 * (self.spec.x_min + self.spec.x_max)
        cy = 0.5 * (self.spec.y_min + self.spec.y_max)
        self.ball_pos: np.ndarray = np.array([cx, cy], dtype=np.float32)
        self.ball_vel: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)
        # Freeze-mode spin helper (track previous anchor to derive motion)
        self._freeze_prev_xy: np.ndarray | None = None
        self.ball_freeze: bool = False
        # Default anchor at domain center (used when freezing the ball)
        cx0 = 0.5 * (self.spec.x_min + self.spec.x_max)
        cy0 = 0.5 * (self.spec.y_min + self.spec.y_max)
        self._ball_anchor: np.ndarray = np.array([cx0, cy0], dtype=np.float32)

        # lighting defaults
        self.light_pos = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        self.shininess = 100.0
        self.light_ambient = np.array([0.35, 0.35, 0.35], dtype=np.float32)
        self.light_diffuse = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.light_specular = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.mat_ambient = np.array([0.4, 0.4, 0.4], dtype=np.float32)
        self.mat_diffuse = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        self.mat_specular = np.array([0.6, 0.6, 0.6], dtype=np.float32)

        self.translate_step = 0.05
        self.rotate_step = 5.0

    # --------------------------------------------------------------- public API
    def set_function(self, expr: str) -> tuple[bool, str]:
        expr = expr.strip()
        if not expr:
            return False, "Function expression cannot be empty."
        try:
            _safe_eval_func(expr)  # validate
        except Exception as exc:  # noqa: BLE001
            return False, f"Invalid expression: {exc}"
        self.spec.expr = expr
        self.mesh = self._build_mesh()
        # Invalidate contour mesh so it rebuilds on next draw if needed
        self.contour_mesh = None
        return True, "Surface updated."

    def set_resolution(self, n: int) -> None:
        n = int(max(4, min(512, n)))
        if n != self.spec.resolution:
            self.spec.resolution = n
            self.mesh = self._build_mesh()
            # Invalidate contour mesh as grid changed
            self.contour_mesh = None

    def set_show_contours(self, show: bool) -> None:
        self.show_contours = bool(show)
        if self.show_contours and self.contour_mesh is None:
            self.contour_mesh = self._build_contour_mesh()

    # ------------------------------- rolling ball controls
    def set_show_ball(self, show: bool) -> None:
        self.show_ball = bool(show)
        if self.show_ball and self._ball_mesh is None:
            self._ball_mesh = self._build_ball_mesh()

    def set_ball_params(self, *, radius: float | None = None, speed: float | None = None) -> None:
        if radius is not None:
            self.ball_radius = max(1e-3, float(radius))
        if speed is not None:
            self.ball_speed = max(0.0, float(speed))
        # Rebuild ball with unit radius mesh; we scale in model, so no rebuild needed

    def set_ball_mode(self, mode: RenderMode) -> None:
        # Allow only non-texture modes for the ball
        if mode in (RenderMode.FLAT, RenderMode.GOURAUD, RenderMode.PHONG, RenderMode.WIREFRAME):
            self.ball_mode = mode
    
    def set_ball_dynamic(self, dyn: bool) -> None:
        self.ball_dyn = bool(dyn)
        # Initialize integrator state when switching on
        if self.ball_dyn:
            cx = 0.5 * (self.spec.x_min + self.spec.x_max)
            cy = 0.5 * (self.spec.y_min + self.spec.y_max)
            self.ball_pos = np.array([cx, cy], dtype=np.float32)
            self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)
            self._ball_last_time = None

    def set_ball_freeze(self, freeze: bool) -> None:
        self.ball_freeze = bool(freeze)
        if self.ball_freeze:
            # Capture current animated XY as anchor if possible
            tnow = self._get_time()
            x, y = self._ball_xy_at_time(tnow)
            self._ball_anchor = np.array([x, y], dtype=np.float32)
        # Reset freeze delta tracking to avoid jumpy spin
        self._freeze_prev_xy = None

    def set_ball_anchor(self, x: float, y: float) -> None:
        s = self.spec
        x = float(max(s.x_min, min(s.x_max, x)))
        y = float(max(s.y_min, min(s.y_max, y)))
        self._ball_anchor = np.array([x, y], dtype=np.float32)

    def get_ball_anchor(self) -> tuple[float, float]:
        return float(self._ball_anchor[0]), float(self._ball_anchor[1])

    # Start dynamic rolling from current anchor: unfreeze and initialize state
    def drop_ball_from_anchor(self) -> None:
        # Ensure dynamics mode is on
        self.ball_dyn = True
        # Start from anchor position with zero initial velocity
        self.ball_pos = self._ball_anchor.astype(np.float32).copy()
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)
        # Unfreeze and clear helpers so integration starts cleanly
        self.ball_freeze = False
        self._freeze_prev_xy = None
        self._ball_last_time = None

    def get_xy_extent(self) -> float:
        s = self.spec
        return float(max(s.x_max - s.x_min, s.y_max - s.y_min))

    # ---------------------------------------------------------------- drawing
    def _draw_object(self, projection: np.ndarray, view: np.ndarray) -> None:
        model = self._model_matrix()
        normal_matrix = np.linalg.inv(model[:3, :3]).T.astype(np.float32)

        # If showing contours, draw line mesh using Gouraud (per-vertex color)
        draw_mesh = self.contour_mesh if self.show_contours else self.mesh
        active_mode = RenderMode.PHONG if self.show_contours else self.mode

        specular_light = self.light_specular if active_mode == RenderMode.PHONG else np.zeros_like(self.light_specular)
        specular_mat = self.mat_specular if active_mode == RenderMode.PHONG else np.zeros_like(self.mat_specular)

        settings = RenderSettings(
            mode=active_mode,
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
        # Lazy build contours on first use
        if self.show_contours and self.contour_mesh is None:
            self.contour_mesh = self._build_contour_mesh()
            draw_mesh = self.contour_mesh
        self.renderer.draw(draw_mesh, projection, view, model, normal_matrix, settings)

        # Draw rolling ball if enabled
        if self.show_ball:
            if self._ball_mesh is None:
                self._ball_mesh = self._build_ball_mesh()
            ball_model = self._ball_model_matrix()
            ball_normal_matrix = np.linalg.inv(ball_model[:3, :3]).T.astype(np.float32)
            ball_settings = RenderSettings(
                mode=self.ball_mode,
                flat_color=np.array([0.9, 0.2, 0.2], dtype=np.float32),
                light_pos=self.light_pos,
                shininess=self.shininess,
                light_ambient=self.light_ambient,
                light_diffuse=self.light_diffuse,
                light_specular=self.light_specular,
                mat_ambient=self.mat_ambient,
                mat_diffuse=self.mat_diffuse,
                mat_specular=self.mat_specular,
            )
            self.renderer.draw(self._ball_mesh, projection, view, ball_model, ball_normal_matrix, ball_settings)

    # -------------------------------------------------------------- mesh build
    def _build_mesh(self) -> Mesh:
        s = self.spec
        n = int(s.resolution)
        xs = np.linspace(s.x_min, s.x_max, n, dtype=np.float32)
        ys = np.linspace(s.y_min, s.y_max, n, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys, indexing='xy')

        f = _safe_eval_func(s.expr)
        try:
            Z = f(X, Y)
        except Exception:
            Z = np.zeros_like(X, dtype=np.float32)

        # normals from gradients: n ~ (-dz/dx, -dz/dy, 1)
        dx = float((s.x_max - s.x_min) / max(n - 1, 1))
        dy = float((s.y_max - s.y_min) / max(n - 1, 1))
        dZ_dy, dZ_dx = np.gradient(Z, dy, dx, edge_order=2)
        Nx = -dZ_dx
        Ny = -dZ_dy
        Nz = np.ones_like(Z, dtype=np.float32)
        N = np.stack([Nx, Ny, Nz], axis=-1)
        N_norm = np.linalg.norm(N, axis=-1, keepdims=True) + 1e-8
        N = (N / N_norm).astype(np.float32)

        # positions and attributes
        positions = np.stack([X, Y, Z], axis=-1).astype(np.float32)

        # colors mapped from height
        zmin, zmax = float(np.min(Z)), float(np.max(Z))
        rng = max(zmax - zmin, 1e-6)
        t = ((Z - zmin) / rng).astype(np.float32)
        colors = np.stack([t, 0.5 * (1.0 - t) + 0.25, 1.0 - t], axis=-1).astype(np.float32)

        # texcoords
        us = (X - s.x_min) / max(s.x_max - s.x_min, 1e-6)
        vs = (Y - s.y_min) / max(s.y_max - s.y_min, 1e-6)
        uvs = np.stack([us, vs], axis=-1).astype(np.float32)

        # indices for triangles
        idx = []
        for j in range(n - 1):
            base0 = j * n
            base1 = (j + 1) * n
            for i in range(n - 1):
                i0 = base0 + i
                i1 = base0 + i + 1
                i2 = base1 + i
                i3 = base1 + i + 1
                idx.extend([i0, i1, i3, i0, i3, i2])
        indices = np.array(idx, dtype=np.uint32)

        return Mesh(
            positions.reshape(-1, 3),
            colors.reshape(-1, 3),
            N.reshape(-1, 3),
            uvs.reshape(-1, 2),
            indices,
        )

    # ---------------------------------------------------------- contours (lines)
    def _generate_contours(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, levels: int = 10, cmap: str = "turbo"):
        
        Zabs = float(np.abs(Z).max())
        scale = 5.0
        if Zabs != 0.0:
            Z_scaled = (Z / Zabs) * scale
        else:
            Z_scaled = Z

        contours = plt.contour(X, Y, Z_scaled, levels=levels, cmap=cmap)

        contour_vertices: list[np.ndarray] = []
        contour_colors: list[np.ndarray] = []

        if hasattr(contours, "collections"):
            # Standard path via LineCollections and Path objects
            for collection, level in zip(contours.collections, contours.levels):
                color = contours.cmap(contours.norm(level))[:3]
                paths = collection.get_paths()
                for path in paths:
                    vertices = path.vertices
                    codes = path.codes
                    if codes is None:
                        segments = [vertices]
                    else:
                        segments = []
                        start_idx = 0
                        for i, code in enumerate(codes):
                            if code == MPath.MOVETO and i > 0:
                                segments.append(vertices[start_idx:i])
                                start_idx = i
                        if start_idx < len(vertices):
                            segments.append(vertices[start_idx:])

                    for segment in segments:
                        n_vertices = int(segment.shape[0])
                        if n_vertices < 2:
                            continue
                        z_val = (float(level) / scale) * Zabs if Zabs != 0.0 else float(level)
                        z_column = np.full((n_vertices, 1), z_val, dtype=np.float32)
                        verts3 = np.column_stack((segment[:, 0], segment[:, 1], z_column)).astype(np.float32)
                        contour_vertices.append(verts3)
                        contour_colors.append(np.array([color] * n_vertices, dtype=np.float32))
        else:
            # Fallback using allsegs (list of arrays per level)
            allsegs = getattr(contours, "allsegs", [])
            for level, segs in zip(contours.levels, allsegs):
                color = contours.cmap(contours.norm(level))[:3]
                for segment in segs:
                    n_vertices = int(segment.shape[0])
                    if n_vertices < 2:
                        continue
                    z_val = (float(level) / scale) * Zabs if Zabs != 0.0 else float(level)
                    z_column = np.full((n_vertices, 1), z_val, dtype=np.float32)
                    verts3 = np.column_stack((segment[:, 0], segment[:, 1], z_column)).astype(np.float32)
                    contour_vertices.append(verts3)
                    contour_colors.append(np.array([color] * n_vertices, dtype=np.float32))

        plt.close()
        return contour_vertices, contour_colors

    def _build_contour_mesh(self) -> Mesh:

        s = self.spec
        n = int(s.resolution)
        xs = np.linspace(s.x_min, s.x_max, n, dtype=np.float32)
        ys = np.linspace(s.y_min, s.y_max, n, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys, indexing='xy')

        f = _safe_eval_func(s.expr)
        try:
            Z = f(X, Y)
        except Exception:
            Z = np.zeros_like(X, dtype=np.float32)

        verts_list, colors_list = self._generate_contours(X, Y, Z, levels=int(self.contour_levels), cmap=self.contour_cmap)

        # Flatten vertices/colors and make GL_LINES index pairs per segment
        if not verts_list:
            # Empty mesh: zero-count indexed draw is safe
            positions_arr = np.zeros((0, 3), dtype=np.float32)
            colors_arr = np.zeros((0, 3), dtype=np.float32)
            normals_arr = np.zeros((0, 3), dtype=np.float32)
            texcoords_arr = np.zeros((0, 2), dtype=np.float32)
            indices_arr = np.zeros((0,), dtype=np.uint32)
            return Mesh(positions_arr, colors_arr, normals_arr, texcoords_arr, indices_arr, primitive=GL.GL_LINES)

        positions = []
        colors = []
        indices = []
        base = 0
        for verts, cols in zip(verts_list, colors_list):
            count = int(verts.shape[0])
            if count < 2:
                continue
            positions.append(verts)
            colors.append(cols)
            # pairs: (0,1), (1,2), ... per segment
            for i in range(count - 1):
                indices.extend([base + i, base + i + 1])
            base += count

        if base == 0:
            positions_arr = np.zeros((0, 3), dtype=np.float32)
            colors_arr = np.zeros((0, 3), dtype=np.float32)
            normals_arr = np.zeros((0, 3), dtype=np.float32)
            texcoords_arr = np.zeros((0, 2), dtype=np.float32)
            indices_arr = np.zeros((0,), dtype=np.uint32)
            return Mesh(positions_arr, colors_arr, normals_arr, texcoords_arr, indices_arr, primitive=GL.GL_LINES)

        positions_arr = np.vstack(positions).astype(np.float32)
        colors_arr = np.vstack(colors).astype(np.float32)
        normals_arr = np.zeros_like(positions_arr, dtype=np.float32)
        texcoords_arr = np.zeros((positions_arr.shape[0], 2), dtype=np.float32)
        indices_arr = np.array(indices, dtype=np.uint32)

        return Mesh(positions_arr, colors_arr, normals_arr, texcoords_arr, indices_arr, primitive=GL.GL_LINES)

    # ---------------------------------------------------------- ball (sphere)
    def _build_ball_mesh(self) -> Mesh:
        # Build a unit-radius UV sphere; scale applied in model matrix
        p, c, n, t, indices = _build_uv_sphere(stacks=20, slices=36, radius=1.0)
        return Mesh(p, c, n, t, indices)

    def _get_time(self) -> float:
        try:
            import glfw
        except Exception:
            return 0.0
        return float(glfw.get_time()) if glfw.get_current_context() is not None else 0.0

    def _ball_xy_at_time(self, t: float) -> tuple[float, float]:
        s = self.spec
        omega = 2.0 * np.pi * float(self.ball_speed)
        cx = 0.5 * (s.x_min + s.x_max)
        cy = 0.5 * (s.y_min + s.y_max)
        rx = 0.4 * (s.x_max - s.x_min)
        ry = 0.4 * (s.y_max - s.y_min)
        x = cx + rx * np.cos(omega * t)
        y = cy + ry * np.sin(0.8 * omega * t)
        return float(x), float(y)

    def _ball_position(self) -> tuple[float, float, float]:
        # XY either animated or frozen at anchor
        if self.ball_freeze:
            x, y = float(self._ball_anchor[0]), float(self._ball_anchor[1])
        else:
            t = self._get_time()
            x, y = self._ball_xy_at_time(t)

        s = self.spec

        # Evaluate function
        f = _safe_eval_func(s.expr)
        try:
            z = float(f(np.array([x], dtype=np.float32),
                        np.array([y], dtype=np.float32))[0])
        except Exception:
            z = 0.0

        # ---- Compute normal using central differences ----
        eps = 1e-3 * max(s.x_max - s.x_min, s.y_max - s.y_min)
        try:
            z_dx1 = float(f(np.array([x + eps], dtype=np.float32),
                            np.array([y], dtype=np.float32))[0])
            z_dx0 = float(f(np.array([x - eps], dtype=np.float32),
                            np.array([y], dtype=np.float32))[0])
            z_dy1 = float(f(np.array([x], dtype=np.float32),
                            np.array([y + eps], dtype=np.float32))[0])
            z_dy0 = float(f(np.array([x], dtype=np.float32),
                            np.array([y - eps], dtype=np.float32))[0])
            dzdx = (z_dx1 - z_dx0) / (2 * eps)
            dzdy = (z_dy1 - z_dy0) / (2 * eps)
        except Exception:
            dzdx = dzdy = 0.0

        # Normal vector (not normalized yet)
        normal = np.array([-dzdx, -dzdy, 1.0], dtype=np.float32)
        normal /= np.linalg.norm(normal) + 1e-8

        # Offset the sphere center along the normal by ball_radius
        pos = np.array([x, y, z], dtype=np.float32) + normal * float(self.ball_radius)

        return float(pos[0]), float(pos[1]), float(pos[2])

    def _ball_model_matrix(self) -> np.ndarray:
        # Compose user transform with local transform to ball position, rotation and scale
        # Update position + spin based on either dynamics or parametric path
        t_now = self._get_time()
        if self._ball_last_time is None:
            self._ball_last_time = t_now
        dt = max(0.0, min(float(t_now - self._ball_last_time), 0.05))
        self._ball_last_time = t_now

        s = self.spec
        f = _safe_eval_func(s.expr)

        # Helper: finite difference slopes (compact form)
        def slopes(xf: float, yf: float) -> tuple[float, float, float]:
            def evalf(x: float, y: float) -> float:
                try:
                    return float(f(np.array([x], dtype=np.float32), np.array([y], dtype=np.float32))[0])
                except Exception:
                    return 0.0
            z0 = evalf(xf, yf)
            eps = 1e-3 * max(s.x_max - s.x_min, s.y_max - s.y_min)
            fx = (evalf(xf + eps, yf) - evalf(xf - eps, yf)) / (2 * eps)
            fy = (evalf(xf, yf + eps) - evalf(xf, yf - eps)) / (2 * eps)
            return z0, fx, fy

        # Determine XY position and velocity
        if self.ball_freeze:
            # Anchor-driven motion: derive velocity from anchor delta per frame
            x, y = float(self._ball_anchor[0]), float(self._ball_anchor[1])
            if self._freeze_prev_xy is None or dt <= 1e-6:
                vx = vy = 0.0
            else:
                dx = x - float(self._freeze_prev_xy[0])
                dy = y - float(self._freeze_prev_xy[1])
                inv_dt = 1.0 / max(dt, 1e-6)
                vx = dx * inv_dt
                vy = dy * inv_dt
            # Update previous after computing velocity
            self._freeze_prev_xy = np.array([x, y], dtype=np.float32)
        elif self.ball_dyn:
            # Dynamics integration on surface
            # Current state
            x, y = float(self.ball_pos[0]), float(self.ball_pos[1])
            z0, fx, fy = slopes(x, y)
            # Tangent basis and Gram matrix
            t1 = np.array([1.0, 0.0, fx], dtype=np.float32)
            t2 = np.array([0.0, 1.0, fy], dtype=np.float32)
            G00 = float(np.dot(t1, t1)); G01 = float(np.dot(t1, t2)); G11 = float(np.dot(t2, t2))
            detG = max(G00 * G11 - G01 * G01, 1e-8)
            invG00 =  G11 / detG
            invG01 = -G01 / detG
            invG11 =  G00 / detG
            # Gravity projected on tangent plane
            n = np.array([-fx, -fy, 1.0], dtype=np.float32)
            n /= (np.linalg.norm(n) + 1e-8)
            gvec = np.array([0.0, 0.0, -float(self.ball_gravity)], dtype=np.float32)
            g_t = gvec - n * float(np.dot(gvec, n))
            b0 = float(np.dot(t1, g_t)); b1 = float(np.dot(t2, g_t))
            ax = (invG00 * b0 + invG01 * b1) * (5.0 / 7.0)
            ay = (invG01 * b0 + invG11 * b1) * (5.0 / 7.0)
            # Integrate velocity with damping
            self.ball_vel[0] += ax * dt
            self.ball_vel[1] += ay * dt
            damp = max(0.0, min(1.0, float(self.ball_damping) * dt))
            self.ball_vel *= (1.0 - damp)
            # Integrate position and clamp to domain
            self.ball_pos[0] = np.clip(self.ball_pos[0] + self.ball_vel[0] * dt, s.x_min, s.x_max)
            self.ball_pos[1] = np.clip(self.ball_pos[1] + self.ball_vel[1] * dt, s.y_min, s.y_max)
            x, y = float(self.ball_pos[0]), float(self.ball_pos[1])
            vx, vy = float(self.ball_vel[0]), float(self.ball_vel[1])
        else:
            # Parametric mode removed: keep current position without induced spin
            x, y = float(self.ball_pos[0]), float(self.ball_pos[1])
            vx = vy = 0.0

        # Height and normal at current (x,y)
        z, fx, fy = slopes(x, y)
        n = np.array([-fx, -fy, 1.0], dtype=np.float32)
        n /= (np.linalg.norm(n) + 1e-8)

        # Tangential velocity and spin update
        v3 = np.array([vx, vy, 0.0], dtype=np.float32)
        v_t = v3 - n * float(np.dot(v3, n))
        speed_t = float(np.linalg.norm(v_t))
        if dt > 0.0 and speed_t > 1e-6:
            t_hat = v_t / speed_t
            # Use right-hand rule: flip axis to match rolling direction
            axis = np.cross(n, t_hat)
            an = float(np.linalg.norm(axis))
            if an > 1e-6:
                axis /= an
                dtheta_rad = (speed_t / max(self.ball_radius, 1e-6)) * dt
                dtheta_deg = float(dtheta_rad * (180.0 / np.pi))
                self._ball_rot = (T.rotate(axis, dtheta_deg) @ self._ball_rot).astype(np.float32)

        # Place ball center along normal by radius to avoid intersection
        cx, cy, cz = x + n[0] * self.ball_radius, y + n[1] * self.ball_radius, z + n[2] * self.ball_radius

        user_model = self._model_matrix()
        local = T.translate(cx, cy, cz) @ self._ball_rot @ T.scale(self.ball_radius)
        return (user_model @ local).astype(np.float32)
