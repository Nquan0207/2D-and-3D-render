from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D


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
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    resolution: int = 64


class SurfaceScene(BaseScene3D):
    def __init__(self, renderer: Renderer) -> None:
        super().__init__(renderer, axes_scale=1.2)
        self.spec = SurfaceSpec()
        self.mesh = self._build_mesh()

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
        return True, "Surface updated."

    def set_resolution(self, n: int) -> None:
        n = int(max(4, min(512, n)))
        if n != self.spec.resolution:
            self.spec.resolution = n
            self.mesh = self._build_mesh()

    def set_ranges(self, x_min: float | None = None, x_max: float | None = None,
                   y_min: float | None = None, y_max: float | None = None) -> None:
        if x_min is not None:
            self.spec.x_min = float(x_min)
        if x_max is not None:
            self.spec.x_max = float(x_max)
        if y_min is not None:
            self.spec.y_min = float(y_min)
        if y_max is not None:
            self.spec.y_max = float(y_max)
        self.mesh = self._build_mesh()

    # ---------------------------------------------------------------- drawing
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
