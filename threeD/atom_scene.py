from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import OpenGL.GL as GL

from ..libs import transform as T
from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D
from .sphere_scene import _build_uv_sphere


def _build_orbit_ring(segments: int = 96) -> Mesh:
    segments = max(16, int(segments))
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
    x = np.cos(theta)
    z = np.sin(theta)
    positions = np.stack([x, np.zeros_like(x), z], axis=-1).astype(np.float32)
    colors = np.ones_like(positions, dtype=np.float32)
    normals = np.zeros_like(positions, dtype=np.float32)
    texcoords = np.zeros((segments, 2), dtype=np.float32)
    return Mesh(
        positions,
        colors,
        normals,
        texcoords,
        primitive=GL.GL_LINE_LOOP,
    )


@dataclass
class AtomPreset:
    shells: List[int]
    nucleus_color: np.ndarray
    electron_color: np.ndarray


class BohrAtomScene(BaseScene3D):
    """Render simple atoms with Bohr-style shells and animated electrons."""

    ATOM_LIBRARY: Dict[str, AtomPreset] = {
        "Hydrogen": AtomPreset(
            shells=[1],
            nucleus_color=np.array([0.9, 0.2, 0.25], dtype=np.float32),
            electron_color=np.array([0.9, 0.95, 1.0], dtype=np.float32),
        ),
        "Helium": AtomPreset(
            shells=[2],
            nucleus_color=np.array([0.95, 0.5, 0.3], dtype=np.float32),
            electron_color=np.array([0.8, 0.95, 1.0], dtype=np.float32),
        ),
        "Carbon": AtomPreset(
            shells=[2, 4],
            nucleus_color=np.array([0.4, 0.4, 0.4], dtype=np.float32),
            electron_color=np.array([0.25, 0.8, 1.0], dtype=np.float32),
        ),
        "Neon": AtomPreset(
            shells=[2, 8],
            nucleus_color=np.array([0.95, 0.6, 0.2], dtype=np.float32),
            electron_color=np.array([0.6, 0.85, 1.0], dtype=np.float32),
        ),
        "Sodium": AtomPreset(
            shells=[2, 8, 1],
            nucleus_color=np.array([0.95, 0.7, 0.35], dtype=np.float32),
            electron_color=np.array([0.35, 0.9, 1.0], dtype=np.float32),
        ),
    }

    def __init__(self, renderer: Renderer) -> None:
        super().__init__(renderer, axes_scale=1.4)
        positions, colors, normals, texcoords, indices = _build_uv_sphere(
            stacks=28,
            slices=48,
            radius=1.0,
        )
        self._sphere_mesh = Mesh(positions, colors, normals, texcoords, indices)
        self._orbit_mesh = _build_orbit_ring()

        self.mode = RenderMode.PHONG
        self.light_pos = np.array([3.0, 3.2, 3.5], dtype=np.float32)
        self.light_ambient = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        self.light_diffuse = np.array([0.95, 0.95, 0.95], dtype=np.float32)
        self.light_specular = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.mat_ambient = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.mat_diffuse = np.array([0.85, 0.85, 0.85], dtype=np.float32)
        self.mat_specular = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        self.shininess = 64.0

        self.atom_name = "Hydrogen"
        self.shell_spacing = 0.9
        self.nucleus_radius = 0.35
        self.electron_radius = 0.08
        self.electron_speed = 1.3
        self.show_orbits = True
        self.orbit_color = np.array([0.3, 0.65, 1.0], dtype=np.float32)
        self._tilt_angles = [0.0, 18.0, -26.0, 35.0]

        self._shell_config = self.ATOM_LIBRARY[self.atom_name]
        self._update_colors()

    # ------------------------------------------------------------------ configuration
    def available_atoms(self) -> List[str]:
        return list(self.ATOM_LIBRARY.keys())

    def set_atom(self, name: str) -> None:
        if name not in self.ATOM_LIBRARY:
            return
        self.atom_name = name
        self._shell_config = self.ATOM_LIBRARY[name]
        self._update_colors()

    def set_shell_spacing(self, value: float) -> None:
        self.shell_spacing = float(max(0.3, min(2.5, value)))

    def set_electron_speed(self, value: float) -> None:
        self.electron_speed = float(max(0.05, min(4.0, value)))

    def set_show_orbits(self, show: bool) -> None:
        self.show_orbits = bool(show)

    def _update_colors(self) -> None:
        preset = self._shell_config
        self.nucleus_color = preset.nucleus_color.copy()
        self.electron_color = preset.electron_color.copy()

    # ------------------------------------------------------------------ helpers
    def _shell_radius(self, shell_idx: int) -> float:
        return self.shell_spacing * float(shell_idx + 1)

    def _shell_tilt(self, shell_idx: int) -> np.ndarray:
        tilt = self._tilt_angles[shell_idx % len(self._tilt_angles)]
        return T.rotate((1.0, 0.0, 0.0), tilt)

    def _get_time(self) -> float:
        try:
            import glfw
        except Exception:
            return 0.0
        if glfw.get_current_context() is None:
            return 0.0
        return float(glfw.get_time())

    def _build_settings(self, color: np.ndarray) -> RenderSettings:
        specular_light = (
            self.light_specular if self.mode == RenderMode.PHONG else np.zeros_like(self.light_specular)
        )
        specular_mat = (
            self.mat_specular if self.mode == RenderMode.PHONG else np.zeros_like(self.mat_specular)
        )
        return RenderSettings(
            mode=self.mode,
            flat_color=color.astype(np.float32),
            light_pos=self.light_pos.astype(np.float32),
            shininess=self.shininess,
            light_ambient=self.light_ambient.astype(np.float32),
            light_diffuse=self.light_diffuse.astype(np.float32),
            light_specular=specular_light.astype(np.float32),
            mat_ambient=(color * self.mat_ambient).astype(np.float32),
            mat_diffuse=(color * self.mat_diffuse).astype(np.float32),
            mat_specular=specular_mat.astype(np.float32),
        )

    # ------------------------------------------------------------------ drawing
    def _draw_object(self, projection: np.ndarray, view: np.ndarray) -> None:
        model_base = self._model_matrix()

        # Draw nucleus
        nucleus_model = model_base @ T.scale(self.nucleus_radius)
        nucleus_norm = np.linalg.inv(nucleus_model[:3, :3]).T.astype(np.float32)
        self.renderer.draw(
            self._sphere_mesh,
            projection,
            view,
            nucleus_model,
            nucleus_norm,
            self._build_settings(self.nucleus_color),
        )

        shells = self._shell_config.shells
        current_time = self._get_time()

        # Orbit guides
        if self.show_orbits:
            orbit_settings = RenderSettings(
                mode=RenderMode.FLAT,
                flat_color=self.orbit_color.astype(np.float32),
                light_pos=self.light_pos.astype(np.float32),
                shininess=1.0,
                light_ambient=self.light_ambient.astype(np.float32),
                light_diffuse=self.light_diffuse.astype(np.float32),
                light_specular=np.zeros_like(self.light_specular, dtype=np.float32),
                mat_ambient=self.orbit_color.astype(np.float32),
                mat_diffuse=self.orbit_color.astype(np.float32),
                mat_specular=np.zeros(3, dtype=np.float32),
            )
            for shell_idx in range(len(shells)):
                radius = self._shell_radius(shell_idx)
                orbit_model = model_base @ self._shell_tilt(shell_idx) @ T.scale(radius)
                orbit_norm = np.linalg.inv(orbit_model[:3, :3]).T.astype(np.float32)
                self.renderer.draw(
                    self._orbit_mesh,
                    projection,
                    view,
                    orbit_model,
                    orbit_norm,
                    orbit_settings,
                )

        # Electrons
        for shell_idx, electron_count in enumerate(shells):
            if electron_count <= 0:
                continue
            radius = self._shell_radius(shell_idx)
            speed = self.electron_speed * (1.0 + 0.25 * shell_idx)
            tilt = self._shell_tilt(shell_idx)
            for electron in range(electron_count):
                offset = (2.0 * np.pi * electron) / float(electron_count)
                angle = offset + speed * current_time
                local_translate = T.translate(
                    radius * np.cos(angle),
                    0.0,
                    radius * np.sin(angle),
                )
                electron_model = model_base @ tilt @ local_translate @ T.scale(self.electron_radius)
                electron_norm = np.linalg.inv(electron_model[:3, :3]).T.astype(np.float32)
                self.renderer.draw(
                    self._sphere_mesh,
                    projection,
                    view,
                    electron_model,
                    electron_norm,
                    self._build_settings(self.electron_color),
                )
