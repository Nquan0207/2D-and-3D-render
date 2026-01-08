from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..libs import transform as T
from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..scene_base import BaseScene3D
from .sphere_scene import _build_uv_sphere
from .cylinder_scene import _build_cylinder


@dataclass
class AtomDef:
    element: str
    position: Tuple[float, float, float]


@dataclass
class MoleculeDef:
    name: str
    atoms: List[AtomDef]
    bonds: List[Tuple[int, int]]


# ELEMENT_COLORS: Dict[str, np.ndarray] = {
#     "H": np.array([0.95, 0.1, 0.1], dtype=np.float32),
#     "O": np.array([0.95, 0.1, 0.1], dtype=np.float32),
#     "C": np.array([0.95, 0.1, 0.1], dtype=np.float32),
#     "N": np.array([0.95, 0.1, 0.1], dtype=np.float32),
# }

ELEMENT_COLORS = {
    "H": np.array([0.1, 0.9, 0.1], dtype=np.float32),
    "O": np.array([0.1, 0.1, 0.9], dtype=np.float32),
    "C": np.array([0.9, 0.1, 0.1], dtype=np.float32),
    "N": np.array([0.9, 0.1, 0.1], dtype=np.float32),
}

ELEMENT_RADII: Dict[str, float] = {
    "H": 0.6,
    "O": 1.0,
    "C": 0.9,
    "N": 0.85,
}

PHONG_BOND_COLOR = np.array([0.85, 0.85, 0.85], dtype=np.float32)


def _default_color(element: str) -> np.ndarray:
    return ELEMENT_COLORS.get(element, np.array([0.7, 0.7, 0.7], dtype=np.float32))


def _default_radius(element: str) -> float:
    return ELEMENT_RADII.get(element, 0.8)


def _water_spec() -> MoleculeDef:
    bond_angle = np.radians(104.5)
    distance = 1.2
    hx = np.sin(bond_angle * 0.5) * distance
    hz = np.cos(bond_angle * 0.5) * distance
    atoms = [
        AtomDef("O", (0.0, 0.0, 0.0)),
        AtomDef("H", (hx, 0.0, hz)),
        AtomDef("H", (-hx, 0.0, hz)),
    ]
    bonds = [(0, 1), (0, 2)]
    return MoleculeDef("Water (H2O)", atoms, bonds)


def _co2_spec() -> MoleculeDef:
    offset = 1.6
    atoms = [
        AtomDef("O", (-offset, 0.0, 0.0)),
        AtomDef("C", (0.0, 0.0, 0.0)),
        AtomDef("O", (offset, 0.0, 0.0)),
    ]
    bonds = [(0, 1), (1, 2)]
    return MoleculeDef("Carbon Dioxide (CO2)", atoms, bonds)


def _methane_spec() -> MoleculeDef:
    # Tetrahedral CH4
    r = 1.1
    atoms = [
        AtomDef("C", (0.0, 0.0, 0.0)),
        AtomDef("H", (r, r, r)),
        AtomDef("H", (-r, -r, r)),
        AtomDef("H", (-r, r, -r)),
        AtomDef("H", (r, -r, -r)),
    ]
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
    return MoleculeDef("Methane (CH4)", atoms, bonds)


MOLECULE_LIBRARY: List[MoleculeDef] = [
    _water_spec(),
    _co2_spec(),
    _methane_spec(),
]


def _uniform_color_mesh(mesh_data, color: np.ndarray) -> Mesh:
    positions, _, normals, texcoords, indices = mesh_data
    uniform_colors = np.tile(color.reshape(1, 3), (positions.shape[0], 1)).astype(np.float32)
    return Mesh(positions, uniform_colors, normals, texcoords, indices)


class MoleculeScene(BaseScene3D):
    def __init__(self, renderer: Renderer) -> None:
        super().__init__(renderer, axes_scale=1.6)
        sp = _build_uv_sphere(stacks=24, slices=48, radius=1.0)
        cyl = _build_cylinder(slices=36, radius=1.0, height=1.0)
        self._sphere_mesh = Mesh(*sp)
        self._cylinder_mesh = Mesh(*cyl)
        self._cylinder_mesh_phong = _uniform_color_mesh(cyl, PHONG_BOND_COLOR)
        self._sphere_mesh_data = sp
        self._phong_sphere_cache: Dict[str, Mesh] = {}
        self._phong_sphere_cache: Dict[str, Mesh] = {}

        self.mode = RenderMode.PHONG
        self.light_pos = np.array([4.0, 3.5, 4.0], dtype=np.float32)
        self.light_ambient = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        self.light_diffuse = np.array([0.95, 0.95, 0.95], dtype=np.float32)
        self.light_specular = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.mat_ambient = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.mat_diffuse = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        self.mat_specular = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        self.shininess = 80.0

        self.atom_scale = 0.35
        self.bond_radius = 0.12
        self.bond_color = np.array([0.85, 0.85, 0.85], dtype=np.float32)

        self.molecule_index = 0
        self.molecule = MOLECULE_LIBRARY[self.molecule_index]

    # ---------------------------------------------------------------- config helpers
    def available_molecules(self) -> List[str]:
        return [mol.name for mol in MOLECULE_LIBRARY]

    def set_molecule(self, index: int) -> None:
        if not MOLECULE_LIBRARY:
            return
        index = int(max(0, min(len(MOLECULE_LIBRARY) - 1, index)))
        self.molecule_index = index
        self.molecule = MOLECULE_LIBRARY[index]

    def set_atom_scale(self, value: float) -> None:
        self.atom_scale = float(max(0.1, min(1.0, value)))

    def set_bond_radius(self, value: float) -> None:
        self.bond_radius = float(max(0.02, min(0.6, value)))

    # ----------------------------------------------------------------- drawing utils
    def _settings_for_color(self, color: np.ndarray) -> RenderSettings:
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

    def _phong_mesh_for_element(self, element: str) -> Mesh:
        if element in self._phong_sphere_cache:
            return self._phong_sphere_cache[element]
        color = _default_color(element)
        mesh = _uniform_color_mesh(self._sphere_mesh_data, color)
        self._phong_sphere_cache[element] = mesh
        return mesh

    def _align_y_axis(self, vec: np.ndarray) -> np.ndarray:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        direction = vec.astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm <= 1e-5:
            return np.identity(4, dtype=np.float32)
        direction /= norm
        dot = float(np.clip(np.dot(y_axis, direction), -1.0, 1.0))
        if np.isclose(dot, 1.0, atol=1e-5):
            return np.identity(4, dtype=np.float32)
        if np.isclose(dot, -1.0, atol=1e-5):
            return T.rotate((1.0, 0.0, 0.0), 180.0)
        axis = np.cross(y_axis, direction)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 1e-5:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            axis /= axis_norm
        angle = np.degrees(np.arccos(dot))
        return T.rotate(axis, angle)

    # ------------------------------------------------------------------ rendering
    def _draw_object(self, projection: np.ndarray, view: np.ndarray) -> None:
        if self.molecule is None:
            return
        model_base = self._model_matrix()
        atom_positions = []
        atom_radii = []
        cylinder_mesh = self._cylinder_mesh_phong if self.mode == RenderMode.PHONG else self._cylinder_mesh

        # Draw atoms as spheres
        for atom in self.molecule.atoms:
            pos = np.array(atom.position, dtype=np.float32)
            atom_positions.append(pos)
            radius = self.atom_scale * _default_radius(atom.element)
            atom_radii.append(radius)
            atom_model = model_base @ T.translate(*pos) @ T.scale(radius)
            atom_norm = np.linalg.inv(atom_model[:3, :3]).T.astype(np.float32)
            color = _default_color(atom.element)
            sphere_mesh = (
                self._phong_mesh_for_element(atom.element)
                if self.mode == RenderMode.PHONG
                else self._sphere_mesh
            )
            self.renderer.draw(
                sphere_mesh,
                projection,
                view,
                atom_model,
                atom_norm,
                self._settings_for_color(color),
            )

        # Draw bonds
        for start_idx, end_idx in self.molecule.bonds:
            if start_idx >= len(atom_positions) or end_idx >= len(atom_positions):
                continue
            start = atom_positions[start_idx]
            end = atom_positions[end_idx]
            vec = end - start
            length = float(np.linalg.norm(vec))
            if length <= 1e-5:
                continue
            direction = vec / length
            trimmed_start = start + direction * atom_radii[start_idx]
            trimmed_end = end - direction * atom_radii[end_idx]
            trimmed_vec = trimmed_end - trimmed_start
            trimmed_len = float(np.linalg.norm(trimmed_vec))
            if trimmed_len <= 1e-5:
                continue
            mid = 0.5 * (trimmed_start + trimmed_end)
            align = self._align_y_axis(trimmed_vec)
            scale = T.scale(
                [
                    self.bond_radius,
                    trimmed_len,
                    self.bond_radius,
                ]
            )
            bond_model = model_base @ T.translate(*mid) @ align @ scale
            bond_norm = np.linalg.inv(bond_model[:3, :3]).T.astype(np.float32)
            bond_color = PHONG_BOND_COLOR if self.mode == RenderMode.PHONG else self.bond_color
            self.renderer.draw(
                cylinder_mesh,
                projection,
                view,
                bond_model,
                bond_norm,
                self._settings_for_color(bond_color),
            )
