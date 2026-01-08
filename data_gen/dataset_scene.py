from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import OpenGL.GL as GL

from ..libs import transform as T
from ..rendering import Mesh, RenderMode, RenderSettings, Renderer
from ..utils import meshio
from .render_target import RenderTarget


@dataclass(slots=True)
class AssetSpec:
    """Description for a mesh asset that can be instantiated in the dataset scene."""

    name: str
    path: str
    class_id: int = 0
    scale_range: Tuple[float, float] = (1.0, 1.0)
    normalize: bool = True
    group: str | None = None
    split_groups: bool = False
    approx_radius: float = 0.5


@dataclass(slots=True)
class InstanceSpec:
    """Runtime instance information for a rendered object."""

    asset: str
    instance_id: int
    class_id: int
    model_matrix: np.ndarray
    color: np.ndarray | None = None
    mode: RenderMode = RenderMode.PHONG
    shininess: float = 48.0


def compose_model_matrix(
    translation: Sequence[float] = (0.0, 0.0, 0.0),
    rotation_deg: Sequence[float] = (0.0, 0.0, 0.0),
    scale: float | Sequence[float] = 1.0,
) -> np.ndarray:
    """Utility to build a model matrix from T/R/S parameters."""

    translate = T.translate(*translation)
    rot_x = T.rotate((1.0, 0.0, 0.0), float(rotation_deg[0]))
    rot_y = T.rotate((0.0, 1.0, 0.0), float(rotation_deg[1]))
    rot_z = T.rotate((0.0, 0.0, 1.0), float(rotation_deg[2]))
    if isinstance(scale, Sequence):
        scale_mat = T.scale(scale[0], scale[1], scale[2])
    else:
        scale_mat = T.scale(float(scale))
    return (translate @ rot_z @ rot_y @ rot_x @ scale_mat).astype(np.float32)


class MeshLibrary:
    """Small cache that loads meshes once and reuses them across frames."""

    def __init__(self, specs: Iterable[AssetSpec]) -> None:
        self._meshes: Dict[str, Mesh] = {}
        self._asset_meshes: Dict[str, List[str]] = {}
        for spec in specs:
            self.register(spec)

    @staticmethod
    def _compute_normalization(positions: np.ndarray, normalize: bool) -> tuple[np.ndarray | None, float | None]:
        if not normalize or positions.size == 0:
            return None, None
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        center = (mins + maxs) * 0.5
        extent = maxs - mins
        max_dim = float(np.max(extent))
        if max_dim > 1e-6:
            scale = 1.0 / max_dim
        else:
            scale = None
        return center.astype(np.float32), scale

    @staticmethod
    def _apply_normalization(positions: np.ndarray, center: np.ndarray | None, scale: float | None) -> np.ndarray:
        if center is None:
            return positions
        shifted = positions - center
        if scale is not None:
            shifted = shifted * scale
        return shifted

    @staticmethod
    def _approximate_radius(positions: np.ndarray) -> float:
        if positions.size == 0:
            return 0.5
        radii = np.linalg.norm(positions, axis=1)
        if radii.size == 0:
            return 0.5
        return float(radii.max())

    def mesh_names_for_asset(self, name: str) -> List[str]:
        if name in self._asset_meshes:
            return self._asset_meshes[name]
        return [name]

    def register(self, spec: AssetSpec) -> None:
        if spec.name in self._asset_meshes:
            return
        path_lower = spec.path.lower()
        split_allowed = spec.split_groups and path_lower.endswith(".obj")
        if split_allowed:
            positions_full, _, _, _ = meshio.load_mesh(spec.path)
            center, scale = self._compute_normalization(positions_full, spec.normalize)
            norm_positions = self._apply_normalization(positions_full, center, scale)
            spec.approx_radius = self._approximate_radius(norm_positions)
            groups = meshio.list_obj_groups(spec.path)
            if not groups:
                groups = ["default"]
            mesh_names: List[str] = []
            for group_name in groups:
                positions, colors, normals, indices = meshio.load_mesh(spec.path, group=group_name)
                positions = self._apply_normalization(positions, center, scale)
                texcoords = np.zeros((positions.shape[0], 2), dtype=np.float32)
                mesh_key = f"{spec.name}::{group_name}"
                self._meshes[mesh_key] = Mesh(positions, colors, normals, texcoords, indices)
                mesh_names.append(mesh_key)
            self._asset_meshes[spec.name] = mesh_names
        else:
            positions, colors, normals, indices = meshio.load_mesh(spec.path, group=spec.group)
            center, scale = self._compute_normalization(positions, spec.normalize)
            positions = self._apply_normalization(positions, center, scale)
            spec.approx_radius = self._approximate_radius(positions)
            texcoords = np.zeros((positions.shape[0], 2), dtype=np.float32)
            self._meshes[spec.name] = Mesh(positions, colors, normals, texcoords, indices)
            self._asset_meshes[spec.name] = [spec.name]

    def get(self, name: str) -> Mesh:
        if name not in self._meshes:
            raise KeyError(f"Mesh '{name}' is not registered in MeshLibrary.")
        return self._meshes[name]


class DatasetScene:
    """Light-weight scene that renders arbitrary meshes with configurable camera poses."""

    def __init__(
        self,
        renderer: Renderer,
        *,
        fov_y: float = 45.0,
        aspect: float = 1.0,
        near: float = 0.1,
        far: float = 100.0,
        assets: Iterable[AssetSpec] | None = None,
    ) -> None:
        self.renderer = renderer
        self.meshes = MeshLibrary(assets or [])
        self.instances: List[InstanceSpec] = []

        self._fov_y = fov_y
        self._near = near
        self._far = far
        self._projection = T.perspective(fov_y, aspect, near, far).astype(np.float32)
        default_eye = np.array((2.5, 2.5, 2.5), dtype=np.float32)
        self._view = T.lookat(default_eye, np.array((0.0, 0.0, 0.0), dtype=np.float32), np.array((0.0, 1.0, 0.0), dtype=np.float32)).astype(np.float32)

        self.light_pos = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        self.light_ambient = np.array([0.35, 0.35, 0.35], dtype=np.float32)
        self.light_diffuse = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.light_specular = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.mat_ambient = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.mat_diffuse = np.array([0.85, 0.85, 0.85], dtype=np.float32)
        self.mat_specular = np.array([0.6, 0.6, 0.6], dtype=np.float32)

    @property
    def projection(self) -> np.ndarray:
        return self._projection

    @property
    def view(self) -> np.ndarray:
        return self._view

    @property
    def near_plane(self) -> float:
        return self._near

    @property
    def far_plane(self) -> float:
        return self._far

    def set_camera_pose(self, eye: Sequence[float], target: Sequence[float], up: Sequence[float]) -> None:
        self._view = T.lookat(np.asarray(eye, dtype=np.float32), np.asarray(target, dtype=np.float32), np.asarray(up, dtype=np.float32)).astype(np.float32)

    def set_perspective(self, fov_y: float, aspect: float, near: float, far: float) -> None:
        self._fov_y = fov_y
        self._near = near
        self._far = far
        self._projection = T.perspective(fov_y, aspect, near, far).astype(np.float32)

    def resize(self, width: int, height: int) -> None:
        aspect = max(width, 1) / float(max(height, 1))
        self.set_perspective(self._fov_y, aspect, self._near, self._far)

    def register_assets(self, specs: Iterable[AssetSpec]) -> None:
        for spec in specs:
            self.meshes.register(spec)

    def set_instances(self, instances: Iterable[InstanceSpec]) -> None:
        self.instances = self._expand_instances(instances)

    def clear_instances(self) -> None:
        self.instances.clear()

    def _expand_instances(self, instances: Iterable[InstanceSpec]) -> List[InstanceSpec]:
        originals = list(instances)
        if not originals:
            return []
        expanded: List[InstanceSpec] = []
        used_ids: set[int] = set()
        base_ids = [inst.instance_id for inst in originals if inst.instance_id is not None]
        max_original_id = max(base_ids) if base_ids else 0
        next_id = max_original_id + 1
        for inst in originals:
            mesh_names = self.meshes.mesh_names_for_asset(inst.asset)
            if len(mesh_names) == 1 and mesh_names[0] == inst.asset:
                used_ids.add(inst.instance_id)
                expanded.append(
                    InstanceSpec(
                        asset=inst.asset,
                        instance_id=inst.instance_id,
                        class_id=inst.class_id,
                        model_matrix=inst.model_matrix,
                        color=inst.color,
                        mode=inst.mode,
                        shininess=inst.shininess,
                    )
                )
                continue
            first_mesh = True
            for mesh_name in mesh_names:
                if first_mesh and inst.instance_id not in used_ids:
                    assigned_id = inst.instance_id
                else:
                    while next_id in used_ids:
                        next_id += 1
                    assigned_id = next_id
                    next_id += 1
                used_ids.add(assigned_id)
                expanded.append(
                    InstanceSpec(
                        asset=mesh_name,
                        instance_id=assigned_id,
                        class_id=inst.class_id,
                        model_matrix=inst.model_matrix,
                        color=inst.color,
                        mode=inst.mode,
                        shininess=inst.shininess,
                    )
                )
                first_mesh = False
        return expanded

    def render(
        self,
        *,
        color_override: Optional[Callable[[InstanceSpec], np.ndarray]] = None,
        mode_override: Optional[RenderMode] = None,
    ) -> None:
        """Draw instances with optional overrides for color/mode."""

        projection = self._projection.astype(np.float32)
        view = self._view.astype(np.float32)
        for inst in self.instances:
            mesh = self.meshes.get(inst.asset)
            model = inst.model_matrix.astype(np.float32)
            normal_matrix = np.linalg.inv(model[:3, :3]).T.astype(np.float32)
            if color_override is not None:
                flat_color = color_override(inst)
            elif inst.color is not None:
                flat_color = inst.color.astype(np.float32)
            else:
                flat_color = self.mat_diffuse
            mode = mode_override if mode_override is not None else inst.mode
            specular_light = self.light_specular if mode == RenderMode.PHONG else np.zeros_like(self.light_specular)
            specular_mat = self.mat_specular if mode == RenderMode.PHONG else np.zeros_like(self.mat_specular)

            settings = RenderSettings(
                mode=mode,
                flat_color=flat_color,
                light_pos=self.light_pos,
                shininess=inst.shininess,
                light_ambient=self.light_ambient,
                light_diffuse=self.light_diffuse,
                light_specular=specular_light,
                mat_ambient=self.mat_ambient,
                mat_diffuse=self.mat_diffuse,
                mat_specular=specular_mat,
            )
            self.renderer.draw(mesh, projection, view, model, normal_matrix, settings)

    def render_into_target(self, target: RenderTarget) -> None:
        """Render RGB and segmentation mask into the provided target."""

        target.bind_color_pass()
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.render()

        target.bind_mask_pass()
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.render(
            color_override=lambda inst: encode_instance_color(inst.instance_id),
            mode_override=RenderMode.FLAT,
        )
        target.unbind()


def encode_instance_color(instance_id: int) -> np.ndarray:
    """Encode an integer instance id into an RGB color (0-1 floats)."""

    instance_id = max(0, int(instance_id)) & 0xFFFFFF
    r = (instance_id & 0xFF) / 255.0
    g = ((instance_id >> 8) & 0xFF) / 255.0
    b = ((instance_id >> 16) & 0xFF) / 255.0
    return np.array([r, g, b], dtype=np.float32)
