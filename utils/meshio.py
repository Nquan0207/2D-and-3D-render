from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow optional
    Image = None


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _compute_vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    v = positions
    idx = indices.reshape(-1, 3)
    normals = np.zeros_like(v, dtype=np.float32)
    for a, b, c in idx:
        ab = v[b] - v[a]
        ac = v[c] - v[a]
        fn = np.cross(ab, ac)
        normals[a] += fn
        normals[b] += fn
        normals[c] += fn
    return _normalize_rows(normals)


def _triangulate_face(face: list[int]) -> list[tuple[int, int, int]]:
    tris: list[tuple[int, int, int]] = []
    for i in range(1, len(face) - 1):
        tris.append((face[0], face[i], face[i + 1]))
    return tris




def _triangles_from_strips(strip_indices: list[int]) -> list[tuple[int, int, int]]:
    tris: list[tuple[int, int, int]] = []
    window: list[int] = []
    flip = False
    for idx in strip_indices:
        if idx < 0:
            window.clear()
            flip = False
            continue
        window.append(idx)
        if len(window) < 3:
            continue
        a, b, c = window[-3], window[-2], window[-1]
        if flip:
            tri = (b, a, c)
        else:
            tri = (a, b, c)
        flip = not flip
        if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
            continue
        tris.append(tri)
    return tris


def _average_texture_color(path: Path) -> tuple[float, float, float] | None:
    if Image is None:
        print(f"[meshio] Pillow not available, skip texture {path}")
        return None
    if not path.exists():
        print(f"[meshio] Texture file not found: {path}")
        return None
    try:
        with Image.open(path) as img:
            data = np.asarray(img.convert("RGB"), dtype=np.float32)
    except OSError:
        print(f"[meshio] Failed to open texture: {path}")
        return None
    if data.size == 0:
        print(f"[meshio] Empty texture: {path}")
        return None
    mean = data.mean(axis=(0, 1)) / 255.0
    return float(mean[0]), float(mean[1]), float(mean[2])


def _list_groups_and_counts(path: Path) -> Tuple[List[str], Dict[str, int]]:
    groups: Dict[str, int] = {}
    order: List[str] = []
    current = "default"
    groups[current] = 0
    order.append(current)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            tag, *rest = parts
            if tag in {"g", "o"} and rest:
                name = rest[0]
                if name not in groups:
                    groups[name] = 0
                    order.append(name)
                current = name
            elif tag == "f" and len(rest) >= 3:
                groups[current] = groups.get(current, 0) + 1
    existing = [name for name in order if groups.get(name, 0) > 0]
    if not existing:
        return ["default"], {"default": 0}
    return existing, groups


def list_obj_groups(path: str | Path) -> List[str]:
    p = Path(path)
    groups, _ = _list_groups_and_counts(p)
    return groups


def load_obj(path: str | Path, *, group: str | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = Path(path)
    default_color: tuple[float, float, float] = (0.85, 0.85, 0.85)
    verts: list[tuple[float, float, float]] = []
    norms: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    materials: Dict[str, tuple[float, float, float]] = {}
    current_mtl: str | None = None
    current_group = "default"
    faces_by_group: Dict[str, list[tuple[list[str], str | None]]] = defaultdict(list)

    def load_mtl(mtl_name: str) -> None:
        mtl_path = (p.parent / mtl_name).resolve()
        if not mtl_path.exists():
            return

        def flush(name: str | None, color: tuple[float, float, float] | None) -> None:
            if name and color is not None and name not in materials:
                materials[name] = color

        try:
            with mtl_path.open("r", encoding="utf-8", errors="ignore") as mtl_file:
                name: str | None = None
                pending_color: tuple[float, float, float] | None = None
                for line in mtl_file:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    tag, *rest = parts
                    tag_lower = tag.lower()
                    if tag_lower == "newmtl" and rest:
                        flush(name, pending_color)
                        name = rest[0]
                        pending_color = None
                    elif tag_lower == "kd" and len(rest) >= 3 and name is not None:
                        try:
                            r, g, b = float(rest[0]), float(rest[1]), float(rest[2])
                            pending_color = (r, g, b)
                        except ValueError:
                            continue
                    elif tag_lower == "map_kd" and rest and name is not None:
                        tex_rel = " ".join(rest).strip('"').replace("\\", "/")
                        color = _average_texture_color((p.parent / tex_rel).resolve())
                        if color is not None:
                            materials[name] = color
                flush(name, pending_color)
        except OSError:
            return

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            tag, *rest = parts
            if tag == "v" and len(rest) >= 3:
                verts.append((float(rest[0]), float(rest[1]), float(rest[2])))
            elif tag == "vn" and len(rest) >= 3:
                norms.append((float(rest[0]), float(rest[1]), float(rest[2])))
            elif tag == "vt" and len(rest) >= 2:
                uvs.append((float(rest[0]), float(rest[1])))
            elif tag in {"g", "o"} and rest:
                current_group = rest[0] or "default"
            elif tag == "f" and len(rest) >= 3:
                faces_by_group[current_group].append((rest, current_mtl))
            elif tag == "mtllib" and rest:
                mtl_name = " ".join(rest).strip('"')
                if mtl_name:
                    load_mtl(mtl_name)
            elif tag == "usemtl" and rest:
                current_mtl = rest[0]

    if group is not None:
        selected_faces = faces_by_group.get(group)
        if not selected_faces:
            raise ValueError(f"Group '{group}' not found in OBJ: {path}")
        faces_iterable = [selected_faces]
    else:
        faces_iterable = faces_by_group.values()

    pos: list[tuple[float, float, float]] = []
    nor: list[tuple[float, float, float]] = []
    col: list[tuple[float, float, float]] = []
    idx: list[int] = []
    vmap: dict[tuple[int, int, int, tuple[float, float, float]], int] = {}

    def get_index(triplet: str, color: tuple[float, float, float]) -> int:
        vi, ti, ni = -1, -1, -1
        parts = triplet.split("/")
        if len(parts) == 1:
            vi = int(parts[0])
        elif len(parts) == 2:
            vi = int(parts[0]); ti = int(parts[1]) if parts[1] else -1
        else:
            vi = int(parts[0]); ti = int(parts[1]) if parts[1] else -1; ni = int(parts[2]) if parts[2] else -1
        if vi < 0:
            vi = len(verts) + 1 + vi
        if ti < 0 and len(uvs) > 0:
            ti = len(uvs) + 1 + ti
        if ni < 0 and len(norms) > 0:
            ni = len(norms) + 1 + ni
        key = (vi, ti, ni, color)
        if key in vmap:
            return vmap[key]
        p3 = verts[vi - 1]
        pos.append(p3)
        if ni > 0 and len(norms) >= ni:
            n3 = norms[ni - 1]
        else:
            n3 = (0.0, 0.0, 0.0)
        nor.append(n3)
        col.append(color)
        new_i = len(pos) - 1
        vmap[key] = new_i
        return new_i

    for face_group in faces_iterable:
        for face, material_name in face_group:
            face_color = materials.get(material_name, default_color)
            if len(face) == 3:
                tri = [get_index(tok, face_color) for tok in face]
                idx.extend(tri)
            else:
                verts_idx = [get_index(tok, face_color) for tok in face]
                for a, b, c in _triangulate_face(verts_idx):
                    idx.extend([a, b, c])

    positions = np.asarray(pos, dtype=np.float32)
    normals = np.asarray(nor, dtype=np.float32)
    colors = np.asarray(col, dtype=np.float32) if col else np.tile(np.array(default_color, dtype=np.float32), (positions.shape[0], 1))
    indices = np.asarray(idx, dtype=np.uint32)
    if not normals.any():
        normals = _compute_vertex_normals(positions, indices)
    return positions, colors, normals, indices



_PLY_INT_TYPES = {
    "char",
    "uchar",
    "short",
    "ushort",
    "int",
    "uint",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
}


def load_ply(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        elements: list[tuple[str, int, list[dict[str, str]]]] = []
        current_props: list[dict[str, str]] | None = None
        while True:
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            token = parts[0]
            if token == "comment":
                continue
            if token == "element" and len(parts) >= 3:
                current_props = []
                elements.append((parts[1], int(parts[2]), current_props))
            elif token == "property" and current_props is not None:
                if len(parts) >= 5 and parts[1] == "list":
                    current_props.append(
                        {
                            "kind": "list",
                            "name": parts[-1],
                            "count_type": parts[2],
                            "item_type": parts[3],
                        }
                    )
                elif len(parts) >= 3:
                    current_props.append(
                        {
                            "kind": "scalar",
                            "name": parts[-1],
                            "type": parts[1],
                        }
                    )
            elif token == "end_header":
                break

        verts: list[tuple[float, float, float]] = []
        colors: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        indices: list[int] = []

        for name, count, prop_list in elements:
            if name == "vertex":
                prop_meta = {prop["name"]: prop for prop in prop_list if prop.get("kind") == "scalar"}
                has_normals = all(k in prop_meta for k in ("nx", "ny", "nz"))
                color_names = ("red", "green", "blue")
                has_colors = all(k in prop_meta for k in color_names)
                color_scale = 255.0 if has_colors and all(
                    prop_meta[k]["type"].lower() in _PLY_INT_TYPES for k in color_names
                ) else 1.0
                for _ in range(count):
                    line = f.readline()
                    if not line:
                        break
                    vals = line.strip().split()
                    if not vals:
                        continue
                    cursor = 0
                    value_map: dict[str, float] = {}
                    valid = True
                    for prop in prop_list:
                        if prop["kind"] == "scalar":
                            if cursor >= len(vals):
                                valid = False
                                break
                            try:
                                value_map[prop["name"]] = float(vals[cursor])
                            except ValueError:
                                value_map[prop["name"]] = 0.0
                            cursor += 1
                        else:
                            if cursor >= len(vals):
                                valid = False
                                break
                            cnt = int(vals[cursor])
                            cursor += 1
                            if cursor + cnt > len(vals):
                                valid = False
                                break
                            cursor += cnt
                    if not valid:
                        continue
                    vx = float(value_map.get("x", 0.0))
                    vy = float(value_map.get("y", 0.0))
                    vz = float(value_map.get("z", 0.0))
                    verts.append((vx, vy, vz))
                    if has_normals:
                        nx = float(value_map.get("nx", 0.0))
                        ny = float(value_map.get("ny", 0.0))
                        nz = float(value_map.get("nz", 0.0))
                        normals.append((nx, ny, nz))
                    if has_colors:
                        r = float(value_map.get("red", 0.0))
                        g = float(value_map.get("green", 0.0))
                        b = float(value_map.get("blue", 0.0))
                        if color_scale != 1.0:
                            r /= color_scale
                            g /= color_scale
                            b /= color_scale
                        colors.append((r, g, b))
            elif name == "face":
                for _ in range(count):
                    line = f.readline()
                    if not line:
                        break
                    vals = line.strip().split()
                    if not vals:
                        continue
                    cursor = 0
                    face_idx: list[int] | None = None
                    valid = True
                    for prop in prop_list:
                        if prop["kind"] == "scalar":
                            cursor += 1
                        else:
                            if cursor >= len(vals):
                                valid = False
                                break
                            cnt = int(vals[cursor])
                            cursor += 1
                            if cursor + cnt > len(vals):
                                valid = False
                                break
                            data = [int(v) for v in vals[cursor:cursor + cnt]]
                            cursor += cnt
                            if prop["name"] == "vertex_indices":
                                face_idx = data
                    if not valid or not face_idx:
                        continue
                    if len(face_idx) == 3:
                        indices.extend(face_idx)
                    else:
                        for a, b, c in _triangulate_face(face_idx):
                            indices.extend([a, b, c])
            elif name == "tristrips":
                for _ in range(count):
                    line = f.readline()
                    if not line:
                        break
                    vals = line.strip().split()
                    if not vals:
                        continue
                    cursor = 0
                    strip_idx: list[int] | None = None
                    valid = True
                    for prop in prop_list:
                        if prop["kind"] == "scalar":
                            cursor += 1
                        else:
                            if cursor >= len(vals):
                                valid = False
                                break
                            cnt = int(vals[cursor])
                            cursor += 1
                            if cursor + cnt > len(vals):
                                valid = False
                                break
                            data = [int(v) for v in vals[cursor:cursor + cnt]]
                            cursor += cnt
                            if prop["name"] == "vertex_indices":
                                strip_idx = data
                    if not valid or not strip_idx:
                        continue
                    for a, b, c in _triangles_from_strips(strip_idx):
                        indices.extend([a, b, c])
            else:
                for _ in range(count):
                    if not f.readline():
                        break

    positions = np.asarray(verts, dtype=np.float32)
    indices_np = np.asarray(indices, dtype=np.uint32)
    if normals and len(normals) == len(verts):
        normals_np = _normalize_rows(np.asarray(normals, dtype=np.float32))
    elif indices_np.size > 0:
        normals_np = _compute_vertex_normals(positions, indices_np)
    else:
        normals_np = np.zeros_like(positions, dtype=np.float32)
    if colors and len(colors) == len(verts):
        colors_np = np.asarray(colors, dtype=np.float32)
    else:
        colors_np = np.tile(np.array([0.85, 0.85, 0.85], dtype=np.float32), (positions.shape[0], 1))
    return positions, colors_np, normals_np, indices_np

def load_mesh(path: str | Path, *, group: str | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ext = str(path).lower().split('.')[-1]
    if ext == 'obj':
        return load_obj(path, group=group)
    if ext == 'ply':
        return load_ply(path)
    raise ValueError(f"Unsupported mesh format: {path}")
