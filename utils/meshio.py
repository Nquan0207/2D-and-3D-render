from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


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


def load_obj(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = Path(path)
    verts: list[tuple[float, float, float]] = []
    norms: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[list[str]] = []

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
            elif tag == "f" and len(rest) >= 3:
                faces.append(rest)

    pos: list[tuple[float, float, float]] = []
    nor: list[tuple[float, float, float]] = []
    idx: list[int] = []
    vmap: dict[tuple[int, int, int], int] = {}

    def get_index(triplet: str) -> int:
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
        key = (vi, ti, ni)
        if key in vmap:
            return vmap[key]
        p3 = verts[vi - 1]
        pos.append(p3)
        if ni > 0 and len(norms) >= ni:
            n3 = norms[ni - 1]
        else:
            n3 = (0.0, 0.0, 0.0)
        nor.append(n3)
        new_i = len(pos) - 1
        vmap[key] = new_i
        return new_i

    for f in faces:
        if len(f) == 3:
            tri = [get_index(tok) for tok in f]
            idx.extend(tri)
        else:
            verts_idx = [get_index(tok) for tok in f]
            for a, b, c in _triangulate_face(verts_idx):
                idx.extend([a, b, c])

    positions = np.asarray(pos, dtype=np.float32)
    normals = np.asarray(nor, dtype=np.float32)
    indices = np.asarray(idx, dtype=np.uint32)
    if not normals.any():
        normals = _compute_vertex_normals(positions, indices)
    colors = np.tile(np.array([0.85, 0.85, 0.85], dtype=np.float32), (positions.shape[0], 1))
    return positions, colors, normals, indices


def load_ply(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        header: list[str] = []
        for line in f:
            header.append(line.rstrip())
            if line.strip() == "end_header":
                break
        n_verts = 0
        n_faces = 0
        v_props: list[str] = []
        reading_vertex_props = False
        for h in header:
            parts = h.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                n_verts = int(parts[2]); reading_vertex_props = True
            elif len(parts) >= 3 and parts[0] == "element" and parts[1] == "face":
                n_faces = int(parts[2]); reading_vertex_props = False
            elif len(parts) >= 3 and parts[0] == "property" and reading_vertex_props:
                v_props.append(parts[-1])

        verts: list[tuple[float, float, float]] = []
        colors: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        for _ in range(n_verts):
            vals = f.readline().strip().split()
            vx = float(vals[v_props.index("x")]) if "x" in v_props else 0.0
            vy = float(vals[v_props.index("y")]) if "y" in v_props else 0.0
            vz = float(vals[v_props.index("z")]) if "z" in v_props else 0.0
            verts.append((vx, vy, vz))
            if all(k in v_props for k in ("nx", "ny", "nz")):
                nx = float(vals[v_props.index("nx")])
                ny = float(vals[v_props.index("ny")])
                nz = float(vals[v_props.index("nz")])
                normals.append((nx, ny, nz))
            if all(k in v_props for k in ("red", "green", "blue")):
                r = float(vals[v_props.index("red")]) / 255.0
                g = float(vals[v_props.index("green")]) / 255.0
                b = float(vals[v_props.index("blue")]) / 255.0
                colors.append((r, g, b))

        indices: list[int] = []
        for _ in range(n_faces):
            parts = f.readline().strip().split()
            if not parts:
                continue
            cnt = int(parts[0])
            face_idx = [int(x) for x in parts[1:1 + cnt]]
            for a, b, c in _triangulate_face(face_idx):
                indices.extend([a, b, c])

    positions = np.asarray(verts, dtype=np.float32)
    indices_np = np.asarray(indices, dtype=np.uint32)
    if normals:
        normals_np = _normalize_rows(np.asarray(normals, dtype=np.float32))
    else:
        normals_np = _compute_vertex_normals(positions, indices_np)
    if colors:
        colors_np = np.asarray(colors, dtype=np.float32)
        if colors_np.shape[0] != positions.shape[0]:
            colors_np = np.tile(np.array([0.85, 0.85, 0.85], dtype=np.float32), (positions.shape[0], 1))
    else:
        colors_np = np.tile(np.array([0.85, 0.85, 0.85], dtype=np.float32), (positions.shape[0], 1))
    return positions, colors_np, normals_np, indices_np


def load_mesh(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ext = str(path).lower().split('.')[-1]
    if ext == 'obj':
        return load_obj(path)
    if ext == 'ply':
        return load_ply(path)
    raise ValueError(f"Unsupported mesh format: {path}")

