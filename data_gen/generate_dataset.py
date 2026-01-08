from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import glfw
import numpy as np
from PIL import Image, ImageDraw

from ..rendering import Renderer
from .annotations import CocoWriter, decode_instance_ids, extract_bboxes
from .dataset_scene import AssetSpec, DatasetScene, InstanceSpec, compose_model_matrix
from .render_target import RenderTarget


@dataclass
class OutputPaths:
    root: Path

    def rgb_dir(self) -> Path:
        path = self.root / "rgb"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def depth_dir(self) -> Path:
        path = self.root / "depth"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def mask_dir(self) -> Path:
        path = self.root / "mask"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def meta_dir(self) -> Path:
        path = self.root / "meta"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def bbox_dir(self) -> Path:
        path = self.root / "bbox"
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(config_path: Path) -> Dict:
    cfg = json.loads(config_path.read_text())
    return cfg


def parse_assets(cfg: Dict, config_path: Path) -> List[AssetSpec]:
    assets = cfg.get("assets", [])
    parsed: List[AssetSpec] = []
    for item in assets:
        parsed.append(
            AssetSpec(
                name=item["name"],
                path=str((config_path.parent / item["path"]).resolve()),
                class_id=int(item.get("class_id", 0)),
                scale_range=tuple(item.get("scale_range", [1.0, 1.0])),
                normalize=bool(item.get("normalize", True)),
                group=item.get("group"),
                split_groups=bool(item.get("split_groups", False)),
            )
        )
    return parsed


class YoloWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, file_stem: str, bboxes: Dict[int, tuple[int, int, int, int]], width: int, height: int, class_lookup: Dict[int, int]) -> None:
        lines: List[str] = []
        for inst_id, bbox in bboxes.items():
            cls = class_lookup.get(inst_id, 0)
            cx = (bbox[0] + bbox[2] / 2.0) / width
            cy = (bbox[1] + bbox[3] / 2.0) / height
            w = bbox[2] / width
            h = bbox[3] / height
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (self.output_dir / f"{file_stem}.txt").write_text("\n".join(lines))


def random_instances(assets: List[AssetSpec], count: int) -> List[InstanceSpec]:
    instances: List[InstanceSpec] = []
    placements: list[tuple[np.ndarray, float]] = []  # (center, radius)
    spawn_radius = 0.4
    base_height = -0.05
    scale_factor = 1.3
    padding = 0.1
    for idx in range(count):
        placed = False
        attempts = 0
        while not placed and attempts < 60:
            attempts += 1
            asset = np.random.choice(assets)
            sx, sy = asset.scale_range
            scale = np.random.uniform(sx, sy) * scale_factor
            tx = np.random.uniform(-spawn_radius, spawn_radius)
            ty = np.random.uniform(base_height, base_height + 0.25)
            tz = np.random.uniform(-spawn_radius, spawn_radius)
            center = np.array((tx, ty, tz), dtype=np.float32)
            radius = (asset.approx_radius or 0.5) * scale
            if any(np.linalg.norm(center - prev_center) < (radius + prev_radius + padding) for prev_center, prev_radius in placements):
                continue
            rot = (0.0, np.random.uniform(0.0, 360.0), 0.0)
            model = compose_model_matrix(center, rot, scale)
            instances.append(
                InstanceSpec(
                    asset=asset.name,
                    instance_id=idx + 1,
                    class_id=asset.class_id,
                    model_matrix=model,
                )
            )
            placements.append((center, radius))
            placed = True
        if not placed:
            break
    return instances


def save_outputs(
    idx: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask_ids: np.ndarray,
    output: OutputPaths,
) -> None:
    rgb_path = output.rgb_dir() / f"{idx:06d}.png"
    depth_path = output.depth_dir() / f"{idx:06d}.npy"
    mask_path = output.mask_dir() / f"{idx:06d}.npy"
    mask_png_path = output.mask_dir() / f"{idx:06d}.png"

    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_png = (depth_norm * 65535).astype(np.uint16)
    Image.fromarray(depth_png, mode="I;16").save(output.depth_dir() / f"{idx:06d}.png")
    np.save(mask_path, mask_ids)
    if mask_ids.max() > 0:
        mask_gray = np.clip((mask_ids.astype(np.float32) / mask_ids.max()) * 255.0, 0, 255).astype(np.uint8)
    else:
        mask_gray = np.zeros_like(mask_ids, dtype=np.uint8)
    Image.fromarray(mask_gray, mode="L").save(mask_png_path)


PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (67, 99, 216),
    (245, 130, 49),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (188, 246, 12),
    (0, 128, 128),
]


def class_color(class_id: int) -> tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]


def save_bbox_visualization(idx: int, rgb: np.ndarray, bboxes: Dict[int, tuple[int, int, int, int]], class_lookup: Dict[int, int], output: OutputPaths) -> None:
    if not bboxes:
        return
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for inst_id, bbox in bboxes.items():
        x, y, w, h = bbox
        class_id = class_lookup.get(inst_id, 0)
        color = class_color(class_id)
        rect = [x, y, x + w, y + h]
        draw.rectangle(rect, outline=color, width=2)
        label = f"{class_id}"
        text_pos = (x + 4, y + 4)
        draw.text(text_pos, label, fill=color)
    img.save(output.bbox_dir() / f"{idx:06d}.png")


def save_metadata(idx: int, scene: DatasetScene, instances: List[InstanceSpec], output: OutputPaths) -> None:
    meta = {
        "camera": {
            "view": scene.view.tolist(),
            "projection": scene.projection.tolist(),
            "near": scene.near_plane,
            "far": scene.far_plane,
        },
        "instances": [
            {
                "instance_id": inst.instance_id,
                "class_id": inst.class_id,
                "asset": inst.asset,
                "model_matrix": inst.model_matrix.tolist(),
            }
            for inst in instances
        ],
    }
    path = output.meta_dir() / f"{idx:06d}.json"
    path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic dataset generator prototype.")
    parser.add_argument("--config", type=Path, required=True, help="Path to asset configuration JSON.")
    parser.add_argument("--count", type=int, default=10, help="Number of images to generate.")
    parser.add_argument("--output", type=Path, default=Path("data_gen/output"), help="Output directory.")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW for off-screen rendering.")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # cần trên macOS
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(args.width, args.height, "dataset-gen", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create GLFW off-screen window.")
    glfw.make_context_current(window)

    renderer = Renderer()
    cfg = load_config(args.config)
    assets = parse_assets(cfg, args.config)
    categories = cfg.get("categories") or [
        {"id": asset.class_id, "name": asset.name} for asset in assets
    ]
    scene = DatasetScene(renderer, assets=assets)
    target = RenderTarget(args.width, args.height)
    output_paths = OutputPaths(args.output)
    coco = CocoWriter(output_paths.root / "annotations" / "coco.json", categories)
    yolo = YoloWriter(output_paths.root / "annotations" / "yolo")

    for idx in range(args.count):
        inst = random_instances(assets, count=np.random.randint(1, 4))
        scene.set_instances(inst)
        expanded_instances = scene.instances
        scene.render_into_target(target)
        rgb = np.flipud(target.read_color())[:, :, :3]
        depth = np.flipud(target.read_depth())
        mask_rgba = np.flipud(target.read_mask())
        mask_ids = decode_instance_ids(mask_rgba)
        save_outputs(idx, rgb, depth, mask_ids, output_paths)
        save_metadata(idx, scene, expanded_instances, output_paths)

        bboxes = extract_bboxes(mask_ids)
        class_lookup = {i.instance_id: i.class_id for i in expanded_instances}
        save_bbox_visualization(idx, rgb, bboxes, class_lookup, output_paths)
        file_stem = f"{idx:06d}"
        coco.add_image(f"{file_stem}.png", args.width, args.height, bboxes, class_lookup)
        yolo.write(file_stem, bboxes, args.width, args.height, class_lookup)

    coco.save()
    renderer.cleanup()
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
