from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json
import numpy as np


def decode_instance_ids(mask_rgba: np.ndarray) -> np.ndarray:
    """Decode instance ids from the RGBA mask buffer produced by DatasetScene."""

    r = mask_rgba[:, :, 0].astype(np.uint32)
    g = mask_rgba[:, :, 1].astype(np.uint32)
    b = mask_rgba[:, :, 2].astype(np.uint32)
    return r | (g << 8) | (b << 16)


def extract_bboxes(instance_ids: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
    """Compute bounding boxes for each instance id (excluding 0)."""

    bboxes: Dict[int, Tuple[int, int, int, int]] = {}
    ids = np.unique(instance_ids)
    for inst_id in ids:
        if inst_id == 0:
            continue
        ys, xs = np.where(instance_ids == inst_id)
        if ys.size == 0 or xs.size == 0:
            continue
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        bboxes[int(inst_id)] = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    return bboxes


@dataclass
class CocoWriter:
    output: Path
    categories: List[Dict[str, int | str]]

    def __post_init__(self) -> None:
        self.images: List[Dict[str, int | str]] = []
        self.annotations: List[Dict[str, int | float]] = []
        self._next_image_id = 1
        self._next_ann_id = 1

    def add_image(
        self,
        file_name: str,
        width: int,
        height: int,
        bboxes: Dict[int, Tuple[int, int, int, int]],
        class_lookup: Dict[int, int],
    ) -> None:
        image_id = self._next_image_id
        self._next_image_id += 1
        self.images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        for inst_id, bbox in bboxes.items():
            category_id = class_lookup.get(inst_id, 0)
            self.annotations.append(
                {
                    "id": self._next_ann_id,
                    "image_id": image_id,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "category_id": category_id,
                }
            )
            self._next_ann_id += 1

    def save(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        data = {"images": self.images, "annotations": self.annotations, "categories": self.categories}
        self.output.write_text(json_dumps(data))


def json_dumps(data: dict) -> str:
    return json.dumps(data, indent=2)
