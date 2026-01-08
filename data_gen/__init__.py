"""Utility helpers for the synthetic dataset generation pipeline."""

from .annotations import CocoWriter, decode_instance_ids, extract_bboxes
from .dataset_scene import (
    AssetSpec,
    DatasetScene,
    InstanceSpec,
    MeshLibrary,
    compose_model_matrix,
    encode_instance_color,
)
from .render_target import RenderTarget

__all__ = [
    "CocoWriter",
    "AssetSpec",
    "DatasetScene",
    "InstanceSpec",
    "MeshLibrary",
    "RenderTarget",
    "compose_model_matrix",
    "encode_instance_color",
    "decode_instance_ids",
    "extract_bboxes",
]
