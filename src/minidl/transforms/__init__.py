from typing import Any

from .mri_transforms import MRITransformPipeline, Normalize, RandomBiasField, RandomFlip, RandomMotion, RandomNoise, Resize, ToTensor
from .transform_registry import TransformBuilder, TransformRegistry

__all__ = [
    "MRITransformPipeline",
    "TransformRegistry",
    "TransformBuilder",
    "build_transform_pipeline",
    "MRITorchIOPipeline",
    "Resize",
    "Normalize",
    "ToTensor",
    "RandomBiasField",
    "RandomNoise",
    "RandomMotion",
    "RandomFlip",
]


def build_transform_pipeline(transform_configs: list[dict[str, Any]] | None = None) -> MRITransformPipeline:
    """Build a transform pipeline based on configuration.

    Args:
        transform_configs: list of transform configurations

    Returns:
        Transform pipeline instance, or None if no transforms are specified
    """
    if not transform_configs:
        return MRITransformPipeline([])

    transforms = []
    for config in transform_configs:
        transform = TransformBuilder.build_transform(config)
        transforms.append(transform)

    return MRITransformPipeline(transforms)
