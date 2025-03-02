from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .transform_registry import TransformRegistry


class BaseTransform:
    """Base class for all transforms"""

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("__call__ method must be implemented in subclasses")

    def __repr__(self) -> str:
        return self.__class__.__name__


@TransformRegistry.register("Normalize")
class Normalize(BaseTransform):
    """Normalize the input data to a specified range

    Args:
        range_min (float): Minimum value of the target range
        range_max (float): Maximum value of the target range
        percentiles (Tuple[float, float], optional): Percentiles for computing normalization range
    """

    def __init__(self, range_min: float = -1.0, range_max: float = 1.0, percentiles: Optional[Tuple[float, float]] = None):
        self.range_min = range_min
        self.range_max = range_max
        self.percentiles = percentiles

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        images = x["images"]
        if isinstance(images, torch.Tensor):
            images = images.float()
            if self.percentiles is not None:
                min_val = torch.quantile(images, self.percentiles[0] / 100)
                max_val = torch.quantile(images, self.percentiles[1] / 100)
            else:
                min_val = images.min()
                max_val = images.max()
        else:
            images = images.astype(np.float32)
            if self.percentiles is not None:
                min_val = np.percentile(images, self.percentiles[0])
                max_val = np.percentile(images, self.percentiles[1])
            else:
                min_val = images.min()
                max_val = images.max()

        # Avoid division by zero
        if max_val == min_val:
            images_scaled = images * 0 + self.range_min
            x["images"] = images_scaled
            return x

        images_scaled = (images - min_val) / (max_val - min_val)
        images_scaled = images_scaled * (self.range_max - self.range_min) + self.range_min
        x["images"] = images_scaled
        return x

    def __repr__(self) -> str:
        return f"Normalize(range=({self.range_min}, {self.range_max}), percentiles={self.percentiles})"


@TransformRegistry.register("ToTensor")
class ToTensor(BaseTransform):
    """Convert numpy array to PyTorch tensor"""

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        images = x["images"]
        if isinstance(images, np.ndarray):
            # Create a contiguous copy of the array to avoid negative stride issues
            images = np.ascontiguousarray(images)
            x["images"] = torch.from_numpy(images)
        return x

    def __repr__(self) -> str:
        return "ToTensor()"


class MRITransformPipeline:
    """Pipeline for MRI image transformations

    Args:
        transforms (List[BaseTransform]): List of transforms to apply

    Example:
        >>> transform = MRITransformPipeline([
        ...     Normalize(range_min=0, range_max=1, percentiles=(1, 99)),
        ...     CropOrPad(target_size=(256, 256, 32)),
        ...     RandomFlip(flip_prob=0.5, dims=[1, 2]),
        ...     ToTensor()
        ... ])
    """

    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
