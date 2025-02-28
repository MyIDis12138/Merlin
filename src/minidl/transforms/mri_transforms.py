from typing import List, Optional, Tuple, Union

import numpy as np
import torch


class BaseTransform:
    """Base class for all transforms"""

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError("__call__ method must be implemented in subclasses")

    def __repr__(self) -> str:
        return self.__class__.__name__


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

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            x = x.float()
            if self.percentiles is not None:
                min_val = torch.quantile(x, self.percentiles[0] / 100)
                max_val = torch.quantile(x, self.percentiles[1] / 100)
            else:
                min_val = x.min()
                max_val = x.max()
        else:
            x = x.astype(np.float32)
            if self.percentiles is not None:
                min_val = np.percentile(x, self.percentiles[0])
                max_val = np.percentile(x, self.percentiles[1])
            else:
                min_val = x.min()
                max_val = x.max()

        # Avoid division by zero
        if max_val == min_val:
            return x * 0 + self.range_min

        x_std = (x - min_val) / (max_val - min_val)
        x_scaled = x_std * (self.range_max - self.range_min) + self.range_min
        return x_scaled

    def __repr__(self) -> str:
        return f"Normalize(range=({self.range_min}, {self.range_max}), percentiles={self.percentiles})"


class CropOrPad(BaseTransform):
    """Crop or pad the input to a target size

    Args:
        target_size (Tuple[int, ...]): Target size for each dimension
        pad_value (float): Value to use for padding
    """

    def __init__(self, target_size: Tuple[int, ...], pad_value: float = 0):
        self.target_size = target_size
        self.pad_value = pad_value

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        current_size = x.shape

        # Ensure dimensions match
        if len(current_size) != len(self.target_size):
            raise ValueError(f"Input dimensions {len(current_size)} do not match target dimensions {len(self.target_size)}")

        # Calculate padding or cropping
        slices = []
        pad_width = []

        for curr, target in zip(current_size, self.target_size):
            if curr < target:
                # Need padding
                pad_before = (target - curr) // 2
                pad_after = target - curr - pad_before
                slices.append(slice(None))
                pad_width.append((pad_before, pad_after))
            else:
                # Need cropping
                crop_start = (curr - target) // 2
                slices.append(slice(crop_start, crop_start + target))
                pad_width.append((0, 0))

        # Apply cropping and padding
        if isinstance(x, torch.Tensor):
            # Crop first
            x = x[tuple(slices)]
            # Then pad
            if any(p[0] > 0 or p[1] > 0 for p in pad_width):
                x = torch.nn.functional.pad(x, [p for pair in reversed(pad_width) for p in pair], mode="constant", value=self.pad_value)
        else:
            # Crop first
            x = x[tuple(slices)]
            # Then pad
            if any(p[0] > 0 or p[1] > 0 for p in pad_width):
                x = np.pad(x, pad_width, mode="constant", constant_values=self.pad_value)

        return x

    def __repr__(self) -> str:
        return f"CropOrPad(target_size={self.target_size}, pad_value={self.pad_value})"


class RandomFlip(BaseTransform):
    """Randomly flip the input along specified dimensions

    Args:
        flip_prob (float): Probability of flipping each dimension
        dims (List[int]): List of dimensions that can be flipped
    """

    def __init__(self, flip_prob: float = 0.5, dims: Optional[List[int]] = None):
        self.flip_prob = flip_prob
        self.dims = dims

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        dims = self.dims if self.dims is not None else list(range(x.ndim))

        for dim in dims:
            if np.random.random() < self.flip_prob:
                if isinstance(x, torch.Tensor):
                    x = torch.flip(x, [dim])
                else:
                    x = np.flip(x, axis=dim)

        return x

    def __repr__(self) -> str:
        return f"RandomFlip(p={self.flip_prob}, dims={self.dims})"


class ToTensor(BaseTransform):
    """Convert numpy array to PyTorch tensor"""

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            # Create a contiguous copy of the array to avoid negative stride issues
            x = np.ascontiguousarray(x)
            return torch.from_numpy(x)
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

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
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
