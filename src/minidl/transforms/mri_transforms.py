from typing import Any

import numpy as np
import torch
from monai.transforms import (
    RandBiasField,
    RandFlip,
    RandGaussianNoise,
    RandGibbsNoise,
    Resize as MonaiResize,
)

from .transform_registry import BaseTransform, TransformRegistry


@TransformRegistry.register("Resize")
class Resize(BaseTransform):
    """Resize the X, Y, and Z dimensions of all images to target size using MONAI.

    Works with both numpy arrays, PyTorch tensors, and lists of numpy arrays.

    Args:
        target_size (Union[tuple[int, int, int], int]): Target size for (Z, X, Y) dimensions.
            Can be a tuple (depth, height, width) or a single int for cubic volumes.
        mode (str): Interpolation mode. Default is 'bilinear'.
            Options: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', etc.
        align_corners (bool): Whether to align corners when using interpolation modes
            that support this option. Default is False.
    """

    def __init__(self, target_size: tuple[int, int, int] | int = 256, mode: str = "bilinear", align_corners: bool = False):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size, target_size)
        else:
            self.target_size = target_size
        self.mode = mode
        self.align_corners = None if mode == "nearest" else align_corners

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        images = x["images"]

        # list of numpy arrays
        if isinstance(images, list):
            resized_list = []
            for img_array in images:
                if not isinstance(img_array, np.ndarray):
                    raise TypeError(f"Expected numpy array in list, got {type(img_array)}")

                # Convert to tensor
                if img_array.ndim == 4:  # If array has format [C, Z, X, Y]
                    tensor = torch.from_numpy(img_array).float()
                else:
                    # Handle 3D case [Z, X, Y] by adding channel dimension
                    tensor = torch.from_numpy(img_array).float().unsqueeze(0)

                # Resize the tensor
                resized_tensor = self._resize_tensor(tensor)

                # Convert back to numpy and match original dimensionality
                if img_array.ndim == 4:
                    resized_list.append(resized_tensor.numpy())
                else:
                    resized_list.append(resized_tensor.squeeze(0).numpy())

            x["images"] = resized_list

        elif isinstance(images, torch.Tensor):
            # PyTorch tensor case
            x["images"] = self._resize_tensor(images)

        else:
            # NumPy array case
            tensor = torch.from_numpy(images).float()
            resized_tensor = self._resize_tensor(tensor)
            x["images"] = resized_tensor.numpy()

        return x

    def _resize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resize a tensor using MONAI transforms.

        Args:
            tensor: Input tensor of shape [C, Z, X, Y]

        Returns:
            Resized tensor
        """
        # MONAI's Resize expects spatial_size in (H, W, D) order
        # Our target_size is in (Z, X, Y) order, so we need to reorder
        monai_spatial_size = (self.target_size[1], self.target_size[2], self.target_size[0])

        # Create and apply MONAI resize transform
        resize_transform = MonaiResize(spatial_size=monai_spatial_size, mode=self.mode, align_corners=self.align_corners)

        return resize_transform(tensor)

    def __repr__(self) -> str:
        return f"Resize(target_size={self.target_size}, mode='{self.mode}', align_corners={self.align_corners})"


@TransformRegistry.register("Normalize")
class Normalize(BaseTransform):
    """Normalize the input data to a specified range using MONAI.

    Args:
        range_min (float): Minimum value of the target range
        range_max (float): Maximum value of the target range
        percentiles (tuple[float, float], optional): Percentiles for computing normalization range
        nonzero (bool): Whether to only normalize non-zero values
        channel_wise (bool): Whether to normalize each channel independently
    """

    def __init__(
        self,
        range_min: float = -1.0,
        range_max: float = 1.0,
        percentiles: tuple[float, float] | None = None,
        nonzero: bool = False,
        channel_wise: bool = False,
    ):
        self.range_min = range_min
        self.range_max = range_max
        self.percentiles = percentiles
        self.nonzero = nonzero
        self.channel_wise = channel_wise

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        images = x["images"]

        if isinstance(images, torch.Tensor):
            images_scaled = self.normalize(images)
        elif isinstance(images, list):
            images_scaled = [self.normalize(image) for image in images]
        else:
            # NumPy array case
            tensor = torch.from_numpy(images).float()
            images_scaled = self.normalize(tensor).numpy()

        x["images"] = images_scaled
        return x

    def normalize(self, image):
        """Normalize input tensor with custom implementation."""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            is_numpy = True
        else:
            is_numpy = False

        # Work with a copy to avoid modifying the original
        result = image.clone()

        # Function to normalize a single tensor
        def _normalize_tensor(t):
            # Handle empty or constant tensors
            if t.numel() == 0 or (t.max() - t.min()) < 1e-7:
                return t

            # Get normalization bounds
            if self.percentiles is not None:
                # Convert percentiles from 0-100 range to 0-1 range for torch.quantile
                low_pct, high_pct = self.percentiles[0] / 100.0, self.percentiles[1] / 100.0
                min_val = torch.quantile(t.flatten(), low_pct)
                max_val = torch.quantile(t.flatten(), high_pct)
            else:
                if self.nonzero:
                    # Only consider non-zero values
                    mask = t != 0
                    if mask.sum() > 0:  # Ensure there are non-zero values
                        masked_t = t[mask]
                        min_val = masked_t.min()
                        max_val = masked_t.max()
                    else:
                        min_val, max_val = t.min(), t.max()
                else:
                    min_val, max_val = t.min(), t.max()

            # Apply normalization
            norm_range = max_val - min_val
            if norm_range > 1e-7:  # Avoid division by very small values
                if self.nonzero:
                    # Only normalize non-zero values
                    mask = t != 0
                    if mask.sum() > 0:
                        t_normalized = t.clone()
                        t_normalized[mask] = (t_normalized[mask] - min_val) / norm_range
                        t_normalized[mask] = t_normalized[mask] * (self.range_max - self.range_min) + self.range_min
                        return t_normalized

                # Standard normalization
                t_normalized = (t - min_val) / norm_range
                t_normalized = t_normalized * (self.range_max - self.range_min) + self.range_min
                return t_normalized
            else:
                # If range is too small, just set to range_min
                return torch.full_like(t, self.range_min)

        # Apply normalization
        if self.channel_wise and image.dim() >= 3:
            # Normalize each channel independently
            if image.dim() == 3:  # [C, H, W]
                for c in range(image.shape[0]):
                    result[c] = _normalize_tensor(image[c])
            elif image.dim() == 4:  # [C, D, H, W]
                for c in range(image.shape[0]):
                    result[c] = _normalize_tensor(image[c])
        else:
            # Normalize whole image
            result = _normalize_tensor(image)

        # Convert back to numpy if input was numpy
        if is_numpy:
            result = result.numpy()

        return result

    def __repr__(self) -> str:
        return f"Normalize(range=({self.range_min}, {self.range_max}), percentiles={self.percentiles},\
                 nonzero={self.nonzero}, channel_wise={self.channel_wise})"


@TransformRegistry.register("ToTensor")
class ToTensor(BaseTransform):
    """Convert numpy array to PyTorch tensor"""

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        images = x["images"]
        if isinstance(images, list):
            # Stack list of numpy arrays
            images = np.stack(images, axis=0)

        if isinstance(images, np.ndarray):
            # Convert to contiguous array and then to tensor
            images = np.ascontiguousarray(images)
            x["images"] = torch.from_numpy(images).float()

        return x

    def __repr__(self) -> str:
        return "ToTensor()"


@TransformRegistry.register("RandomBiasField")
class RandomBiasField(BaseTransform):
    """Apply random MRI bias field artifact using MONAI.

    Args:
        coeff_range (tuple[float, float]): Range of bias field coefficients
        prob (float): Probability of applying the transform
    """

    def __init__(self, coeff_range: float | tuple[float, float] = 0.3, prob: float = 1.0):
        # Convert scalar to range if needed
        if isinstance(coeff_range, (int, float)):
            self.coeff_range = (0.0, float(coeff_range))
        else:
            self.coeff_range = coeff_range
        self.prob = prob
        self.transform = RandBiasField(coeff_range=self.coeff_range, prob=self.prob)

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        images = x["images"]

        if isinstance(images, list):
            processed_list = []
            for img in images:
                if isinstance(img, np.ndarray):
                    tensor = torch.from_numpy(img).float()
                    processed = self.transform(tensor)
                    processed_list.append(processed.numpy())
                else:
                    processed = self.transform(img)
                    processed_list.append(processed)
            x["images"] = processed_list
        elif isinstance(images, np.ndarray):
            tensor = torch.from_numpy(images).float()
            x["images"] = self.transform(tensor).numpy()
        else:
            # Already a tensor
            x["images"] = self.transform(images)

        return x

    def __repr__(self) -> str:
        return f"RandomBiasField(coeff_range={self.coeff_range}, prob={self.prob})"


@TransformRegistry.register("RandomNoise")
class RandomNoise(BaseTransform):
    """Apply random noise using MONAI.

    Args:
        mean (float): Mean value of Gaussian noise
        std (float or tuple[float, float]): Standard deviation or range of standard deviations
        prob (float): Probability of applying the transform
    """

    def __init__(self, mean: float = 0.0, std: float | tuple[float, float] = 0.1, prob: float = 1.0):
        self.mean = mean
        self.prob = prob

        # Store the original std parameter
        self.std_param = std

        # For MONAI transform initialization, we'll use a single float value
        # The actual range handling will be done in our custom implementation
        if isinstance(std, (tuple, list)):
            std_value = sum(std) / 2  # Use average as default
        else:
            std_value = float(std)

        self.transform = RandGaussianNoise(mean=self.mean, std=std_value, prob=self.prob)

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        images = x["images"]

        # Handle different input types
        if isinstance(images, list):
            processed_list = []
            for img in images:
                if isinstance(img, np.ndarray):
                    tensor = torch.from_numpy(img).float()
                    processed = self._apply_noise(tensor)
                    processed_list.append(processed.numpy())
                else:
                    processed = self._apply_noise(img)
                    processed_list.append(processed)
            x["images"] = processed_list
        elif isinstance(images, np.ndarray):
            tensor = torch.from_numpy(images).float()
            x["images"] = self._apply_noise(tensor).numpy()
        else:
            # Already a tensor
            x["images"] = self._apply_noise(images)

        return x

    def _apply_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise manually to avoid broadcasting issues."""
        # Skip if random chance is below probability threshold
        if torch.rand(1).item() > self.prob:
            return tensor

        # Determine std value - either use the given value or sample from range
        if isinstance(self.std_param, (tuple, list)):
            low, high = self.std_param
            std = torch.rand(1).item() * (high - low) + low
        else:
            std = float(self.std_param)

        # Generate noise with same shape as input tensor
        noise = torch.randn_like(tensor) * std + self.mean

        # Add noise to tensor
        return tensor + noise

    def __repr__(self) -> str:
        return f"RandomNoise(mean={self.mean}, std={self.std_param}, prob={self.prob})"


@TransformRegistry.register("RandomMotion")
class RandomMotion(BaseTransform):
    """Simulate random motion artifacts using MONAI Gibbs noise.

    Note: MONAI doesn't have a direct equivalent to TorchIO's RandomMotion,
          so we use RandGibbsNoise which produces somewhat similar artifacts.

    Args:
        prob (float): Probability of applying the transform
        num_transforms (int): Number of transforms to apply
    """

    def __init__(self, prob: float = 1.0, num_transforms: int = 2):
        self.prob = prob
        self.num_transforms = num_transforms
        self.transform = RandGibbsNoise(prob=self.prob)

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        images = x["images"]

        if isinstance(images, list):
            processed_list = []
            for img in images:
                if isinstance(img, np.ndarray):
                    tensor = torch.from_numpy(img).float()
                    processed = self._apply_multiple(tensor)
                    processed_list.append(processed.numpy())
                else:
                    processed = self._apply_multiple(img)
                    processed_list.append(processed)
            x["images"] = processed_list
        elif isinstance(images, np.ndarray):
            tensor = torch.from_numpy(images).float()
            x["images"] = self._apply_multiple(tensor).numpy()
        else:
            # Already a tensor
            x["images"] = self._apply_multiple(images)

        return x

    def _apply_multiple(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the transform multiple times to simulate more complex motion."""
        result = tensor
        for _ in range(self.num_transforms):
            result = self.transform(result)
        return result

    def __repr__(self) -> str:
        return f"RandomMotion(prob={self.prob}, num_transforms={self.num_transforms})"


@TransformRegistry.register("RandomFlip")
class RandomFlip(BaseTransform):
    """Apply random flip using MONAI.

    Args:
        axes (int or tuple[int, ...]): Axes along which to flip
        flip_probability (float): Probability of applying the transform
    """

    def __init__(self, axes: int | tuple[int, ...] = 0, flip_probability: float = 0.5):
        self.axes = axes
        self.flip_probability = flip_probability

        # Convert axes to MONAI format (spatial_axis)
        if isinstance(self.axes, int):
            # Convert single axis to MONAI format
            # Note: MONAI uses different indexing for spatial dimensions in transforms
            monai_axis = self.axes
            # Adjust if needed based on your expected input format
            self.transform = RandFlip(prob=self.flip_probability, spatial_axis=monai_axis)
        else:
            # Multiple axes
            self.transform = RandFlip(prob=self.flip_probability, spatial_axis=self.axes)

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        images = x["images"]

        if isinstance(images, list):
            processed_list = []
            for img in images:
                if isinstance(img, np.ndarray):
                    tensor = torch.from_numpy(img).float()
                    processed = self.transform(tensor)
                    processed_list.append(processed.numpy())
                else:
                    processed = self.transform(img)
                    processed_list.append(processed)
            x["images"] = processed_list
        elif isinstance(images, np.ndarray):
            tensor = torch.from_numpy(images).float()
            x["images"] = self.transform(tensor).numpy()
        else:
            # Already a tensor
            x["images"] = self.transform(images)

        return x

    def __repr__(self) -> str:
        return f"RandomFlip(axes={self.axes}, flip_probability={self.flip_probability})"


class MRITransformPipeline:
    """Pipeline for MRI image transformations

    Args:
        transforms (list[BaseTransform]): list of transforms to apply

    Example:
        >>> transform = MRITransformPipeline([
        ...     Normalize(range_min=0, range_max=1, percentiles=(1, 99)),
        ...     RandomFlip(flip_prob=0.5, dims=[1, 2]),
        ...     ToTensor()
        ... ])
    """

    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
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
