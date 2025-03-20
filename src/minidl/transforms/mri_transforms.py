from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .transform_registry import BaseTransform, TransformRegistry


@TransformRegistry.register("ResizeXY")
class ResizeXY(BaseTransform):
    """Resize the X and Y dimensions of all images to target size using bicubic interpolation.

    Works with both numpy arrays, PyTorch tensors, and lists of numpy arrays.

    Args:
        target_size (Union[Tuple[int, int], int]): Target size for X and Y dimensions.
            Can be a tuple (width, height) or a single int for square images.
        mode (str): Interpolation mode. Default is 'bicubic'.
            Options for torch: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
            Options for numpy: 'nearest', 'linear', 'area', 'cubic'
    """

    def __init__(self, target_size: tuple[int, int] | int = 512, mode: str = "bicubic"):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
        self.mode = mode

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        images = x["images"]

        # New case: List of numpy arrays
        if isinstance(images, list):
            import cv2

            # Process each array in the list separately
            resized_list = []
            for img_array in images:
                if not isinstance(img_array, np.ndarray):
                    raise TypeError(f"Expected numpy array in list, got {type(img_array)}")

                # Get the shape of the current array
                if img_array.ndim == 4:  # If array has format [C, Z, X, Y]
                    C, Z, _, _ = img_array.shape
                    resized_array = np.zeros((C, Z, self.target_size[0], self.target_size[1]), dtype=img_array.dtype)

                    # Map PyTorch interpolation modes to OpenCV interpolation flags
                    mode_map = {
                        "nearest": cv2.INTER_NEAREST,
                        "linear": cv2.INTER_LINEAR,
                        "bilinear": cv2.INTER_LINEAR,
                        "bicubic": cv2.INTER_CUBIC,
                        "area": cv2.INTER_AREA,
                    }

                    interpolation = mode_map.get(self.mode, cv2.INTER_CUBIC)

                    # Resize each slice
                    for c in range(C):
                        for z in range(Z):
                            resized_array[c, z] = cv2.resize(
                                img_array[c, z],
                                (self.target_size[1], self.target_size[0]),  # OpenCV expects (width, height)
                                interpolation=interpolation,
                            )

                    resized_list.append(resized_array)
                else:
                    # Handle other dimensionalities as needed
                    # For example, if you have [Z, X, Y] format:
                    Z, _, _ = img_array.shape
                    resized_array = np.zeros((Z, self.target_size[0], self.target_size[1]), dtype=img_array.dtype)

                    mode_map = {
                        "nearest": cv2.INTER_NEAREST,
                        "linear": cv2.INTER_LINEAR,
                        "bilinear": cv2.INTER_LINEAR,
                        "bicubic": cv2.INTER_CUBIC,
                        "area": cv2.INTER_AREA,
                    }

                    interpolation = mode_map.get(self.mode, cv2.INTER_CUBIC)

                    for z in range(Z):
                        resized_array[z] = cv2.resize(img_array[z], (self.target_size[1], self.target_size[0]), interpolation=interpolation)

                    resized_list.append(resized_array)

            x["images"] = resized_list

        elif isinstance(images, torch.Tensor):
            # PyTorch tensor case: (C, Z, X, Y)
            C, Z, _, _ = images.shape

            # Reshape to merge C and Z dimensions for 2D resizing
            reshaped = images.reshape(C * Z, 1, images.shape[2], images.shape[3])

            # Resize using F.interpolate (handles batched 2D images)
            if self.mode == "bicubic":
                # Using align_corners=False as the default behavior
                resized = F.interpolate(reshaped, size=self.target_size, mode="bicubic", align_corners=False)
            else:
                align_corners = None if self.mode == "nearest" else False
                resized = F.interpolate(reshaped, size=self.target_size, mode=self.mode, align_corners=align_corners)

            # Reshape back to original format
            x["images"] = resized.reshape(C, Z, self.target_size[0], self.target_size[1])

        else:
            # NumPy array case
            import cv2

            C, Z, _, _ = images.shape
            resized_images = np.zeros((C, Z, self.target_size[0], self.target_size[1]), dtype=images.dtype)

            # Map PyTorch interpolation modes to OpenCV interpolation flags
            mode_map = {
                "nearest": cv2.INTER_NEAREST,
                "linear": cv2.INTER_LINEAR,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
            }

            interpolation = mode_map.get(self.mode, cv2.INTER_CUBIC)

            # Resize each slice
            for c in range(C):
                for z in range(Z):
                    resized_images[c, z] = cv2.resize(
                        images[c, z], (self.target_size[1], self.target_size[0]), interpolation=interpolation  # OpenCV expects (width, height)
                    )

            x["images"] = resized_images

        return x

    def __repr__(self) -> str:
        return f"ResizeXY(target_size={self.target_size}, mode='{self.mode}')"


@TransformRegistry.register("ResampleZ")
class ResampleZ(BaseTransform):
    """Resample the Z-dimension (number of slices) of all volumes to a fixed number.

    Works with PyTorch tensors, numpy arrays, and lists of numpy arrays.

    Args:
        target_slices (int): Target number of slices in Z dimension
        mode (str): Interpolation mode. Default is 'linear'.
            Options for torch: 'nearest', 'linear', 'bilinear', 'trilinear'
            Options for numpy: 'nearest', 'linear'
        center_around_median (bool): If True, centers the resampling around the median slice.
            If False, resamples the entire volume uniformly. Default is True.
    """

    def __init__(self, target_slices: int = 174, mode: str = "linear", center_around_median: bool = True):
        self.target_slices = target_slices
        self.mode = mode
        self.center_around_median = center_around_median

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        images = x["images"]

        # New case: List of numpy arrays
        if isinstance(images, list):
            resampled_list = []

            for img_array in images:
                if not isinstance(img_array, np.ndarray):
                    raise TypeError(f"Expected numpy array in list, got {type(img_array)}")

                # Get the shape and determine how to process
                if img_array.ndim == 4:  # [C, Z, X, Y] format
                    C, Z, H, W = img_array.shape

                    if Z == self.target_slices:
                        # No resampling needed
                        resampled_list.append(img_array)
                        continue

                    if self.center_around_median and Z > 1:
                        # Find the median slice index
                        median_slice = Z // 2

                        # Calculate start and end indices to center around median
                        if self.target_slices >= Z:
                            # Upsampling: use the full range and interpolate
                            resampled = self._resample_numpy(img_array, self.target_slices)
                        else:
                            # Downsampling: take a centered subset and then interpolate if needed
                            half_target = self.target_slices // 2
                            start = max(0, median_slice - half_target)
                            end = min(Z, start + self.target_slices)

                            # Adjust start if end hit the boundary
                            if end == Z:
                                start = max(0, Z - self.target_slices)

                            subset = img_array[:, start:end]
                            if subset.shape[1] != self.target_slices:
                                resampled = self._resample_numpy(subset, self.target_slices)
                            else:
                                resampled = subset
                    else:
                        # Resample the entire Z dimension uniformly
                        resampled = self._resample_numpy(img_array, self.target_slices)

                    resampled_list.append(resampled)
                else:
                    # Handle [Z, X, Y] format
                    Z, H, W = img_array.shape

                    if Z == self.target_slices:
                        resampled_list.append(img_array)
                        continue

                    if self.center_around_median and Z > 1:
                        median_slice = Z // 2

                        if self.target_slices >= Z:
                            # Expand dimension to [1, Z, H, W] for resampling then squeeze back
                            expanded = np.expand_dims(img_array, axis=0)
                            resampled = self._resample_numpy(expanded, self.target_slices)
                            resampled = np.squeeze(resampled, axis=0)
                        else:
                            half_target = self.target_slices // 2
                            start = max(0, median_slice - half_target)
                            end = min(Z, start + self.target_slices)

                            if end == Z:
                                start = max(0, Z - self.target_slices)

                            subset = img_array[start:end]
                            if subset.shape[0] != self.target_slices:
                                # Expand dimension for resampling then squeeze back
                                expanded = np.expand_dims(subset, axis=0)
                                resampled = self._resample_numpy(expanded, self.target_slices)
                                resampled = np.squeeze(resampled, axis=0)
                            else:
                                resampled = subset
                    else:
                        # Expand dimension for resampling then squeeze back
                        expanded = np.expand_dims(img_array, axis=0)
                        resampled = self._resample_numpy(expanded, self.target_slices)
                        resampled = np.squeeze(resampled, axis=0)

                    resampled_list.append(resampled)

            x["images"] = resampled_list

        elif isinstance(images, torch.Tensor):
            # PyTorch tensor case
            C, Z, H, W = images.shape

            if Z == self.target_slices:
                return x  # No resampling needed

            if self.center_around_median and Z > 1:
                # Find the median slice index
                median_slice = Z // 2

                # Calculate start and end indices to center around median
                if self.target_slices >= Z:
                    # Upsampling: use the full range and interpolate
                    resampled = self._resample_torch(images, self.target_slices)
                else:
                    # Downsampling: take a centered subset and then interpolate if needed
                    half_target = self.target_slices // 2
                    start = max(0, median_slice - half_target)
                    end = min(Z, start + self.target_slices)

                    # Adjust start if end hit the boundary
                    if end == Z:
                        start = max(0, Z - self.target_slices)

                    subset = images[:, start:end]
                    if subset.shape[1] != self.target_slices:
                        resampled = self._resample_torch(subset, self.target_slices)
                    else:
                        resampled = subset
            else:
                # Resample the entire Z dimension uniformly
                resampled = self._resample_torch(images, self.target_slices)

            x["images"] = resampled

        else:
            # NumPy array case
            C, Z, H, W = images.shape

            if Z == self.target_slices:
                return x  # No resampling needed

            if self.center_around_median and Z > 1:
                # Find the median slice index
                median_slice = Z // 2

                # Calculate start and end indices to center around median
                if self.target_slices >= Z:
                    # Upsampling: use the full range and interpolate
                    resampled = self._resample_numpy(images, self.target_slices)
                else:
                    # Downsampling: take a centered subset and then interpolate if needed
                    half_target = self.target_slices // 2
                    start = max(0, median_slice - half_target)
                    end = min(Z, start + self.target_slices)

                    # Adjust start if end hit the boundary
                    if end == Z:
                        start = max(0, Z - self.target_slices)

                    subset = images[:, start:end]
                    if subset.shape[1] != self.target_slices:
                        resampled = self._resample_numpy(subset, self.target_slices)
                    else:
                        resampled = subset
            else:
                # Resample the entire Z dimension uniformly
                resampled = self._resample_numpy(images, self.target_slices)

            x["images"] = resampled

        return x

    def _resample_torch(self, images: torch.Tensor, target_slices: int) -> torch.Tensor:
        """Helper method to resample PyTorch tensor in Z dimension."""
        C, Z, H, W = images.shape

        # Reshape to format expected by F.interpolate: [B, C, D]
        reshaped = images.reshape(C, Z, H * W).permute(0, 2, 1)

        # Use 1D interpolation for Z dimension
        if self.mode == "nearest":
            resampled = F.interpolate(reshaped, size=target_slices, mode="nearest")
        else:
            # For 1D interpolation, we use linear (which is 1D)
            resampled = F.interpolate(reshaped, size=target_slices, mode="linear", align_corners=False)

        # Reshape back to original format: [C, Z, H, W]
        return resampled.permute(0, 2, 1).reshape(C, target_slices, H, W)

    def _resample_numpy(self, images: np.ndarray, target_slices: int) -> np.ndarray:
        """Helper method to resample NumPy array in Z dimension."""
        from scipy.ndimage import zoom

        # Handle the dimensionality of the input array
        if images.ndim == 4:  # [C, Z, H, W] format
            C, Z, H, W = images.shape

            # Calculate zoom factor for Z dimension
            z_factor = target_slices / Z

            # Create output array
            resampled = np.zeros((C, target_slices, H, W), dtype=images.dtype)

            # Apply zoom for each channel
            for c in range(C):
                if self.mode == "nearest":
                    resampled[c] = zoom(images[c], (z_factor, 1, 1), order=0)
                else:
                    # Linear interpolation
                    resampled[c] = zoom(images[c], (z_factor, 1, 1), order=1)
        elif images.ndim == 3:  # [Z, H, W] format
            Z, H, W = images.shape

            # Calculate zoom factor for Z dimension
            z_factor = target_slices / Z

            # Apply zoom directly
            if self.mode == "nearest":
                resampled = zoom(images, (z_factor, 1, 1), order=0)
            else:
                resampled = zoom(images, (z_factor, 1, 1), order=1)
        else:
            raise ValueError(f"Unsupported array shape with dimensions: {images.ndim}")

        return resampled

    def __repr__(self) -> str:
        return f"ResampleZ(target_slices={self.target_slices}, mode='{self.mode}', center_around_median={self.center_around_median})"


@TransformRegistry.register("Normalize")
class Normalize(BaseTransform):
    """Normalize the input data to a specified range

    Args:
        range_min (float): Minimum value of the target range
        range_max (float): Maximum value of the target range
        percentiles (Tuple[float, float], optional): Percentiles for computing normalization range
    """

    def __init__(self, range_min: float = -1.0, range_max: float = 1.0, percentiles: tuple[float, float] | None = None):
        self.range_min = range_min
        self.range_max = range_max
        self.percentiles = percentiles

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        images = x["images"]
        if isinstance(images, list):
            images = np.stack(images, axis=0)

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

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
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
        transforms (list[BaseTransform]): list of transforms to apply

    Example:
        >>> transform = MRITransformPipeline([
        ...     Normalize(range_min=0, range_max=1, percentiles=(1, 99)),
        ...     CropOrPad(target_size=(256, 256, 32)),
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
