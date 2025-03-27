import logging
from typing import Any, cast

import numpy as np
import torch
import torchio as tio

from .transform_registry import BaseTransform, TransformRegistry


@TransformRegistry.register("MRITorchIOPipeline")
class MRITorchIOPipeline(BaseTransform):
    """TorchIO-based pipeline for MRI image transformations.

    A complete pipeline that converts numpy arrays to TorchIO subjects,
    applies a series of transforms, and converts back to numpy arrays.

    Args:
        transforms_config: List of transform configurations with 'name' and 'params'
    """

    def __init__(self, transforms_config: list):
        """Initialize the pipeline with configuration.

        Args:
            **config: Configuration dictionary that should contain transform definitions
        """
        self.transforms_config = transforms_config
        self.transform_chain = self._build_transform_chain()

    def _build_transform_chain(self) -> tio.Transform:
        """Build the TorchIO transform chain from configuration.

        Returns:
            TorchIO transform chain
        """
        transforms = []

        for config in self.transforms_config:
            transform = self._create_transform(config)
            if transform is not None:
                transforms.append(transform)

        return tio.Compose(transforms)

    def _create_transform(self, config: dict[str, Any]) -> tio.Transform | None:
        """Create a TorchIO transform from configuration.

        Args:
            config: Transform configuration with 'name' and 'params'

        Returns:
            TorchIO transform instance
        """
        transform_name = config.get("name")
        params = config.get("params", {})

        if transform_name == "Resize":
            target_shape = params.get("target_shape", (256, 256, 174))
            interpolation = params.get("interpolation", "linear")
            return tio.Resize(target_shape, image_interpolation=interpolation)

        elif transform_name == "Resample":
            target_spacing = params.get("target_spacing", 1.0)
            interpolation = params.get("interpolation", "linear")
            return tio.Resample(target_spacing, image_interpolation=interpolation)

        elif transform_name == "ZNormalization":
            masking_method = params.get("masking_method", None)
            return tio.ZNormalization(masking_method=masking_method)

        elif transform_name == "RescaleIntensity":
            out_min_max = params.get("out_min_max", (0, 1))
            percentiles = params.get("percentiles", (1, 99))
            return tio.RescaleIntensity(out_min_max=out_min_max, percentiles=percentiles)

        elif transform_name == "RandomBiasField":
            coefficients = params.get("coefficients", 0.5)
            order = params.get("order", 3)
            return tio.RandomBiasField(
                coefficients=coefficients,
                order=order,
            )

        elif transform_name == "RandomMotion":
            degrees = params.get("degrees", 5)
            translation = params.get("translation", 5)
            num_transforms = params.get("num_transforms", 2)
            return tio.RandomMotion(
                degrees=degrees,
                translation=translation,
                num_transforms=num_transforms,
            )

        elif transform_name == "RandomNoise":
            mean = params.get("mean", 0.1)
            std = params.get("std", 0.5)
            return tio.RandomNoise(mean=mean, std=std)

        elif transform_name == "RandomFlip":
            axes = params.get("axes", 1)
            flip_probability = params.get("flip_probability", 0.5)
            return tio.RandomFlip(axes=axes, flip_probability=flip_probability)

        else:
            logging.warning(f"Unknown transform: {transform_name}")
            return None

    def _numpy_to_subject(self, images: np.ndarray | list[np.ndarray]) -> tio.Subject:
        """Convert numpy array(s) to TorchIO Subject.

        Args:
            images: Single numpy array or list of numpy arrays

        Returns:
            TorchIO Subject
        """
        # Handle single array vs. list of arrays
        if isinstance(images, np.ndarray) and images.ndim == 4:  # [C, D, H, W]
            image_list = [images[i] for i in range(images.shape[0])]
        elif isinstance(images, np.ndarray) and images.ndim == 3:  # [D, H, W]
            image_list = [images]
        elif isinstance(images, list):
            image_list = images
        else:
            raise ValueError(f"Unsupported input format: {type(images)}")

        subject_dict: dict[str, tio.ScalarImage] = {}
        for i, img_array in enumerate(image_list):
            if isinstance(img_array, np.ndarray):
                # Add channel dimension if needed [D, H, W] -> [1, D, H, W]
                if img_array.ndim == 3:
                    tensor = torch.from_numpy(img_array.astype(np.float32)).unsqueeze(0)
                else:
                    tensor = torch.from_numpy(img_array.astype(np.float32))
            else:
                tensor = img_array

            subject_dict[f"Phase_{i+1}"] = tio.ScalarImage(tensor=tensor)

        subject = tio.Subject(subject_dict)

        return subject

    def _subject_to_numpy(self, subject: tio.Subject) -> np.ndarray | list[np.ndarray]:
        """Convert TorchIO Subject back to numpy array(s).

        Args:
            subject: TorchIO Subject

        Returns:
            Numpy array or list of arrays
        """
        image_keys = sorted([k for k in subject.keys() if isinstance(subject[k], tio.ScalarImage)])

        images = []
        for key in image_keys:
            if isinstance(subject[key], tio.ScalarImage):
                image = cast(tio.ScalarImage, subject[key]).numpy()

                if image.shape[0] == 1:
                    image = image[0]

                images.append(image)

        if len(images) == 1:
            return images[0]

        return images

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        """Apply the pipeline to input data.

        Args:
            x: Dictionary containing 'images' key with numpy array(s)

        Returns:
            Dictionary with transformed images
        """
        images = x["images"]

        subject = self._numpy_to_subject(images)
        transformed = self.transform_chain(subject)
        x["images"] = self._subject_to_numpy(transformed)

        return x

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return f"MRITorchIOPipeline(transforms={self.transform_chain})"
