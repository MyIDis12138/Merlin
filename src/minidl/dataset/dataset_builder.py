from typing import Any, Callable, Dict, List, Literal, Optional, Type

from torch.utils.data import Dataset

from ..transforms.mri_transforms import BaseTransform, CropOrPad, MRITransformPipeline, Normalize, RandomFlip, ToTensor


class DatasetRegistry:
    """Registry for dataset classes."""

    _datasets: Dict[str, Type[Dataset]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a dataset class.

        Args:
            name: Name of the dataset

        Returns:
            Decorator function for registration
        """

        def decorator(dataset_cls: Type[Dataset]) -> Type[Dataset]:
            cls._datasets[name] = dataset_cls
            return dataset_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[Dataset]:
        """Get a dataset class by name.

        Args:
            name: Name of the dataset

        Returns:
            Dataset class

        Raises:
            ValueError: If dataset name is not registered
        """
        if name not in cls._datasets:
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(cls._datasets.keys())}")
        return cls._datasets[name]


class TransformBuilder:
    """Builder for transform pipelines."""

    _transform_registry = {"Normalize": Normalize, "CropOrPad": CropOrPad, "RandomFlip": RandomFlip, "ToTensor": ToTensor}

    @classmethod
    def register_transform(cls, name: str, transform_cls: Type[BaseTransform]) -> None:
        """Register a new transform.

        Args:
            name: Name of the transform
            transform_cls: Transform class to register
        """
        cls._transform_registry[name] = transform_cls

    @classmethod
    def build_transforms(cls, transform_configs: Optional[list] = None) -> Optional[MRITransformPipeline]:
        """Build transformation pipeline from configs.

        Args:
            transform_configs: List of transform configurations

        Returns:
            MRITransformPipeline if transforms are specified, None otherwise

        Raises:
            ValueError: If transform name is not registered
        """
        if not transform_configs:
            return None

        transforms: List[BaseTransform] = []
        for config in transform_configs:
            transform_name = config["name"]
            params = config.get("params", {})

            if transform_name not in cls._transform_registry:
                raise ValueError(f"Unknown transform: {transform_name}. " f"Available transforms: {list(cls._transform_registry.keys())}")

            transform_cls = cls._transform_registry[transform_name]
            transform = transform_cls(**params) if params else transform_cls()
            transforms.append(transform)

        return MRITransformPipeline(transforms)


class DatasetBuilder:
    """A builder class for constructing dataset instances from configuration.

    This builder supports creating instances of any registered dataset with specified
    configurations for both the dataset and its transformation pipeline.

    Example Config:
        data:
          dataset:
            name: "BreastMRIDataset"  # Name of the registered dataset
            params:  # Dataset-specific parameters
              root_dir: "data/raw/breast_mri"
              clinical_data_path: "data/raw/Clinical_and_Other_Features.xlsx"
            transforms:  # Optional transforms for each split
              train: [...]
              val: [...]
              test: [...]
    """

    @classmethod
    def build_dataset(cls, config: Dict[str, Any], split: Literal["train", "val", "test"]) -> Dataset:
        """Build dataset instance from config for a specific split.

        Args:
            config: Configuration dictionary
            split: Dataset split to build ('train', 'val', or 'test')

        Returns:
            Configured dataset instance

        Raises:
            KeyError: If required configuration is missing
            ValueError: If dataset name is not registered
        """
        try:
            dataset_config = config["data"]["dataset"]
        except KeyError:
            raise KeyError("Configuration must contain 'data.dataset' section")

        dataset_name = dataset_config.get("name")
        if not dataset_name:
            raise ValueError("Dataset name must be specified in config")

        # Get dataset class from registry
        dataset_cls = DatasetRegistry.get(dataset_name)

        # Get dataset parameters
        params = dataset_config.get("params", {}).copy()

        # Get indices for the specified split if available
        indices_key = f"{split}_indices"
        if indices_key in dataset_config:
            params["indices"] = dataset_config[indices_key]

        # Build transform pipeline for the specified split if available
        transform_configs = dataset_config.get("transforms", {}).get(split)
        transform = TransformBuilder.build_transforms(transform_configs)
        if transform:
            params["transform"] = transform

        # Create dataset instance
        try:
            dataset = dataset_cls(**params)
        except Exception as e:
            raise RuntimeError(f"Failed to create dataset '{dataset_name}': {str(e)}")

        return dataset
