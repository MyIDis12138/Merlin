from collections.abc import Callable
from typing import Any, Literal

from torch.utils.data import Dataset

from minidl.transforms import build_transform_pipeline


class DatasetRegistry:
    """Registry for dataset classes.

    This class provides a registry for registering and retrieving dataset classes.
    """

    _datasets: dict[str, type[Dataset]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a dataset class.

        Args:
            name: Name of the dataset

        Returns:
            Decorator function
        """

        def decorator(dataset_cls: type[Dataset]) -> type[Dataset]:
            if name in cls._datasets:
                raise ValueError(f"Dataset '{name}' already registered")

            cls._datasets[name] = dataset_cls
            return dataset_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Dataset]:
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

    @classmethod
    def available_datasets(cls) -> dict[str, type[Dataset]]:
        """Get all available datasets.

        Returns:
            Dictionary of dataset names and classes
        """
        return cls._datasets.copy()


class DatasetBuilder:
    """Builder for creating dataset instances.

    This class provides a builder for creating dataset instances based on configuration.
    """

    @staticmethod
    def build_dataset(config: dict[str, Any], split: Literal["train", "val", "test"], train_indices) -> Dataset:
        """Build a dataset instance based on configuration.

        Args:
            config: Configuration dictionary
            split: Dataset split ('train', 'val', or 'test')

        Returns:
            Dataset instance

        Raises:
            ValueError: If dataset type is not specified or not registered
            KeyError: If required configuration is missing
        """
        dataset_config = config.get("dataset", {})

        dataset_type = dataset_config.get("name")

        if dataset_type is None:
            raise ValueError("Dataset type must be specified in config (dataset.name)")

        dataset_cls = DatasetRegistry.get(dataset_type)

        dataset_args = {}
        dataset_params = dataset_config.get("params", {})
        dataset_args.update(dataset_params)
        dataset_args["training_patient_ids"] = train_indices

        indices = dataset_config.get("indices", {}).get(split)
        if indices is None:
            raise ValueError(f"Indices for split '{split}' must be specified in config (dataset.indices.{split})")
        dataset_args["patient_indices"] = indices

        transform_configs = dataset_config.get("transforms", {}).get(split, {})
        if transform_configs:
            transform = build_transform_pipeline(transform_configs)
            if transform:
                dataset_args["transform"] = transform

        # Create dataset instance
        try:
            dataset = dataset_cls(**dataset_args)
        except Exception as e:
            raise RuntimeError(f"Failed to create dataset '{dataset_type}': {str(e)}")

        return dataset
