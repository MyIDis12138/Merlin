from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from typing import Literal

from ..transforms.mri_transforms import BaseTransform, CropOrPad, MRITransformPipeline, Normalize, RandomFlip, ToTensor
from .breast_mri_dataset import BreastMRIDataset


class DatasetBuilder:
    """A builder class for constructing dataset instances from configuration.

    This builder supports creating instances of BreastMRIDataset with specified
    configurations for both the dataset and its transformation pipeline.

    Example Config:
        data:
          dataset:
            name: "BreastMRIDataset"
            params:
              root_dir: "data/raw/breast_mri"
              clinical_data_path: "data/raw/Clinical_and_Other_Features.xlsx"
              clinical_features_columns:
                - ["Demographics", "Date of Birth (Days)", "(Taking date of diagnosis as day 0)"]
              train_indices: [0, 1, 2, 3, 4, 5, 6, 7]
              val_indices: [8, 9]
              test_indices: [10, 11]

            transforms:
              train:  # Transforms for training set
                - name: "Normalize"
                  params:
                    range_min: -1.0
                    range_max: 1.0
              val:  # Transforms for validation set
                - name: "Normalize"
                  params:
                    range_min: -1.0
                    range_max: 1.0
    """

    @staticmethod
    def build_transforms(transform_configs: Optional[list] = None) -> Optional[MRITransformPipeline]:
        """Build transformation pipeline from configs.

        Args:
            transform_configs: List of transform configurations

        Returns:
            MRITransformPipeline if transforms are specified, None otherwise
        """
        if not transform_configs:
            return None

        transforms: List[BaseTransform] = []
        for config in transform_configs:
            transform_name = config["name"]
            params = config.get("params", {})

            transform: BaseTransform
            if transform_name == "Normalize":
                transform = Normalize(**params)
            elif transform_name == "CropOrPad":
                transform = CropOrPad(**params)
            elif transform_name == "RandomFlip":
                transform = RandomFlip(**params)
            elif transform_name == "ToTensor":
                transform = ToTensor()
            else:
                raise ValueError(f"Unknown transform: {transform_name}")

            transforms.append(transform)

        return MRITransformPipeline(transforms)

    @staticmethod
    def build_dataset(config: Dict[str, Any], split: Literal["train", "val", "test"]) -> BreastMRIDataset:
        """Build dataset instance from config for a specific split.

        Args:
            config: Configuration dictionary
            split: Dataset split to build ('train', 'val', or 'test')

        Returns:
            Configured dataset instance

        Raises:
            ValueError: If dataset name is not supported or config is invalid
            KeyError: If required configuration is missing
        """
        try:
            dataset_config = config["data"]["dataset"]
        except KeyError:
            raise KeyError("Configuration must contain 'data.dataset' section")

        dataset_name = dataset_config.get("name")
        if dataset_name != "BreastMRIDataset":
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Get dataset parameters
        params = dataset_config.get("params", {}).copy()  # Make a copy to modify

        # Get indices for the specified split and set as patient_indices
        indices_key = f"{split}_indices"
        if indices_key in params:
            patient_indices = params.pop(indices_key)
            params["patient_indices"] = patient_indices

        # Remove other split indices to avoid unexpected argument errors
        for other_split in ["train", "val", "test"]:
            other_key = f"{other_split}_indices"
            if other_key in params:
                params.pop(other_key)

        # Build transform pipeline for the specified split
        transform_configs = dataset_config.get("transforms", {}).get(split)
        transform = DatasetBuilder.build_transforms(transform_configs)

        # Create dataset instance
        dataset = BreastMRIDataset(transform=transform, **params)

        return dataset
