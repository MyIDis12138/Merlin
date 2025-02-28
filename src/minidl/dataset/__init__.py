"""Dataset module initialization and registration."""

from .breast_mri_dataset import BreastMRIDataset
from .dataset_builder import DatasetRegistry

# Register all datasets
DatasetRegistry.register("BreastMRIDataset")(BreastMRIDataset)

# Clean up namespace
del DatasetRegistry  # Optional: remove from namespace after registration
