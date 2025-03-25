import glob
import logging
import os
import re
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Hashable

import numpy as np
import pandas as pd
import pydicom
from torch.utils.data import Dataset

from .dataset_registry import DatasetRegistry

logger = logging.getLogger(__name__)


@DatasetRegistry.register("bmri_dataset")
class BreastMRIDataset(Dataset):
    """Breast MRI Dataset Loader

    A specialized dataset loader for handling dynamic contrast-enhanced breast MRI sequences,
    with support for clinical features and clinical labels information.

    Dataset Structure:
        root_dir/
        ├── Breast_MRI_001/
        │   └── patient_directory/
        │       ├── dynamic_sequence_Ph1/
        │       │   └── *.dcm files
        │       ├── dynamic_sequence_Ph2/
        │       │   └── *.dcm files
        │       ├── dynamic_sequence_Ph3/
        │       │   └── *.dcm files
        │       └── ...
        ├── Breast_MRI_002/
        └── ...

    Features:
        1. Automatic handling of multi-level DICOM file structures
        2. Batch loading of 3 specific dynamic sequences per patient (Ph1, Ph2, Ph3)
        3. Integration of clinical data and clinical label information
        4. Support for flexible data transformation pipelines
        5. Comprehensive data validation and error handling

    Possible clinical_label:
        Molecular Subtype Mapping:
            - 0: 'luminal-like'
            - 1: 'ER/PR pos, HER2 pos'
            - 2: 'her2'
            - 3: 'trip neg'
        
        Recurrence Mapping:
            - 0: no
            - 1: yes

    Return Format:
        Each sample returns a dictionary containing:
        - 'images': Tensor of shape [3, D, H, W] representing 3D images at 3 timepoints
        - 'patient_id': Patient identifier
        - 'clinical_label': clinical_label (if clinical data is provided)
        - 'clinical_features': dictionary of additional clinical features (if specified)

    Args:
        root_dir (str): Root directory path containing the dataset
        clinical_data_path (str, optional): Path to Clinical_and_Other_Features.xlsx file
        clinical_features_columns (list[tuple[str, str, str]], optional): list of clinical features to extract.
            Each tuple should contain (category, feature_name, description) matching the Excel file's
            multi-level column headers. For example:
            [
                ('Demographics', 'Date of Birth (Days)', '(Taking date of diagnosis as day 0)'),
                ('Demographics', 'Menopause (at diagnosis)', '{0 = pre, 1 = post, 2 = N/A}'),
            ]
        transform (callable, optional): Transform pipeline for image preprocessing
        patient_indices (list[int], optional): list of Breast_MRI_XXX indices to load
        max_workers (int): Maximum number of worker threads for parallel processing
        cache_size (int): Size of the LRU cache for DICOM reading

    Raises:
        FileNotFoundError: When root directory doesn't exist or no valid Breast_MRI_XXX directories found
        RuntimeError: When no valid patient data or required dynamic sequences not found
        ValueError: When patient indices are out of range or invalid
    """

    def __init__(
        self,
        root_dir: str,
        clinical_label: list[str, str, str],
        clinical_data_path: str | None = None,
        clinical_features_columns: list[tuple[str, str, str]] | None = None,
        transform: Callable | None = None,
        patient_indices: list[int] | None = None,
        max_workers: int = 4,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.clinical_label = tuple(clinical_label)
        self.clinical_features_columns = [tuple(col) for col in clinical_features_columns] if clinical_features_columns else []
        self.clinical_ID_col = ("Patient Information", "Patient ID", "")
        self.max_workers = max_workers

        # Required sequence phases
        self.required_phases = ["Ph1", "Ph2", "Ph3"]

        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        self._initialize_clinical_data(clinical_data_path)
        self._initialize_patient_data(patient_indices)

    def _initialize_clinical_data(self, clinical_data_path: str | None) -> None:
        """Initialize clinical data from Excel file."""
        self.clinical_data = None
        if clinical_data_path is not None:
            try:
                self.clinical_data = pd.read_excel(clinical_data_path, header=[0, 1, 2])
                self.clinical_data.columns = [col[:-1] + ("",) if "Unnamed" in col[-1] else col for col in self.clinical_data.columns]  # type: ignore
                logger.info(f"Successfully loaded clinical data from {clinical_data_path}")
            except Exception as e:
                logger.warning(f"Failed to load clinical data: {e}")
                self.clinical_data = None

    def _initialize_patient_data(self, patient_indices: list[int] | None) -> None:
        """Initialize patient directories and validate data."""
        # Find and validate directories
        all_mri_dirs = self._get_valid_mri_dirs(patient_indices)

        # Get patient directories with sufficient dynamic sequences
        self.patient_dirs = self._get_valid_patient_dirs(all_mri_dirs)

        if not self.patient_dirs:
            raise RuntimeError("No valid patient data with dynamic sequences found")

        logger.info(f"Found {len(self.patient_dirs)} patient directories")

        # Initialize patient data
        self.patient_data = self._initialize_dynamic_sequences()

    def _get_valid_mri_dirs(self, patient_indices: list[int] | None) -> list[str]:
        """
        Get valid MRI directories based on the specified patient indices.

        Args:
            patient_indices: Optional list of patient indices to filter directories

        Returns:
            list of valid MRI directory paths
        """
        # Find all directories matching Breast_MRI_XXX pattern
        all_mri_dirs = glob.glob(os.path.join(self.root_dir, "Breast_MRI_*"))
        valid_mri_dirs = []
        dir_indices = {}

        # Validate directory format and build index mapping
        for mri_dir in all_mri_dirs:
            dir_name = os.path.basename(mri_dir)
            if os.path.isdir(mri_dir) and re.match(r"Breast_MRI_\d+$", dir_name):
                index = int(dir_name.split("_")[-1])
                dir_indices[index] = mri_dir
                valid_mri_dirs.append(mri_dir)

        if not valid_mri_dirs:
            raise FileNotFoundError(f"No valid Breast_MRI_XXX format directories found in {self.root_dir}")

        # Sort directories by numerical order
        valid_mri_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[-1]))
        available_indices = sorted(dir_indices.keys())

        # If patient indices are provided, filter Breast_MRI_XXX directories
        if patient_indices is not None:
            invalid_indices = [idx for idx in patient_indices if idx not in dir_indices]
            if invalid_indices:
                raise ValueError(f"Invalid patient indices: {invalid_indices}. Available indices are: {available_indices}")

            valid_mri_dirs = [dir_indices[idx] for idx in patient_indices]
            logger.info(f"Using {len(patient_indices)} specified Breast_MRI_XXX directories")

        return valid_mri_dirs

    def _identify_phase(self, sequence_folder: str) -> str | None:
        """
        Identify dynamic phase based on folder name.

        Args:
            sequence_folder: Path to sequence folder

        Returns:
            Phase name or None if not a valid phase
        """
        folder_name = os.path.basename(sequence_folder)

        # Check if it's a dynamic sequence first
        if not (("dyn" in folder_name.lower()) or ("vibrant" in folder_name.lower())):
            return None

        # Identify specific phase
        if ("ph1" in folder_name.lower()) or ("1st" in folder_name.lower()):
            return "Ph1"
        elif ("ph2" in folder_name.lower()) or ("2nd" in folder_name.lower()):
            return "Ph2"
        elif ("ph3" in folder_name.lower()) or ("3rd" in folder_name.lower()):
            return "Ph3"

        return None

    def _get_valid_patient_dirs(self, mri_dirs: list[str]) -> list[str]:
        """
        Get valid patient directories from MRI directories.

        Args:
            mri_dirs: list of MRI directory paths

        Returns:
            list of valid patient directory paths
        """
        patient_dirs = []
        for mri_dir in mri_dirs:
            patient_subdirs = [d for d in glob.glob(os.path.join(mri_dir, "*")) if os.path.isdir(d)]

            # Check each patient directory
            for patient_dir in patient_subdirs:
                sequence_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]

                # Check if all required phases are available
                phases_found = set()
                for seq_dir in sequence_dirs:
                    phase = self._identify_phase(seq_dir)
                    if phase:
                        phases_found.add(phase)

                if all(phase in phases_found for phase in self.required_phases):
                    patient_dirs.append(patient_dir)
                else:
                    missing_phases = set(self.required_phases) - phases_found
                    logger.warning(f"Warning: Patient directory {os.path.basename(patient_dir)} is missing required phases: {missing_phases}")

        return sorted(patient_dirs)

    def _initialize_dynamic_sequences(self) -> list[dict[str, str]]:
        """
        Initialize dynamic sequences for all valid patient directories.

        Returns:
            list of dictionaries mapping phase names to their directory paths
        """
        patient_data = []

        for patient_dir in self.patient_dirs:
            sequence_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]

            # Map of phases to directories
            phase_to_dir = {}

            # Identify phases
            for seq_dir in sequence_dirs:
                phase = self._identify_phase(seq_dir)
                if phase in self.required_phases:
                    phase_to_dir[phase] = seq_dir

            # Check if all required phases are available
            if all(phase in phase_to_dir for phase in self.required_phases):
                patient_id = os.path.basename(os.path.dirname(patient_dir))
                
                # Skip patients without valid clinical labels if clinical data is available
                if self.clinical_data is not None:
                    try:
                        patient_data_row = self.clinical_data[self.clinical_data[self.clinical_ID_col] == patient_id]
                        if patient_data_row.empty:
                            logger.warning(f"Skipping patient {patient_id}: no matching clinical data found")
                            continue
                        
                        clinical_label_value = patient_data_row[self.clinical_label].values[0]
                        if np.isnan(clinical_label_value):
                            logger.warning(f"Skipping patient {patient_id}: invalid clinical label: {clinical_label_value}")
                            continue
                    except Exception as e:
                        logger.warning(f"Skipping patient {patient_id}: error checking clinical label: {e}")
                        continue
                
                patient_data.append(phase_to_dir)
                logger.debug(f"Successfully loaded patient directory: {os.path.basename(patient_dir)}")
            else:
                logger.warning(f"Skipping patient {os.path.basename(patient_dir)}: missing required phases")

        if not patient_data:
            raise RuntimeError("No valid patient data found with all required dynamic phases")

        logger.info(f"Successfully loaded {len(patient_data)} valid patient datasets")
        return patient_data

    @staticmethod
    @lru_cache(maxsize=128)
    def _read_dicom_file(file_path: str) -> np.ndarray | None:
        """Read a DICOM file and return its pixel array."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return pydicom.dcmread(file_path).pixel_array.astype(np.float32)
        except Exception as e:
            logger.error(f"Error reading DICOM file {file_path}: {e}")
            return None

    def _load_sequence_images(self, series_path: str) -> np.ndarray | None:
        """Load all DICOM images in a sequence directory."""
        dicom_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
        if not dicom_files:
            return None

        # Parallel load DICOM files
        slices = list(self.thread_pool.map(self._read_dicom_file, dicom_files))
        slices = [s for s in slices if s is not None]

        return np.stack(slices, axis=0) if slices else None

    def _get_clinical_features(self, patient_id: str) -> dict[str, Any]:
        """Get clinical features for a patient."""
        if self.clinical_data is None:
            return {"clinical_label": None, "clinical_features": {}}

        try:
            patient_data = self.clinical_data[self.clinical_data[self.clinical_ID_col] == patient_id]
            if patient_data.empty:
                return {"clinical_label": None, "clinical_features": {}}

            clinical_label = int(patient_data[self.clinical_label].values[0])

            clinical_features: dict[Hashable, Any] = {}
            if self.clinical_features_columns:
                features_df = patient_data[self.clinical_features_columns]
                clinical_features = features_df.to_dict(orient="records")[0] if not features_df.empty else {}

            return {"clinical_label": clinical_label, "clinical_features": clinical_features}
        except Exception as e:
            logger.exception(f"Failed to load clinical data for patient {patient_id}: {e}")
            return {"clinical_label": None, "clinical_features": {}}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single patient's data."""
        phase_dirs = self.patient_data[idx]

        # Get patient ID from directory structure
        patient_dir = os.path.dirname(next(iter(phase_dirs.values())))
        patient_id = os.path.basename(os.path.dirname(patient_dir))

        # Load phases in order
        series_images = []
        for phase in self.required_phases:
            volume = self._load_sequence_images(phase_dirs[phase])
            if volume is not None:
                series_images.append(volume)
            else:
                raise RuntimeError(f"Failed to load {phase} for patient {patient_id}")

        if len(series_images) != len(self.required_phases):
            raise RuntimeError(f"Not all required phases loaded for patient {patient_id}")

        clinical_data = self._get_clinical_features(patient_id)

        data = {"images": series_images, "patient_id": patient_id, **clinical_data}
        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.patient_data)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=True)
