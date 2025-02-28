import glob
import logging
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BreastMRIDataset(Dataset):
    """Breast MRI Dataset Loader

    A specialized dataset loader for handling dynamic contrast-enhanced breast MRI sequences,
    with support for clinical features and molecular subtype information.

    Dataset Structure:
        root_dir/
        ├── Breast_MRI_001/
        │   └── patient_directory/
        │       ├── dynamic_sequence_1/
        │       │   └── *.dcm files
        │       ├── dynamic_sequence_2/
        │       │   └── *.dcm files
        │       └── ...
        ├── Breast_MRI_002/
        └── ...

    Features:
        1. Automatic handling of multi-level DICOM file structures
        2. Batch loading of dynamic sequences (5 timepoints per patient)
        3. Integration of clinical data and molecular subtype information
        4. Support for flexible data transformation pipelines
        5. Comprehensive data validation and error handling

    Molecular Subtype Mapping:
        - 0: 'luminal-like'
        - 1: 'ER/PR pos, HER2 pos'
        - 2: 'her2'
        - 3: 'trip neg'

    Return Format:
        Each sample returns a dictionary containing:
        - 'images': Tensor of shape [5, D, H, W] representing 3D images at 5 timepoints
        - 'patient_id': Patient identifier
        - 'molecular_subtype': Molecular subtype (if clinical data is provided)
        - 'clinical_features': Dictionary of additional clinical features (if specified)

    Args:
        root_dir (str): Root directory path containing the dataset
        clinical_data_path (str, optional): Path to Clinical_and_Other_Features.xlsx file
        clinical_features_columns (List[Tuple[str, str, str]], optional): List of clinical features to extract.
            Each tuple should contain (category, feature_name, description) matching the Excel file's
            multi-level column headers. For example:
            [
                ('Demographics', 'Date of Birth (Days)', '(Taking date of diagnosis as day 0)'),
                ('Demographics', 'Menopause (at diagnosis)', '{0 = pre, 1 = post, 2 = N/A}'),
            ]
        transform (callable, optional): Transform pipeline for image preprocessing
        patient_indices (List[int], optional): List of Breast_MRI_XXX indices to load
        max_workers (int): Maximum number of worker threads for parallel processing
        cache_size (int): Size of the LRU cache for DICOM reading

    Raises:
        FileNotFoundError: When root directory doesn't exist or no valid Breast_MRI_XXX directories found
        RuntimeError: When no valid patient data or insufficient dynamic sequences found
        ValueError: When patient indices are out of range or invalid
    """

    def __init__(
        self,
        root_dir: str,
        clinical_data_path: Optional[str] = None,
        clinical_label: Tuple[str, str, str] = (
            "Tumor Characteristics",
            "Mol Subtype",
            "{0 = luminal-like,\n1 = ER/PR pos, HER2 pos,\n2 = her2,\n3 = trip neg}",
        ),
        clinical_features_columns: Optional[List[Tuple[str, str, str]]] = None,
        transform: Optional[Callable] = None,
        patient_indices: Optional[List[int]] = None,
        max_workers: int = 4,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.clinical_label = clinical_label
        self.clinical_features_columns = [tuple(col) for col in clinical_features_columns] if clinical_features_columns else []
        self.clinical_ID_col = ("Patient Information", "Patient ID", "")
        self.max_workers = max_workers

        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        self._initialize_clinical_data(clinical_data_path)
        self._initialize_patient_data(patient_indices)

    def _initialize_clinical_data(self, clinical_data_path: Optional[str]) -> None:
        """Initialize clinical data from Excel file."""
        self.clinical_data = None
        if clinical_data_path is not None:
            try:
                self.clinical_data = pd.read_excel(clinical_data_path, header=[0, 1, 2])
                self.clinical_data.columns = [col[:-1] + ("",) if "Unnamed" in col[-1] else col for col in self.clinical_data.columns]
                logger.info(f"Successfully loaded clinical data from {clinical_data_path}")
            except Exception as e:
                logger.warning(f"Failed to load clinical data: {e}")
                self.clinical_data = None

    def _initialize_patient_data(self, patient_indices: Optional[List[int]]) -> None:
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

    def _get_valid_mri_dirs(self, patient_indices: Optional[List[int]]) -> List[str]:
        """
        Get valid MRI directories based on the specified patient indices.

        Args:
            patient_indices: Optional list of patient indices to filter directories

        Returns:
            List of valid MRI directory paths
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
        logger.info(f"Found {len(valid_mri_dirs)} valid Breast_MRI_XXX directories with indices: {available_indices}")

        # If patient indices are provided, filter Breast_MRI_XXX directories
        if patient_indices is not None:
            invalid_indices = [idx for idx in patient_indices if idx not in dir_indices]
            if invalid_indices:
                raise ValueError(f"Invalid patient indices: {invalid_indices}. Available indices are: {available_indices}")

            valid_mri_dirs = [dir_indices[idx] for idx in patient_indices]
            logger.info(f"Using {len(patient_indices)} specified Breast_MRI_XXX directories")

        return valid_mri_dirs

    def _get_valid_patient_dirs(self, mri_dirs: List[str]) -> List[str]:
        """
        Get valid patient directories from MRI directories.

        Args:
            mri_dirs: List of MRI directory paths

        Returns:
            List of valid patient directory paths
        """
        patient_dirs = []
        for mri_dir in mri_dirs:
            patient_subdirs = [d for d in glob.glob(os.path.join(mri_dir, "*")) if os.path.isdir(d)]

            # Only add directories that have at least 5 dynamic sequences
            for patient_dir in patient_subdirs:
                dyn_series = sorted([d for d in glob.glob(os.path.join(patient_dir, "*")) if "dyn" in d.lower()])
                if len(dyn_series) >= 5:
                    patient_dirs.append(patient_dir)
                else:
                    logger.warning(
                        f"Warning: Patient directory {os.path.basename(patient_dir)} has insufficient dynamic sequences (found {len(dyn_series)})"
                    )

        return sorted(patient_dirs)

    def _initialize_dynamic_sequences(self) -> List[List[str]]:
        """
        Initialize dynamic sequences for all valid patient directories.

        Returns:
            List of lists containing paths to dynamic sequence directories for each patient
        """
        patient_data = []

        # Preprocess: find 5 dynamic sequences for each patient
        for patient_dir in self.patient_dirs:
            dyn_series = sorted([d for d in glob.glob(os.path.join(patient_dir, "*")) if "dyn" in d.lower()])

            if len(dyn_series) >= 5:  # Ensure at least 5 dynamic sequences
                patient_data.append(dyn_series[:5])  # Take only the first 5
                logger.debug(f"Successfully loaded patient directory: {os.path.basename(patient_dir)}")

        if not patient_data:
            raise RuntimeError("No valid patient data found (requires at least 5 dynamic sequences per patient)")

        logger.info(f"Successfully loaded {len(patient_data)} valid patient datasets")
        return patient_data

    @staticmethod
    @lru_cache(maxsize=128)
    def _read_dicom_file(file_path: str) -> np.ndarray:
        """Read a DICOM file and return its pixel array."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return pydicom.dcmread(file_path).pixel_array.astype(np.float32)
        except Exception as e:
            logger.error(f"Error reading DICOM file {file_path}: {e}")
            return None

    def _load_sequence_images(self, series_path: str) -> Optional[np.ndarray]:
        """Load all DICOM images in a sequence directory."""
        dicom_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
        if not dicom_files:
            return None

        # Parallel load DICOM files
        slices = list(self.thread_pool.map(self._read_dicom_file, dicom_files))
        slices = [s for s in slices if s is not None]

        return np.stack(slices, axis=0) if slices else None

    def _get_clinical_features(self, patient_id: str) -> Dict[str, Any]:
        """Get clinical features for a patient."""
        if self.clinical_data is None:
            return {"molecular_subtype": None, "clinical_features": {}}

        try:
            patient_data = self.clinical_data[self.clinical_data[self.clinical_ID_col] == patient_id]
            if patient_data.empty:
                return {"molecular_subtype": None, "clinical_features": {}}

            molecular_subtype = patient_data[self.clinical_label].values[0]

            clinical_features: Dict[str, Any] = {}
            if self.clinical_features_columns:
                features_df = patient_data[self.clinical_features_columns]
                clinical_features = features_df.to_dict(orient="records")[0] if not features_df.empty else {}

            return {"molecular_subtype": molecular_subtype, "clinical_features": clinical_features}
        except Exception as e:
            logger.exception(f"Failed to load clinical data for patient {patient_id}: {e}")
            return {"molecular_subtype": None, "clinical_features": {}}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single patient's data."""
        patient_series = self.patient_data[idx]
        patient_dir = os.path.dirname(patient_series[0])
        patient_id = os.path.basename(os.path.dirname(patient_dir))

        # Load sequences in parallel
        series_images = []
        for series_path in patient_series:
            volume = self._load_sequence_images(series_path)
            if volume is not None:
                series_images.append(volume)

        if not series_images:
            raise RuntimeError(f"No valid DICOM images loaded for patient {patient_id}")

        # Stack sequences and apply transforms
        images = np.stack(series_images, axis=0)
        if self.transform:
            images = self.transform(images)
        else:
            images = torch.from_numpy(images).float()

        # Get clinical features
        clinical_data = self._get_clinical_features(patient_id)

        return {"images": images, "patient_id": patient_id, **clinical_data}

    def __len__(self) -> int:
        return len(self.patient_data)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=True)
