import glob
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

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

    Raises:
        FileNotFoundError: When root directory doesn't exist or no valid Breast_MRI_XXX directories found
        RuntimeError: When no valid patient data or insufficient dynamic sequences found
        ValueError: When patient indices are out of range or invalid
    """

    def __init__(
        self,
        root_dir: str,
        clinical_data_path: str,
        clinical_label: Tuple[str, str, str] = (
            "Tumor Characteristics",
            "Mol Subtype",
            "{0 = luminal-like,\n1 = ER/PR pos, HER2 pos,\n2 = her2,\n3 = trip neg}",
        ),
        clinical_features_columns: Optional[List[Tuple[str, str, str]]] = None,
        transform=None,
        patient_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            root_dir (str): Root directory path
            clinical_data_path (str): Path to Clinical_and_Other_Features.xlsx file
            clinical_features_columns (List[Tuple[str, str, str]]): List of clinical features to extract.
                Each tuple should contain (category, feature_name, description) matching the Excel file's
                multi-level column headers.  See class docstring for an example.
            clinical_label (List[Tuple[str, str, str]]):  Clinical label to use for molecular subtype.
                Default: ('Tumor Characteristics', 'Mol Subtype', '{0 = luminal-like,\n1 = ER/PR pos, HER2 pos,\n2 = her2,\n3 = trip neg}')
            transform (callable): Transform pipeline for all image preprocessing
            patient_indices (list): List of Breast_MRI_XXX indices to load
        """
        self.root_dir = root_dir
        self.transform = transform
        self.clinical_label = clinical_label  # Now expects a list of tuples
        self.clinical_features_columns = [tuple(col) for col in clinical_features_columns] if clinical_features_columns else []

        self.clinical_ID_col = ("Patient Information", "Patient ID", "")

        # Load clinical data if provided
        self.clinical_data = None
        if clinical_data_path is not None:
            try:
                # Read Excel file with multi-level headers
                self.clinical_data = pd.read_excel(clinical_data_path, header=[0, 1, 2])
                self.clinical_data.columns = [col[:-1] + ("",) if "Unnamed" in col[-1] else col for col in self.clinical_data.columns]
                logger.info(f"Successfully loaded clinical data from {clinical_data_path}")

                logger.debug(f"Clinical data shape: {self.clinical_data.shape}")
                logger.debug(f"Patient IDs in clinical data:\n{self.clinical_data[self.clinical_ID_col].tolist()}")

            except Exception as e:
                logger.warning(f"Failed to load clinical data: {e}")
                logger.debug(f"Attempted to load from: {clinical_data_path}")
                logger.debug(f"Exception details: {str(e)}")
                self.clinical_data = None

        # Find all directories matching Breast_MRI_XXX pattern
        all_mri_dirs = glob.glob(os.path.join(root_dir, "Breast_MRI_*"))
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
            raise FileNotFoundError(f"No valid Breast_MRI_XXX format directories found in {root_dir}")

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

        # Get all patient directories from selected Breast_MRI_XXX directories
        self.patient_dirs = []
        for mri_dir in valid_mri_dirs:
            patient_subdirs = [d for d in glob.glob(os.path.join(mri_dir, "*")) if os.path.isdir(d)]

            # Only add directories that have at least 5 dynamic sequences
            for patient_dir in patient_subdirs:
                dyn_series = sorted([d for d in glob.glob(os.path.join(patient_dir, "*")) if "dyn" in d.lower()])
                if len(dyn_series) >= 5:
                    self.patient_dirs.append(patient_dir)
                else:
                    logger.warning(
                        f"Warning: Patient directory {os.path.basename(patient_dir)} has insufficient dynamic sequences (found {len(dyn_series)})"
                    )

        self.patient_dirs = sorted(self.patient_dirs)

        if not self.patient_dirs:
            raise RuntimeError("No valid patient data with dynamic sequences found")

        logger.info(f"Found {len(self.patient_dirs)} patient directories")

        self.patient_data = []

        # Preprocess: find 5 dynamic sequences for each patient
        for patient_dir in self.patient_dirs:
            dyn_series = sorted([d for d in glob.glob(os.path.join(patient_dir, "*")) if "dyn" in d.lower()])

            if len(dyn_series) >= 5:  # Ensure at least 5 dynamic sequences
                self.patient_data.append(dyn_series[:5])  # Take only the first 5
                logger.debug(f"Successfully loaded patient directory: {os.path.basename(patient_dir)}")

        if not self.patient_data:
            raise RuntimeError("No valid patient data found (requires at least 5 dynamic sequences per patient)")

        logger.info(f"Successfully loaded {len(self.patient_data)} valid patient datasets")

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Load 5 dynamic sequences for a single patient.

        Returns:
        -------
        dict: {
            'images': tensor of shape [5, D, H, W],  # 3D images at 5 time points
            'patient_id': str,  # Patient ID
            'molecular_subtype': str,  # Molecular subtype if clinical data is available
            'clinical_features': dict  # Additional clinical features if available
        }
        """
        patient_series = self.patient_data[idx]

        # Get patient ID and parent directory
        patient_dir = os.path.dirname(patient_series[0])
        patient_id = os.path.basename(os.path.dirname(patient_dir))

        # Debug information
        logger.debug(f"Patient ID: {patient_id}")

        # Load DICOM images for 5 dynamic sequences
        series_images = []
        for series_path in patient_series:
            # Load all DICOM files in the sequence, handling potential errors
            dicom_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
            if not dicom_files:
                logger.warning(f"No DICOM files found in {series_path}. Skipping this sequence.")
                continue  # Skip this sequence if no DICOM files found

            slices = []
            for dcm_path in dicom_files:
                try:
                    dicom_data = pydicom.dcmread(dcm_path)
                    slices.append(dicom_data.pixel_array.astype(np.float32))
                except pydicom.errors.InvalidDicomError as e:
                    logger.error(f"Error reading DICOM file {dcm_path}: {e}")
                    # Depending on the situation, you might want to:
                    # 1.  Skip the file: continue
                    # 2.  Skip the entire series:  break
                    # 3.  Raise the exception if it's critical: raise
                    continue  # Skip this problematic file.

            if not slices:  # Check if any valid slices were read
                logger.warning(f"No valid slices read from {series_path}. Skipping this sequence.")
                continue
            volume = np.stack(slices, axis=0)
            series_images.append(volume)

        if not series_images:
            raise RuntimeError(f"No valid DICOM images loaded for patient {patient_id} at index {idx}.")

        # Stack all sequences together
        images = np.stack(series_images, axis=0)

        # Apply transform pipeline if provided
        if self.transform:
            images = self.transform(images)
        else:
            images = torch.from_numpy(images).float()

        # Initialize return dictionary
        result = {"images": images, "patient_id": patient_id, "molecular_subtype": None, "clinical_features": {}}

        # Add clinical data if available
        if self.clinical_data is not None:
            try:
                logger.debug(f"Looking for patient {patient_id} in clinical data")

                # Get molecular subtype, TODO: handle missing values and type conversion
                result["molecular_subtype"] = self.clinical_data[self.clinical_data[self.clinical_ID_col] == patient_id][self.clinical_label].values[
                    0
                ]

                # Get other clinical features, if requested, TODO: handle missing values and type conversion
                if self.clinical_features_columns:
                    clinical_features_df = self.clinical_data[self.clinical_data[self.clinical_ID_col] == patient_id][self.clinical_features_columns]
                    if clinical_features_df is not None:
                        # Convert DataFrame to a dictionary, handling MultiIndex
                        result["clinical_features"] = clinical_features_df.to_dict(orient="records")[0]
                    else:
                        logger.warning(f"No clinical features found for patient {patient_id}")
                else:
                    logger.debug("No clinical features columns requested.")

            except Exception as e:
                logger.exception(f"Failed to load clinical data for patient {patient_id}: {e}")
                logger.debug(f"Clinical data columns: {self.clinical_data.columns.tolist()}")
        return result
