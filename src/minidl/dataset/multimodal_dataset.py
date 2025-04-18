import glob
import logging
import os
import re
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .dataset_registry import DatasetRegistry

logger = logging.getLogger(__name__)


@DatasetRegistry.register("multimodal_dataset")
class MultiModalBreastMRIDataset(Dataset):
    """Multimodal Breast MRI Dataset Loader

    A specialized dataset loader that handles both dynamic contrast-enhanced breast MRI sequences
    and clinical features, supporting joint analysis of imaging and tabular data.

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
        3. Integration and preprocessing of clinical data from Excel file
        4. Support for flexible data transformation pipelines
        5. Comprehensive data validation and error handling

    Return Format:
        Each sample returns a dictionary containing:
        - 'images': Tensor of shape [3, D, H, W] representing 3D images at 3 timepoints
        - 'patient_id': Patient identifier
        - 'clinical_label': clinical_label (if clinical data is provided)

    Args:
        root_dir (str): Root directory path containing the dataset
        clinical_data_path (str): Path to Clinical_and_Other_Features.xlsx file
        clinical_label (list[str, str, str]): MultiIndex column tuple for the target variable.
            Example: ["Recurrence", "Recurrence event(s)", "{0 = no, 1 = yes}"]
        clinical_features_filter_dict (dict, optional): Dictionary to filter clinical columns by level.
            Example: {0: ["Recurrence", "Follow Up"]} will filter out all columns where the first level
            header is either 'Recurrence' or 'Follow Up'.
        clinical_features_exclude_columns (list, optional): List of specific clinical column tuples to exclude.
            Example: [("Recurrence", "Recurrence event(s)", "{0 = no, 1 = yes}")]
        transform (callable, optional): Transform pipeline for image preprocessing
        patient_indices (list[int], optional): List of Breast_MRI_XXX indices to load
        max_workers (int): Maximum number of worker threads for parallel processing

    Raises:
        FileNotFoundError: When root directory doesn't exist or no valid Breast_MRI_XXX directories found
        RuntimeError: When no valid patient data or required dynamic sequences not found
        ValueError: When patient indices are out of range or invalid
    """

    def __init__(
        self,
        root_dir: str,
        clinical_data_path: str,
        clinical_label: list[str, str, str],
        clinical_features_filter_dict: dict = None,
        clinical_features_exclude_columns: list = None,
        transform: Callable = None,
        patient_indices: list[int] = None,
        max_workers: int = 4,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.clinical_data_path = clinical_data_path
        self.clinical_label = tuple(clinical_label)
        self.clinical_features_filter_dict = clinical_features_filter_dict
        self.clinical_features_exclude_columns = clinical_features_exclude_columns
        self.clinical_ID_col = ("Patient Information", "Patient ID", "")
        self.max_workers = max_workers

        # Required sequence phases
        self.required_phases = ["Ph1", "Ph2", "Ph3"]

        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        # Load and process clinical data first (we need this for patient matching)
        self._initialize_clinical_data()

        # Then initialize patient directory data
        self._initialize_patient_data(patient_indices)

    def _initialize_clinical_data(self) -> None:
        """Initialize clinical data from Excel file and process it."""
        try:
            # Load data with multi-index header
            self.clinical_df = pd.read_excel(self.clinical_data_path, header=[0, 1, 2])

            # Clean up unnamed columns in multi-index
            new_cols = []
            for col in self.clinical_df.columns:
                if isinstance(col, tuple) and isinstance(col[-1], str) and "Unnamed" in col[-1]:
                    new_cols.append(col[:-1] + ("",))
                else:
                    new_cols.append(col)

            self.clinical_df.columns = pd.MultiIndex.from_tuples(new_cols)
            logger.info(f"Successfully loaded clinical data from {self.clinical_data_path}")
            logger.info(f"Clinical data shape: {self.clinical_df.shape}")

            # Process the clinical data
            self._process_clinical_features()

        except FileNotFoundError:
            logger.error(f"Error: Clinical data file not found at {self.clinical_data_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load clinical data: {e}")
            raise

    def _process_clinical_features(self) -> None:
        """Process clinical features: filter columns, handle missing values, encode categorical features."""
        # Store the original clinical dataframe for reference
        df_processed = self.clinical_df.copy()

        # Apply filters if provided
        if self.clinical_features_filter_dict is not None or self.clinical_features_exclude_columns is not None:
            df_processed = self._filter_clinical_features(df_processed)

        # Make sure target column is present
        if self.clinical_label not in df_processed.columns:
            logger.error(f"Error: Target column {self.clinical_label} not found in the clinical data")
            logger.info(f"Available columns: {df_processed.columns}")
            raise ValueError(f"Target column {self.clinical_label} not found in the clinical data")

        # Remove rows with missing target values
        initial_rows = len(df_processed)
        df_processed.dropna(subset=[self.clinical_label], inplace=True)
        rows_after_dropna = len(df_processed)
        logger.info(f"Removed {initial_rows - rows_after_dropna} rows with missing target values")

        # Convert target column to integer
        try:
            df_processed[self.clinical_label] = df_processed[self.clinical_label].astype(int)
            logger.info(f"Target column distribution:\n{df_processed[self.clinical_label].value_counts(normalize=True)}")
        except Exception as e:
            logger.error(f"Error converting target column to integer: {e}")
            raise

        # Store target values
        self.clinical_y = df_processed[self.clinical_label]

        # Remove target column from features and ensure ID column is kept separately
        id_series = df_processed[self.clinical_ID_col]

        # Remove both target column and ID column from features
        clinical_X = df_processed.drop(columns=[self.clinical_label, self.clinical_ID_col])

        # Store original columns for reference
        original_X_columns = clinical_X.columns.copy()

        # Flatten multi-index column names for easier processing
        flat_X_columns = ["_".join(filter(None, map(str, col))).strip("_") for col in original_X_columns]
        clinical_X.columns = flat_X_columns

        # Create a mapping from flat column names to original multi-level columns
        self.column_mapping = dict(zip(clinical_X.columns, original_X_columns))

        # Identify numerical and categorical columns
        numerical_cols = []
        categorical_cols = []

        for i, col in enumerate(clinical_X.columns):
            original_col_tuple = original_X_columns[i]
            # Try to convert to numeric, setting errors to 'coerce'
            clinical_X[col] = pd.to_numeric(clinical_X[col], errors="coerce")

            # Use original column data to determine type
            original_col_data_series = df_processed[original_col_tuple]

            if pd.api.types.is_numeric_dtype(clinical_X[col]) and not clinical_X[col].isnull().all():
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
                clinical_X[col] = original_col_data_series.astype(str)

        logger.info(f"Identified {len(numerical_cols)} numerical columns")
        logger.info(f"Identified {len(categorical_cols)} categorical columns")

        # Handle missing values in numerical columns
        if numerical_cols:
            self.num_imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
            clinical_X[numerical_cols] = self.num_imputer.fit_transform(clinical_X[numerical_cols])
            logger.info("Imputed missing values in numerical columns using mean")

            self.scaler = StandardScaler()
            clinical_X[numerical_cols] = self.scaler.fit_transform(clinical_X[numerical_cols])
            logger.info("Normalized numerical columns using StandardScaler")

        # Handle categorical columns
        if categorical_cols:
            # Impute missing values in categorical columns
            self.cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)
            clinical_X[categorical_cols] = self.cat_imputer.fit_transform(clinical_X[categorical_cols])
            logger.info("Imputed missing values in categorical columns with 'Missing'")

            # Store the mapping for categorical features before one-hot encoding
            self.categorical_mapping = {}
            for cat_col in categorical_cols:
                original_col = self.column_mapping[cat_col]
                unique_vals = clinical_X[cat_col].unique()
                for val in unique_vals:
                    # Create mappings for the one-hot encoded columns that will be created
                    encoded_col = f"{cat_col}_{val}"
                    self.categorical_mapping[encoded_col] = (original_col, val)

            # Apply one-hot encoding
            clinical_X = pd.get_dummies(clinical_X, columns=categorical_cols, drop_first=True, dummy_na=False, dtype=int)
            logger.info("Applied one-hot encoding to categorical columns")
            logger.info(f"Clinical data shape after encoding: {clinical_X.shape}")

            # Update the column mapping with the one-hot encoded columns
            for encoded_col in clinical_X.columns:
                if encoded_col in self.column_mapping:
                    continue  # Skip columns that already have a mapping

                # Try to find the base column name
                for cat_col in categorical_cols:
                    if encoded_col.startswith(cat_col + "_"):
                        value = encoded_col[len(cat_col) + 1 :]
                        self.column_mapping[encoded_col] = (self.column_mapping[cat_col], value)
                        break

            # Clean column names for compatibility
            logger.info("Cleaning column names for compatibility...")
            original_cols = clinical_X.columns.tolist()
            clinical_X.columns = clinical_X.columns.str.replace(r"\[|\]|\<", "_", regex=True)
            cleaned_cols = clinical_X.columns.tolist()

            # Update column mapping after cleaning column names
            cleaned_mapping = {}
            for orig, clean in zip(original_cols, cleaned_cols):
                if orig in self.column_mapping:
                    cleaned_mapping[clean] = self.column_mapping[orig]

            self.column_mapping = cleaned_mapping

        # Final check for non-numeric columns
        non_numeric_cols = clinical_X.select_dtypes(exclude=np.number).columns
        if len(non_numeric_cols) > 0:
            logger.warning(f"Found {len(non_numeric_cols)} non-numeric columns after processing. Attempting final conversion.")
            for col in non_numeric_cols:
                try:
                    clinical_X[col] = pd.to_numeric(clinical_X[col], errors="coerce")
                except Exception as e:
                    logger.error(f"Could not convert column {col} to numeric: {e}")

            final_numerical_cols = clinical_X.select_dtypes(include=np.number).columns
            if clinical_X.isnull().any().any():
                final_num_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                clinical_X[final_numerical_cols] = final_num_imputer.fit_transform(clinical_X[final_numerical_cols])

        # Store the processed clinical features and feature names
        self.clinical_X = clinical_X
        self.feature_names = clinical_X.columns.tolist()
        logger.info(f"Final clinical feature shape: {self.clinical_X.shape}")

        # Create a dictionary mapping patient IDs to their clinical data
        self.patient_to_clinical_data = {}
        for idx, (pid, _) in enumerate(zip(id_series, df_processed.iterrows())):
            if pid and not pd.isna(pid):
                clinical_features = self.clinical_X.iloc[idx].values.astype(np.float32)
                clinical_label = int(df_processed.iloc[idx][self.clinical_label])
                self.patient_to_clinical_data[str(pid)] = {
                    "clinical_features": clinical_features,
                    "clinical_label": clinical_label,
                }

        logger.info(f"Created clinical data mapping for {len(self.patient_to_clinical_data)} patients")

    def _filter_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter clinical features based on multi-level headers.

        Args:
            df: DataFrame with clinical data

        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()

        all_columns = filtered_df.columns
        columns_to_drop = []

        if self.clinical_features_filter_dict is not None:
            logger.info(f"Filtering columns based on level criteria: {self.clinical_features_filter_dict}")

            if isinstance(all_columns, pd.MultiIndex):
                for level, values_to_filter in self.clinical_features_filter_dict.items():
                    for value in values_to_filter:
                        level_matches = all_columns[all_columns.get_level_values(level) == value]
                        logger.info(f"Found {len(level_matches)} columns with '{value}' at level {level}")
                        columns_to_drop.extend(level_matches.tolist())
            else:
                logger.warning("Columns are not MultiIndex, applying simple filtering")
                for level, values_to_filter in self.clinical_features_filter_dict.items():
                    if level != 0:
                        logger.warning(f"Cannot filter on level {level} as columns are not MultiIndex")
                        continue
                    columns_to_drop.extend([col for col in all_columns if col in values_to_filter])

        if self.clinical_features_exclude_columns is not None:
            logger.info(f"Excluding specific columns: {self.clinical_features_exclude_columns}")
            for col in self.clinical_features_exclude_columns:
                if tuple(col) in all_columns:
                    columns_to_drop.append(tuple(col))

        columns_to_drop = list(set(columns_to_drop))

        if not columns_to_drop:
            logger.warning("No columns matched the filtering criteria")
        else:
            logger.info(f"Total columns to drop: {len(columns_to_drop)}")
            if columns_to_drop:
                sample_size = min(5, len(columns_to_drop))
                logger.info(f"Sample columns being dropped (first {sample_size}): {columns_to_drop[:sample_size]}")

        if columns_to_drop:
            filtered_df = filtered_df.drop(columns=columns_to_drop)

        logger.info(f"Original shape: {df.shape}, Filtered shape: {filtered_df.shape}")

        return filtered_df

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
        valid_patient_dirs = []

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

        # Now check which patients have clinical data available
        for patient_dir in patient_dirs:
            patient_id = os.path.basename(os.path.dirname(patient_dir))

            # Check if this patient ID exists in our clinical data
            if patient_id in self.patient_to_clinical_data:
                valid_patient_dirs.append(patient_dir)
            else:
                logger.warning(f"Skipping patient {patient_id}: no matching clinical data found")

        logger.info(f"Found {len(valid_patient_dirs)} patient directories with both MRI sequences and clinical data")
        return sorted(valid_patient_dirs)

    def _initialize_patient_data(self, patient_indices: list[int] | None) -> None:
        """Initialize patient directories and validate data."""
        # Find and validate directories
        all_mri_dirs = self._get_valid_mri_dirs(patient_indices)

        # Get patient directories with sufficient dynamic sequences and clinical data
        self.patient_dirs = self._get_valid_patient_dirs(all_mri_dirs)

        if not self.patient_dirs:
            raise RuntimeError("No valid patient data with dynamic sequences and clinical data found")

        logger.info(f"Found {len(self.patient_dirs)} valid patient directories")

        # Initialize patient data
        self.patient_data = self._initialize_dynamic_sequences()

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

                # Skip patients without valid clinical data
                if patient_id not in self.patient_to_clinical_data:
                    logger.warning(f"Skipping patient {patient_id}: no matching clinical data found")
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single patient's data with both MRI and clinical features."""
        phase_dirs = self.patient_data[idx]

        # Get patient ID from directory structure
        patient_dir = os.path.dirname(next(iter(phase_dirs.values())))
        patient_id = os.path.basename(os.path.dirname(patient_dir))

        # Load MRI phases in order
        series_images = []
        for phase in self.required_phases:
            volume = self._load_sequence_images(phase_dirs[phase])
            if volume is not None:
                series_images.append(volume)
            else:
                raise RuntimeError(f"Failed to load {phase} for patient {patient_id}")

        if len(series_images) != len(self.required_phases):
            raise RuntimeError(f"Not all required phases loaded for patient {patient_id}")

        # Get clinical data for this patient
        clinical_data = self.patient_to_clinical_data.get(
            patient_id,
            {
                "clinical_features": np.zeros(len(self.feature_names), dtype=np.float32),
                "clinical_label": -1,  # Use -1 as missing label indicator
            },
        )

        # Combine data
        data = {
            "images": series_images,
            "patient_id": patient_id,
            "clinical_features": clinical_data["clinical_features"],
            "clinical_label": clinical_data["clinical_label"],
        }

        # Apply transformations
        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.patient_data)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=True)

    def get_clinical_feature_names(self) -> list[str]:
        """Get the list of clinical feature names."""
        return self.feature_names


@DatasetRegistry.register("multimodal_dataset_v2")
class MultiModalBreastMRIDatasetV2(Dataset):
    """Multimodal Breast MRI Dataset Loader

    A specialized dataset loader that handles both dynamic contrast-enhanced breast MRI sequences
    and clinical features, supporting joint analysis of imaging and tabular data.

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
        3. Integration and preprocessing of clinical data from Excel file
        4. Support for flexible data transformation pipelines
        5. Comprehensive data validation and error handling
        6. Proper handling of train/validation/test splits to prevent data leakage

    Return Format:
        Each sample returns a dictionary containing:
        - 'images': Tensor of shape [3, D, H, W] representing 3D images at 3 timepoints
        - 'patient_id': Patient identifier
        - 'clinical_features': Preprocessed clinical features vector
        - 'clinical_label': clinical_label (if clinical data is provided)

    Args:
        root_dir (str): Root directory path containing the dataset
        clinical_data_path (str): Path to Clinical_and_Other_Features.xlsx file
        clinical_label (list[str, str, str]): MultiIndex column tuple for the target variable.
            Example: ["Recurrence", "Recurrence event(s)", "{0 = no, 1 = yes}"]
        clinical_features_filter_dict (dict, optional): Dictionary to filter clinical columns by level.
            Example: {0: ["Recurrence", "Follow Up"]} will filter out all columns where the first level
            header is either 'Recurrence' or 'Follow Up'.
        clinical_features_exclude_columns (list, optional): List of specific clinical column tuples to exclude.
            Example: [("Recurrence", "Recurrence event(s)", "{0 = no, 1 = yes}")]
        transform (callable, optional): Transform pipeline for image preprocessing
        patient_indices (list[int], optional): List of Breast_MRI_XXX indices to load
        max_workers (int): Maximum number of worker threads for parallel processing
        training_patient_ids (list[str], optional): List of patient IDs in the training set.
            Used to fit preprocessors (imputers, scalers) on training data only.
    """

    def __init__(
        self,
        root_dir: str,
        clinical_data_path: str,
        clinical_label: list[str, str, str],
        clinical_features_filter_dict: dict = None,
        clinical_features_exclude_columns: list = None,
        transform: Callable = None,
        patient_indices: list[int] = None,
        max_workers: int = 4,
        training_patient_ids: list[str] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.clinical_data_path = clinical_data_path
        self.clinical_label = tuple(clinical_label)
        self.clinical_features_filter_dict = clinical_features_filter_dict
        self.clinical_features_exclude_columns = clinical_features_exclude_columns
        self.clinical_ID_col = ("Patient Information", "Patient ID", "")
        self.max_workers = max_workers
        self.training_patient_ids = [f"Breast_MRI_{pid:03}" for pid in training_patient_ids]

        self.required_phases = ["Ph1", "Ph2", "Ph3"]

        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        self._initialize_clinical_data()

        self._initialize_patient_data(patient_indices)

    def _initialize_clinical_data(self) -> None:
        """Initialize clinical data from Excel file and process it."""
        try:
            self.clinical_df = pd.read_excel(self.clinical_data_path, header=[0, 1, 2])

            new_cols = []
            for col in self.clinical_df.columns:
                if isinstance(col, tuple) and isinstance(col[-1], str) and "Unnamed" in col[-1]:
                    new_cols.append(col[:-1] + ("",))
                else:
                    new_cols.append(col)

            self.clinical_df.columns = pd.MultiIndex.from_tuples(new_cols)
            logger.info(f"Successfully loaded clinical data from {self.clinical_data_path}")
            logger.info(f"Clinical data shape: {self.clinical_df.shape}")

            self._process_clinical_features()

        except FileNotFoundError:
            logger.error(f"Error: Clinical data file not found at {self.clinical_data_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load clinical data: {e}")
            raise

    def _process_clinical_features(self) -> None:
        """Process clinical features: filter columns, handle missing values, encode categorical features."""
        df_processed = self.clinical_df.copy()

        if self.clinical_features_filter_dict is not None or self.clinical_features_exclude_columns is not None:
            df_processed = self._filter_clinical_features(df_processed)

        if self.clinical_label not in df_processed.columns:
            logger.error(f"Error: Target column {self.clinical_label} not found in the clinical data")
            logger.info(f"Available columns: {df_processed.columns}")
            raise ValueError(f"Target column {self.clinical_label} not found in the clinical data")

        initial_rows = len(df_processed)
        df_processed.dropna(subset=[self.clinical_label], inplace=True)
        rows_after_dropna = len(df_processed)
        logger.info(f"Removed {initial_rows - rows_after_dropna} rows with missing target values")

        try:
            df_processed[self.clinical_label] = df_processed[self.clinical_label].astype(int)
            logger.info(f"Target column distribution:\n{df_processed[self.clinical_label].value_counts(normalize=True)}")
        except Exception as e:
            logger.error(f"Error converting target column to integer: {e}")
            raise

        self.clinical_y = df_processed[self.clinical_label]

        id_series = df_processed[self.clinical_ID_col]

        if self.training_patient_ids is not None:
            training_mask = id_series.isin(self.training_patient_ids)
            training_df = df_processed[training_mask]
            logger.info(f"Using {len(training_df)} patients for fitting preprocessors")
            if len(training_df) == 0:
                logger.warning("No matching patients found in training set. Check training_patient_ids.")
                training_df = df_processed
                logger.warning("Falling back to using all data for fitting preprocessors.")
        else:
            training_df = df_processed
            logger.warning("No training patient IDs provided. Using all patients for fitting preprocessors.")

        clinical_X = df_processed.drop(columns=[self.clinical_label, self.clinical_ID_col])
        training_X = training_df.drop(columns=[self.clinical_label, self.clinical_ID_col])

        original_X_columns = clinical_X.columns.copy()

        def normalize_column_name(col):
            if isinstance(col, tuple):
                col_str = "_".join(filter(None, map(str, col))).strip("_")
            else:
                col_str = str(col)
            return col_str

        self.normalize_column_name = normalize_column_name

        flat_X_columns = [normalize_column_name(col) for col in original_X_columns]
        clinical_X.columns = flat_X_columns
        training_X.columns = flat_X_columns

        self.column_mapping = dict(zip(clinical_X.columns, original_X_columns))

        numerical_cols = []
        categorical_cols = []

        for i, col in enumerate(clinical_X.columns):
            original_col_tuple = original_X_columns[i]
            clinical_X[col] = pd.to_numeric(clinical_X[col], errors="coerce")
            if not training_X.empty:
                training_X[col] = pd.to_numeric(training_X[col], errors="coerce")

            original_col_data_series = df_processed[original_col_tuple]

            if pd.api.types.is_numeric_dtype(clinical_X[col]) and not clinical_X[col].isnull().all():
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
                clinical_X[col] = original_col_data_series.astype(str)
                if not training_X.empty:
                    training_X[col] = training_df[original_col_tuple].astype(str)

        logger.info(f"Identified {len(numerical_cols)} numerical columns")
        logger.info(f"Identified {len(categorical_cols)} categorical columns")

        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

        if numerical_cols:
            self.num_cols_for_imputer = numerical_cols.copy()

        if numerical_cols:
            self.num_imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
            self.num_imputer.fit(training_X[numerical_cols])
            logger.info("Fitted numerical imputer on training data only")

            self.scaler = StandardScaler()
            self.scaler.fit(self.num_imputer.transform(training_X[numerical_cols]))
            logger.info("Fitted scaler on training data only")

        if categorical_cols:
            self.cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)
            self.cat_imputer.fit(training_X[categorical_cols])
            logger.info("Fitted categorical imputer on training data only")

            self.categorical_mapping = {}
            for cat_col in categorical_cols:
                original_col = self.column_mapping[cat_col]
                unique_vals = training_X[cat_col].unique()
                for val in unique_vals:
                    encoded_col = f"{cat_col}_{val}"
                    self.categorical_mapping[encoded_col] = (original_col, val)

            self.dummy_columns = {}
            for cat_col in categorical_cols:
                unique_vals = training_X[cat_col].unique()
                if len(unique_vals) > 0:
                    self.dummy_columns[cat_col] = [f"{cat_col}_{val}" for val in unique_vals[1:]]

        if categorical_cols:
            temp_X = training_X.copy()
            if numerical_cols:
                temp_X[numerical_cols] = self.num_imputer.transform(temp_X[numerical_cols])

            temp_X[categorical_cols] = self.cat_imputer.transform(temp_X[categorical_cols])
            dummies = pd.get_dummies(temp_X[categorical_cols], drop_first=True, dummy_na=False, dtype=int)

            if numerical_cols:
                final_X = pd.concat([temp_X[numerical_cols], dummies], axis=1)
            else:
                final_X = dummies

            self.feature_names = final_X.columns.tolist()
        else:
            self.feature_names = numerical_cols

        self.patient_to_clinical_data = {}

        for idx, (pid, _) in enumerate(zip(id_series, df_processed.iterrows())):
            if pid and not pd.isna(pid):
                patient_features = self._process_single_patient_features(df_processed.iloc[idx], original_X_columns, numerical_cols, categorical_cols)

                clinical_label = int(df_processed.iloc[idx][self.clinical_label])
                self.patient_to_clinical_data[str(pid)] = {
                    "clinical_features": patient_features,
                    "clinical_label": clinical_label,
                    "is_training": str(pid) in self.training_patient_ids if self.training_patient_ids else None,
                }

        logger.info(f"Created clinical data mapping for {len(self.patient_to_clinical_data)} patients")
        if self.training_patient_ids:
            training_count = sum(1 for pid, data in self.patient_to_clinical_data.items() if data["is_training"])
            logger.info(
                f"Dataset contains {training_count} training patients and \
                 {len(self.patient_to_clinical_data) - training_count} validation/test patients"
            )

    def _process_single_patient_features(self, patient_row, original_columns, numerical_cols, categorical_cols):
        """Process features for a single patient to avoid data leakage"""
        patient_data = {}
        for col in original_columns:
            if col != self.clinical_label and col != self.clinical_ID_col:
                patient_data[col] = patient_row[col]

        processed_features = {}

        if numerical_cols:
            num_data_dict = {}
            for col in self.num_cols_for_imputer:
                if col in self.column_mapping and self.column_mapping[col] in patient_data:
                    # Convert to numeric safely
                    try:
                        val = float(patient_data[self.column_mapping[col]])
                    except (ValueError, TypeError):
                        val = np.nan
                else:
                    val = np.nan
                num_data_dict[col] = val

            num_df = pd.DataFrame([num_data_dict])

            num_imputed = self.num_imputer.transform(num_df)

            num_scaled = self.scaler.transform(num_imputed)

            for i, col in enumerate(self.num_cols_for_imputer):
                processed_features[col] = num_scaled[0, i]

        if categorical_cols:
            for cat_col in categorical_cols:
                if cat_col in self.column_mapping:
                    orig_col = self.column_mapping[cat_col]

                    if orig_col in patient_data and not pd.isna(patient_data[orig_col]):
                        cat_value = str(patient_data[orig_col])
                    else:
                        cat_value = "Missing"

                    if cat_col in self.dummy_columns:
                        for dummy_col in self.dummy_columns[cat_col]:
                            val = dummy_col[len(cat_col) + 1 :]
                            if val == cat_value:
                                processed_features[dummy_col] = 1.0
                            else:
                                processed_features[dummy_col] = 0.0

        feature_array = np.zeros(len(self.feature_names), dtype=np.float32)

        for i, feat_name in enumerate(self.feature_names):
            feature_array[i] = processed_features.get(feat_name, 0.0)

        return feature_array

    def _filter_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter clinical features based on multi-level headers.

        Args:
            df: DataFrame with clinical data

        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()

        all_columns = filtered_df.columns
        columns_to_drop = []

        if self.clinical_features_filter_dict is not None:
            logger.info(f"Filtering columns based on level criteria: {self.clinical_features_filter_dict}")

            if isinstance(all_columns, pd.MultiIndex):
                for level, values_to_filter in self.clinical_features_filter_dict.items():
                    for value in values_to_filter:
                        level_matches = all_columns[all_columns.get_level_values(level) == value]
                        logger.info(f"Found {len(level_matches)} columns with '{value}' at level {level}")
                        columns_to_drop.extend(level_matches.tolist())
            else:
                logger.warning("Columns are not MultiIndex, applying simple filtering")
                for level, values_to_filter in self.clinical_features_filter_dict.items():
                    if level != 0:
                        logger.warning(f"Cannot filter on level {level} as columns are not MultiIndex")
                        continue
                    columns_to_drop.extend([col for col in all_columns if col in values_to_filter])

        if self.clinical_features_exclude_columns is not None:
            logger.info(f"Excluding specific columns: {self.clinical_features_exclude_columns}")
            for col in self.clinical_features_exclude_columns:
                if tuple(col) in all_columns:
                    columns_to_drop.append(tuple(col))

        columns_to_drop = list(set(columns_to_drop))

        if not columns_to_drop:
            logger.warning("No columns matched the filtering criteria")
        else:
            logger.info(f"Total columns to drop: {len(columns_to_drop)}")
            if columns_to_drop:
                sample_size = min(5, len(columns_to_drop))
                logger.info(f"Sample columns being dropped (first {sample_size}): {columns_to_drop[:sample_size]}")

        if columns_to_drop:
            filtered_df = filtered_df.drop(columns=columns_to_drop)

        logger.info(f"Original shape: {df.shape}, Filtered shape: {filtered_df.shape}")

        return filtered_df

    def _get_valid_mri_dirs(self, patient_indices: list[int] | None) -> list[str]:
        """
        Get valid MRI directories based on the specified patient indices.

        Args:
            patient_indices: Optional list of patient indices to filter directories

        Returns:
            list of valid MRI directory paths
        """
        all_mri_dirs = glob.glob(os.path.join(self.root_dir, "Breast_MRI_*"))
        valid_mri_dirs = []
        dir_indices = {}

        for mri_dir in all_mri_dirs:
            dir_name = os.path.basename(mri_dir)
            if os.path.isdir(mri_dir) and re.match(r"Breast_MRI_\d+$", dir_name):
                index = int(dir_name.split("_")[-1])
                dir_indices[index] = mri_dir
                valid_mri_dirs.append(mri_dir)

        if not valid_mri_dirs:
            raise FileNotFoundError(f"No valid Breast_MRI_XXX format directories found in {self.root_dir}")

        valid_mri_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[-1]))
        available_indices = sorted(dir_indices.keys())

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

        if not (("dyn" in folder_name.lower()) or ("vibrant" in folder_name.lower())):
            return None

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
        valid_patient_dirs = []

        for mri_dir in mri_dirs:
            patient_subdirs = [d for d in glob.glob(os.path.join(mri_dir, "*")) if os.path.isdir(d)]

            for patient_dir in patient_subdirs:
                sequence_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]

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

        for patient_dir in patient_dirs:
            patient_id = os.path.basename(os.path.dirname(patient_dir))

            if patient_id in self.patient_to_clinical_data:
                valid_patient_dirs.append(patient_dir)
            else:
                logger.warning(f"Skipping patient {patient_id}: no matching clinical data found")

        logger.info(f"Found {len(valid_patient_dirs)} patient directories with both MRI sequences and clinical data")
        return sorted(valid_patient_dirs)

    def _initialize_patient_data(self, patient_indices: list[int] | None) -> None:
        """Initialize patient directories and validate data."""
        all_mri_dirs = self._get_valid_mri_dirs(patient_indices)

        self.patient_dirs = self._get_valid_patient_dirs(all_mri_dirs)

        if not self.patient_dirs:
            raise RuntimeError("No valid patient data with dynamic sequences and clinical data found")

        logger.info(f"Found {len(self.patient_dirs)} valid patient directories")

        self.patient_data = self._initialize_dynamic_sequences()

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

                # Skip patients without valid clinical data
                if patient_id not in self.patient_to_clinical_data:
                    logger.warning(f"Skipping patient {patient_id}: no matching clinical data found")
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single patient's data with both MRI and clinical features."""
        phase_dirs = self.patient_data[idx]

        # Get patient ID from directory structure
        patient_dir = os.path.dirname(next(iter(phase_dirs.values())))
        patient_id = os.path.basename(os.path.dirname(patient_dir))

        # Load MRI phases in order
        series_images = []
        for phase in self.required_phases:
            volume = self._load_sequence_images(phase_dirs[phase])
            if volume is not None:
                series_images.append(volume)
            else:
                raise RuntimeError(f"Failed to load {phase} for patient {patient_id}")

        if len(series_images) != len(self.required_phases):
            raise RuntimeError(f"Not all required phases loaded for patient {patient_id}")

        # Get clinical data for this patient
        clinical_data = self.patient_to_clinical_data.get(
            patient_id,
            {
                "clinical_features": np.zeros(len(self.feature_names), dtype=np.float32),
                "clinical_label": -1,  # Use -1 as missing label indicator
                "is_training": False if self.training_patient_ids else None,
            },
        )

        # Combine data
        data = {
            "images": series_images,
            "patient_id": patient_id,
            "clinical_features": clinical_data["clinical_features"],
            "clinical_label": clinical_data["clinical_label"],
            "is_training": clinical_data["is_training"],
        }

        # Apply transformations
        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.patient_data)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=True)

    def get_clinical_feature_names(self) -> list[str]:
        """Get the list of clinical feature names."""
        return self.feature_names

    def filter_by_patient_ids(self, patient_ids: list[str]) -> None:
        """
        Filter the dataset to include only the specified patient IDs.

        Args:
            patient_ids: List of patient IDs to include
        """
        # Convert to set for faster lookup
        patient_id_set = set(patient_ids)

        filtered_data = []
        for phase_dirs in self.patient_data:
            # Get patient ID from directory structure
            patient_dir = os.path.dirname(next(iter(phase_dirs.values())))
            patient_id = os.path.basename(os.path.dirname(patient_dir))

            if patient_id in patient_id_set:
                filtered_data.append(phase_dirs)

        logger.info(f"Filtered dataset from {len(self.patient_data)} to {len(filtered_data)} patients")
        self.patient_data = filtered_data

    def get_preprocessor_status(self) -> dict:
        """
        Get information about the preprocessors used in this dataset.

        Returns:
            Dictionary with preprocessor information
        """
        info = {
            "feature_count": len(self.feature_names),
            "numerical_columns": self.numerical_cols,
            "categorical_columns": self.categorical_cols,
            "has_imputers": hasattr(self, "num_imputer") and hasattr(self, "cat_imputer"),
            "has_scaler": hasattr(self, "scaler"),
            "training_mode": self.training_patient_ids is not None,
        }

        if hasattr(self, "num_imputer") and self.numerical_cols:
            info["numerical_imputer_statistics"] = {
                "strategy": self.num_imputer.strategy,
                "sample_means": dict(
                    zip(
                        self.numerical_cols[:3],
                        self.num_imputer.statistics_[:3] if len(self.num_imputer.statistics_) > 3 else self.num_imputer.statistics_,
                    )
                ),
            }

        if hasattr(self, "scaler") and self.numerical_cols:
            info["scaler_statistics"] = {
                "sample_means": dict(zip(self.numerical_cols[:3], self.scaler.mean_[:3] if len(self.scaler.mean_) > 3 else self.scaler.mean_)),
                "sample_scales": dict(zip(self.numerical_cols[:3], self.scaler.scale_[:3] if len(self.scaler.scale_) > 3 else self.scaler.scale_)),
            }

        return info
