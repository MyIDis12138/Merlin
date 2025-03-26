import glob
import logging
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from torch.utils.data import Dataset

from .dataset_registry import DatasetRegistry

logger = logging.getLogger(__name__)


@DatasetRegistry.register("parallel_bmri_dataset")
class ParallelBreastMRIDataset(Dataset):
    """
    Optimized Breast MRI Dataset Loader with parallel processing capabilities

    This implementation focuses on speed optimization using:
    1. Parallel loading of DICOM files across patients
    2. Parallel processing of different phases for the same patient
    3. Batch preprocessing with thread pools
    4. Optional caching of processed data
    """

    def __init__(
        self,
        root_dir: str,
        clinical_label: list[str, str, str],
        clinical_data_path: str | None = None,
        clinical_features_columns: list[tuple[str, str, str]] | None = None,
        transform=None,
        patient_indices: list[int] | None = None,
        max_workers: int = None,
        preload: bool = False,
        batch_size: int = 1,
    ):
        """
        Initialize the dataset with optimized loading parameters.

        Args:
            root_dir: Root directory path containing the dataset
            clinical_label: Clinical label column info as [category, name, description]
            clinical_data_path: Path to clinical data Excel file
            clinical_features_columns: list of clinical features to extract
            transform: Transform pipeline for image preprocessing
            patient_indices: list of specific patient indices to include
            max_workers: Maximum number of worker threads (defaults to CPU count)
            cache_dir: Directory to cache processed data (None for no caching)
            preload: Whether to preload all data into memory during initialization
            batch_size: Batch size for parallel processing
        """
        self.root_dir = root_dir
        self.transform = transform
        self.clinical_label = tuple(clinical_label)
        self.clinical_features_columns = [tuple(col) for col in clinical_features_columns] if clinical_features_columns else []
        self.clinical_ID_col = ("Patient Information", "Patient ID", "")

        # Set up parallel processing parameters
        self.max_workers = max_workers if max_workers is not None else min(32, mp.cpu_count() * 2)
        self.preload = preload
        self.batch_size = batch_size
        self.preloaded_data = {}

        # Required MRI phases
        self.required_phases = ["Ph1", "Ph2", "Ph3"]

        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        # Initialize the dataset
        start_time = time.time()
        self._initialize_clinical_data(clinical_data_path)
        self._initialize_patient_data(patient_indices)
        logger.info(f"Dataset initialization took {time.time() - start_time:.2f} seconds")

        # Preload data if requested
        if self.preload:
            self._preload_all_data()

    def _initialize_clinical_data(self, clinical_data_path: str | None) -> None:
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

    def _initialize_patient_data(self, patient_indices: list[int] | None) -> None:
        """Initialize patient directories and validate data in parallel."""
        # Find and validate directories
        all_mri_dirs = self._get_valid_mri_dirs(patient_indices)

        # Get patient directories with sufficient dynamic sequences (in parallel)
        self.patient_dirs = self._get_valid_patient_dirs_parallel(all_mri_dirs)

        if not self.patient_dirs:
            raise RuntimeError("No valid patient data with dynamic sequences found")

        logger.info(f"Found {len(self.patient_dirs)} patient directories")

        # Initialize patient data
        self.patient_data = self._initialize_dynamic_sequences_parallel()

    def _get_valid_mri_dirs(self, patient_indices: list[int] | None) -> list[str]:
        """Get valid MRI directories based on the specified patient indices."""
        # Find all directories matching Breast_MRI_XXX pattern
        all_mri_dirs = glob.glob(os.path.join(self.root_dir, "Breast_MRI_*"))
        valid_mri_dirs = []
        dir_indices = {}

        # Process directories in parallel
        def validate_dir(mri_dir):
            dir_name = os.path.basename(mri_dir)
            if os.path.isdir(mri_dir) and re.match(r"Breast_MRI_\d+$", dir_name):
                index = int(dir_name.split("_")[-1])
                return index, mri_dir
            return None

        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(validate_dir, mri_dir) for mri_dir in all_mri_dirs]

            for future in as_completed(futures):
                result = future.result()
                if result:
                    index, mri_dir = result
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
        """Identify dynamic phase based on folder name."""
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

    def _process_patient_dir(self, mri_dir: str) -> list[str]:
        """Process a single patient directory to find valid sequences."""
        valid_dirs = []
        patient_subdirs = [d for d in glob.glob(os.path.join(mri_dir, "*")) if os.path.isdir(d)]

        for patient_dir in patient_subdirs:
            sequence_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]

            # Check phases in parallel
            with ThreadPoolExecutor(max_workers=min(len(sequence_dirs), 16)) as executor:
                phase_results = list(executor.map(self._identify_phase, sequence_dirs))

            phases_found = set(phase_results)

            if all(phase in phases_found for phase in self.required_phases):
                valid_dirs.append(patient_dir)
            else:
                missing_phases = set(self.required_phases) - phases_found
                logger.warning(f"Patient directory {os.path.basename(patient_dir)} is missing phases: {missing_phases}")

        return valid_dirs

    def _get_valid_patient_dirs_parallel(self, mri_dirs: list[str]) -> list[str]:
        """Get valid patient directories from MRI directories using parallel processing."""
        patient_dirs = []

        # Process MRI directories in parallel batches
        batch_size = min(len(mri_dirs), self.batch_size)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(0, len(mri_dirs), batch_size):
                batch = mri_dirs[i : i + batch_size]
                futures.extend([executor.submit(self._process_patient_dir, mri_dir) for mri_dir in batch])

            for future in as_completed(futures):
                patient_dirs.extend(future.result())

        return sorted(patient_dirs)

    def _process_patient_sequences(self, patient_dir: str) -> dict[str, str]:
        """Process sequences for a single patient directory."""
        sequence_dirs = [d for d in glob.glob(os.path.join(patient_dir, "*")) if os.path.isdir(d)]
        phase_to_dir = {}

        # Identify phases in parallel
        with ThreadPoolExecutor(max_workers=min(len(sequence_dirs), 8)) as executor:
            phase_results = list(executor.map(lambda dir_path: (dir_path, self._identify_phase(dir_path)), sequence_dirs))

        for dir_path, phase in phase_results:
            if phase in self.required_phases:
                phase_to_dir[phase] = dir_path

        return phase_to_dir if all(phase in phase_to_dir for phase in self.required_phases) else None

    def _initialize_dynamic_sequences_parallel(self) -> list[dict[str, str]]:
        """Initialize dynamic sequences for all valid patient directories in parallel."""
        patient_data = []

        # Prepare patient directory batches
        batch_size = min(len(self.patient_dirs), self.batch_size)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(0, len(self.patient_dirs), batch_size):
                batch = self.patient_dirs[i : i + batch_size]
                futures.extend([executor.submit(self._process_patient_sequences, patient_dir) for patient_dir in batch])

            for future, patient_dir in zip(as_completed(futures), self.patient_dirs):
                phase_to_dir = future.result()
                if phase_to_dir:
                    patient_id = os.path.basename(os.path.dirname(patient_dir))

                    # Skip patients without valid clinical labels if clinical data is available
                    if self.clinical_data is not None:
                        try:
                            patient_data_row = self.clinical_data[self.clinical_data[self.clinical_ID_col] == patient_id]
                            if patient_data_row.empty:
                                logger.warning(f"Skipping patient {patient_id}: no matching clinical data")
                                continue

                            clinical_label_value = patient_data_row[self.clinical_label].values[0]
                            if np.isnan(clinical_label_value):
                                logger.warning(f"Skipping patient {patient_id}: invalid clinical label")
                                continue
                        except Exception as e:
                            logger.warning(f"Skipping patient {patient_id}: {e}")
                            continue

                    patient_data.append(phase_to_dir)
                    logger.debug(f"Loaded patient: {patient_id}")

        if not patient_data:
            raise RuntimeError("No valid patient data found with all required dynamic phases")

        logger.info(f"Successfully loaded {len(patient_data)} valid patient datasets")
        return patient_data

    def _read_dicom_file(self, file_path: str) -> np.ndarray | None:
        """Read a DICOM file and return its pixel array."""
        try:
            ds = pydicom.dcmread(file_path, force=True)
            return ds.pixel_array.astype(np.float32)
        except Exception as e:
            logger.error(f"Error reading DICOM file {file_path}: {e}")
            return None

    def _load_sequence_images_parallel(self, series_path: str) -> np.ndarray | None:
        """Load all DICOM images in a sequence directory using parallel processing."""
        dicom_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
        if not dicom_files:
            return None

        # Load slices in parallel
        slices = []
        batch_size = min(len(dicom_files), 25)  # Process in batches to avoid overwhelming the thread pool

        with ThreadPoolExecutor(max_workers=min(len(dicom_files), 25)) as executor:
            future_slices = []
            for i in range(0, len(dicom_files), batch_size):
                batch = dicom_files[i : i + batch_size]
                future_slices.extend(executor.map(self._read_dicom_file, batch))

            slices = [s for s in future_slices if s is not None]

        if not slices:
            return None

        return np.stack(slices, axis=0)

    def _preload_all_data(self) -> None:
        """Preload all patient data into memory."""
        logger.info(f"Preloading data for {len(self.patient_data)} patients...")
        start_time = time.time()

        # Process in batches
        batch_size = min(len(self.patient_data), self.batch_size)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i in range(0, len(self.patient_data), batch_size):
                batch_indices = list(range(i, min(i + batch_size, len(self.patient_data))))
                for idx in batch_indices:
                    futures[executor.submit(self._load_patient_data, idx)] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    self.preloaded_data[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error preloading data for patient at index {idx}: {e}")

        logger.info(f"Preloaded {len(self.preloaded_data)} patients in {time.time() - start_time:.2f} seconds")

    def _load_patient_data(self, idx: int) -> dict[str, Any]:
        """Load data for a single patient with optimized parallel loading."""
        phase_dirs = self.patient_data[idx]

        # Get patient ID from directory structure
        patient_dir = os.path.dirname(next(iter(phase_dirs.values())))
        patient_id = os.path.basename(os.path.dirname(patient_dir))

        # Load phases in parallel
        series_images = []
        with ThreadPoolExecutor(max_workers=min(len(self.required_phases), 3)) as executor:
            future_volumes = {executor.submit(self._load_sequence_images_parallel, phase_dirs[phase]): phase for phase in self.required_phases}

            for future in as_completed(future_volumes):
                phase = future_volumes[future]
                try:
                    volume = future.result()
                    if volume is None:
                        raise RuntimeError(f"Failed to load {phase} for patient {patient_id}")
                    series_images.append(volume)
                except Exception as e:
                    logger.error(f"Error loading {phase} for patient {patient_id}: {e}")
                    raise RuntimeError(f"Failed to load {phase} for patient {patient_id}: {e}")

        if len(series_images) != len(self.required_phases):
            raise RuntimeError(f"Not all required phases loaded for patient {patient_id}")

        # Get clinical data
        clinical_data = self._get_clinical_features(patient_id)

        # Create data dictionary
        data = {"images": series_images, "patient_id": patient_id, **clinical_data}

        return data

    def _get_clinical_features(self, patient_id: str) -> dict[str, Any]:
        """Get clinical features for a patient."""
        if self.clinical_data is None:
            return {"clinical_label": None, "clinical_features": {}}

        try:
            patient_data = self.clinical_data[self.clinical_data[self.clinical_ID_col] == patient_id]
            if patient_data.empty:
                return {"clinical_label": None, "clinical_features": {}}

            clinical_label = int(patient_data[self.clinical_label].values[0])

            clinical_features = {}
            if self.clinical_features_columns:
                features_df = patient_data[self.clinical_features_columns]
                clinical_features = features_df.to_dict(orient="records")[0] if not features_df.empty else {}

            return {"clinical_label": clinical_label, "clinical_features": clinical_features}
        except Exception as e:
            logger.exception(f"Failed to load clinical data for patient {patient_id}: {e}")
            return {"clinical_label": None, "clinical_features": {}}

    def _apply_transform(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply transforms safely with TorchIO compatibility.

        Args:
            data: Dictionary containing 'images' key

        Returns:
            Transformed data dictionary or TorchIO Subject
        """
        if self.transform is None:
            return data

        # Check if the transform is MRITorchIOPipeline with return_subject=True
        if hasattr(self.transform, "transforms"):
            for t in self.transform.transforms:
                if getattr(t, "return_subject", False):
                    # Apply transforms and return a TorchIO Subject
                    return t(data)

        # Regular transform
        return self.transform(data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single patient's data with caching and preloading optimizations."""
        # Check if data is preloaded
        if self.preload and idx in self.preloaded_data:
            data = self.preloaded_data[idx]
        else:
            data = self._load_patient_data(idx)

        if self.transform:
            data = self.transform(data.copy())

        return data

    def __len__(self) -> int:
        return len(self.patient_data)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=False)
