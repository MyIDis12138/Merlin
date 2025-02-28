import logging
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import pydicom
import torch
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid

from minidl.dataset.breast_mri_dataset import BreastMRIDataset
from minidl.transforms.mri_transforms import MRITransformPipeline, Normalize, ToTensor

logger = logging.getLogger(__name__)


class TestDicomDataCreation:
    """Helper class for creating test DICOM data"""

    @staticmethod
    def create_test_hierarchy(base_dir):
        """Create a test directory hierarchy with DICOM files"""
        # Create 3 Breast_MRI_XXX directories
        for i in range(1, 4):
            mri_dir = os.path.join(base_dir, f"Breast_MRI_{i:03d}")
            os.makedirs(mri_dir)

            # Create a patient directory in each Breast_MRI_XXX
            patient_dir = os.path.join(mri_dir, "01-01-1990-NA-MRI BREAST WWO-12345")
            os.makedirs(patient_dir)

            # Create 5 dynamic sequence directories
            for j in range(5):
                dyn_dir = os.path.join(patient_dir, f"{600+j}.000000-Ph{j+1}ax 3d dyn MP-{j:05d}")
                os.makedirs(dyn_dir)

                # Create dummy DICOM files with varying intensities
                for k in range(3):  # 3 slices per sequence
                    dcm_path = os.path.join(dyn_dir, f"slice_{k}.dcm")
                    intensity_base = 100 * (j + 1)  # Different base intensity for each sequence
                    TestDicomDataCreation._create_test_dicom(dcm_path, k, intensity_base)

    @staticmethod
    def create_test_clinical_data(file_path):
        """Create a test clinical data Excel file with multi-level headers"""
        # Create multi-level columns
        columns = pd.MultiIndex.from_tuples(
            [
                ("Patient Information", "Patient ID", ""),
                ("Tumor Characteristics", "Mol Subtype", "{0 = luminal-like,\n1 = ER/PR pos, HER2 pos,\n2 = her2,\n3 = trip neg}"),
                ("Demographics", "Date of Birth (Days)", "(Taking date of diagnosis as day 0)"),
                ("Demographics", "Menopause (at diagnosis)", "{0 = pre,\n1 = post,\n2 = N/A}"),
                ("Demographics", "Race and Ethnicity", "{1 = white,\n2 = black,\n3 = asian}"),
                ("Tumor Characteristics", "Nottingham grade", "1=low 2=intermediate 3=high"),
                ("Recurrence", "Recurrence event(s)", "{0 = no, 1 = yes}"),
            ]
        )

        # Create test data
        data = [
            ["Breast_MRI_001", 0, -15000, 0, 1, 2, 0],  # Patient 1
            ["Breast_MRI_002", 1, -14000, 1, 2, 1, 1],  # Patient 2
            ["Breast_MRI_003", 2, -13000, 0, 3, 3, 0],  # Patient 3
        ]

        # Create DataFrame with multi-level columns
        df = pd.DataFrame(data, columns=columns)
        df.to_excel(file_path)
        return df

    @staticmethod
    def _create_test_dicom(filepath, instance_number, intensity_base):
        """Create a test DICOM file with minimal required attributes"""
        # Create test data with varying intensities
        x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        test_array = np.uint16((intensity_base + intensity_base * x + intensity_base * y).clip(0, 65535))

        # Create the FileDataset instance
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Add required DICOM attributes
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = "MR"
        ds.SeriesNumber = "1"
        ds.InstanceNumber = str(instance_number)
        ds.ImagePositionPatient = [0, 0, instance_number * 5.0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 5.0

        # Set creation date/time
        dt = datetime.now()
        ds.ContentDate = dt.strftime("%Y%m%d")
        ds.ContentTime = dt.strftime("%H%M%S.%f")

        # Set pixel data attributes
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = test_array.shape[0]
        ds.Columns = test_array.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = test_array.tobytes()

        ds.save_as(filepath)


class TestBreastMRIDataset(unittest.TestCase):
    """Test the BreastMRIDataset class"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.clinical_data_path = os.path.join(cls.test_dir, "clinical_data.xlsx")
        TestDicomDataCreation.create_test_hierarchy(cls.test_dir)
        cls.clinical_data = TestDicomDataCreation.create_test_clinical_data(cls.clinical_data_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)

    def test_initialization(self):
        """Test dataset initialization"""
        dataset = BreastMRIDataset(root_dir=self.test_dir, clinical_data_path=self.clinical_data_path)
        self.assertIsInstance(dataset.thread_pool, ThreadPoolExecutor)
        self.assertEqual(dataset.max_workers, 4)  # Default value
        self.assertEqual(len(dataset), 3)

    def test_clinical_data_loading(self):
        """Test clinical data loading and processing"""
        clinical_features = [
            ("Demographics", "Date of Birth (Days)", "(Taking date of diagnosis as day 0)"),
            ("Demographics", "Menopause (at diagnosis)", "{0 = pre,\n1 = post,\n2 = N/A}"),
        ]

        dataset = BreastMRIDataset(root_dir=self.test_dir, clinical_data_path=self.clinical_data_path, clinical_features_columns=clinical_features)

        sample = dataset[0]
        self.assertIn("molecular_subtype", sample)
        self.assertIn("clinical_features", sample)
        self.assertEqual(len(sample["clinical_features"]), len(clinical_features))

    def test_patient_filtering(self):
        """Test patient index filtering"""
        dataset = BreastMRIDataset(root_dir=self.test_dir, patient_indices=[1])
        self.assertEqual(len(dataset), 1)

        with self.assertRaises(ValueError):
            BreastMRIDataset(root_dir=self.test_dir, patient_indices=[999])

    def test_parallel_loading(self):
        """Test parallel loading of DICOM files"""
        dataset = BreastMRIDataset(root_dir=self.test_dir, max_workers=2)
        sample = dataset[0]
        self.assertEqual(sample["images"].shape[0], 5)  # 5 time points
        self.assertEqual(sample["images"].shape[1], 3)  # 3 slices

    def test_cache_mechanism(self):
        """Test DICOM file caching mechanism"""
        dataset = BreastMRIDataset(root_dir=self.test_dir)

        # First access should cache the result
        first_sample = dataset[0]

        # Second access should use cached data
        second_sample = dataset[0]

        # Verify data consistency
        self.assertTrue(torch.equal(first_sample["images"], second_sample["images"]))
        self.assertEqual(first_sample["patient_id"], second_sample["patient_id"])
        self.assertEqual(first_sample["molecular_subtype"], second_sample["molecular_subtype"])
        self.assertEqual(first_sample["clinical_features"], second_sample["clinical_features"])

        # Test cache with different indices
        sample_1 = dataset[1]
        sample_1_again = dataset[1]
        self.assertTrue(torch.equal(sample_1["images"], sample_1_again["images"]))

    def test_transform_pipeline(self):
        """Test transform pipeline integration"""
        # Test with multiple transforms
        transform = MRITransformPipeline([Normalize(range_min=0, range_max=1), ToTensor()])

        dataset = BreastMRIDataset(root_dir=self.test_dir, transform=transform)

        sample = dataset[0]

        # Check normalization range
        self.assertTrue(torch.all(sample["images"] >= 0))
        self.assertTrue(torch.all(sample["images"] <= 1))

        # Check tensor properties
        self.assertIsInstance(sample["images"], torch.Tensor)
        self.assertEqual(sample["images"].dtype, torch.float32)
        self.assertEqual(len(sample["images"].shape), 4)  # [5, D, H, W]

        # Test without transform
        dataset_no_transform = BreastMRIDataset(root_dir=self.test_dir)
        sample_no_transform = dataset_no_transform[0]
        self.assertIsInstance(sample_no_transform["images"], torch.Tensor)

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with non-existent directory
        with self.assertRaises(FileNotFoundError):
            BreastMRIDataset(root_dir="/nonexistent/path")

        # Test with invalid clinical data path
        with self.assertLogs(level="WARNING") as log:
            dataset = BreastMRIDataset(root_dir=self.test_dir, clinical_data_path="invalid_path.xlsx")
            self.assertIsNone(dataset.clinical_data)
            self.assertTrue(any("Failed to load clinical data" in message for message in log.output))

        # Test with invalid patient indices
        with self.assertRaises(ValueError) as context:
            BreastMRIDataset(root_dir=self.test_dir, patient_indices=[999])
        self.assertTrue("Invalid patient indices" in str(context.exception))

        # Test with empty patient indices
        with self.assertRaises(RuntimeError) as context:
            BreastMRIDataset(root_dir=self.test_dir, patient_indices=[])
        self.assertTrue("No valid patient data" in str(context.exception))

    def test_clinical_features_edge_cases(self):
        """Test clinical features handling in edge cases"""
        # Test with invalid clinical features
        invalid_features = [("NonExistent", "Feature", "Description"), ("Demographics", "InvalidFeature", "")]

        # Test with invalid features - should log warning but not fail
        with self.assertLogs(level="WARNING") as log:
            dataset = BreastMRIDataset(root_dir=self.test_dir, clinical_data_path=self.clinical_data_path, clinical_features_columns=invalid_features)

            sample = dataset[0]
            self.assertIn("clinical_features", sample)
            # Invalid features should result in an empty dictionary
            self.assertEqual(len(sample["clinical_features"]), 0)
            # Verify warning was logged
            self.assertTrue(any("Failed to load clinical data" in message for message in log.output))

        # Test with partially valid features
        with self.assertLogs(level="WARNING") as log:
            mixed_features = [
                ("Demographics", "Date of Birth (Days)", "(Taking date of diagnosis as day 0)"),  # Valid
                ("NonExistent", "Feature", "Description"),  # Invalid
            ]

            dataset_mixed = BreastMRIDataset(
                root_dir=self.test_dir, clinical_data_path=self.clinical_data_path, clinical_features_columns=mixed_features
            )

            sample_mixed = dataset_mixed[0]
            self.assertIn("clinical_features", sample_mixed)
            # When any feature is invalid, all features are skipped for safety
            self.assertEqual(len(sample_mixed["clinical_features"]), 0)
            # Verify warning was logged
            self.assertTrue(any("Failed to load clinical data" in message for message in log.output))

        # Test with valid features
        valid_features = [
            ("Demographics", "Date of Birth (Days)", "(Taking date of diagnosis as day 0)"),
            ("Demographics", "Menopause (at diagnosis)", "{0 = pre,\n1 = post,\n2 = N/A}"),
        ]

        dataset_valid = BreastMRIDataset(root_dir=self.test_dir, clinical_data_path=self.clinical_data_path, clinical_features_columns=valid_features)

        sample_valid = dataset_valid[0]
        self.assertIn("clinical_features", sample_valid)
        # Valid features should all be included
        self.assertEqual(len(sample_valid["clinical_features"]), len(valid_features))

        # Test with empty clinical features
        dataset_empty = BreastMRIDataset(root_dir=self.test_dir, clinical_data_path=self.clinical_data_path, clinical_features_columns=[])
        sample_empty = dataset_empty[0]
        self.assertEqual(sample_empty["clinical_features"], {})

        # Test with None clinical features
        dataset_none = BreastMRIDataset(root_dir=self.test_dir, clinical_data_path=self.clinical_data_path, clinical_features_columns=None)
        sample_none = dataset_none[0]
        self.assertEqual(sample_none["clinical_features"], {})

    def test_data_consistency(self):
        """Test data consistency across different access patterns"""
        dataset = BreastMRIDataset(root_dir=self.test_dir, clinical_data_path=self.clinical_data_path)

        # Test sequential access
        first_pass = [dataset[i] for i in range(len(dataset))]
        second_pass = [dataset[i] for i in range(len(dataset))]

        for f, s in zip(first_pass, second_pass):
            self.assertTrue(torch.equal(f["images"], s["images"]))
            self.assertEqual(f["patient_id"], s["patient_id"])

        # Test random access
        indices = [2, 0, 1]  # Reversed order
        random_access = [dataset[i] for i in indices]
        self.assertEqual(len(random_access), len(indices))

        # Verify data integrity
        for sample in random_access:
            self.assertIsNotNone(sample["images"])
            self.assertIsNotNone(sample["patient_id"])

    def test_resource_cleanup(self):
        """Test proper resource cleanup"""
        dataset = BreastMRIDataset(root_dir=self.test_dir)
        self.assertIsInstance(dataset.thread_pool, ThreadPoolExecutor)

        # Test thread pool functionality before cleanup
        self.assertFalse(dataset.thread_pool._shutdown)

        # Trigger cleanup
        dataset.__del__()

        # Verify thread pool is shut down
        self.assertTrue(dataset.thread_pool._shutdown)

        # Verify cleanup is idempotent
        dataset.__del__()
        self.assertTrue(dataset.thread_pool._shutdown)


if __name__ == "__main__":
    unittest.main()
