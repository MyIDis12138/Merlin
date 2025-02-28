import logging
import os
import shutil
import tempfile
import unittest
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
        """Create a test clinical data Excel file"""
        # Create single-level columns first
        columns = [
            "Patient_ID",  # Will be mapped to ('Patient Information', 'Patient ID', '')
            "Mol_Subtype",  # Will be mapped to ('Tumor Characteristics', 'Mol Subtype', ...)
            "Date_of_Birth",  # Will be mapped to ('Demographics', 'Date of Birth (Days)', ...)
            "Menopause",  # Will be mapped to ('Demographics', 'Menopause (at diagnosis)', ...)
            "Race_and_Ethnicity",  # Will be mapped to ('Demographics', 'Race and Ethnicity', ...)
            "Nottingham_grade",  # Will be mapped to ('Tumor Characteristics', 'Nottingham grade', ...)
            "Recurrence_events",  # Will be mapped to ('Recurrence', 'Recurrence event(s)', ...)
        ]

        # Create test data with Breast_MRI_XXX format patient IDs
        data = [
            ["Breast_MRI_001", 0, -15000, 0, 1, 2, 0],  # Patient 1: luminal-like
            ["Breast_MRI_002", 1, -14000, 1, 2, 1, 1],  # Patient 2: ER/PR pos, HER2 pos
            ["Breast_MRI_003", 2, -13000, 0, 3, 3, 0],  # Patient 3: her2
        ]

        # Create DataFrame with single-level columns
        df = pd.DataFrame(data, columns=columns)

        # Save to Excel
        df.to_excel(file_path, index=False)

        # Create the multi-level mapping that will be used in the dataset
        multi_level_columns = {
            "Patient_ID": ("Patient Information", "Patient ID", ""),
            "Mol_Subtype": (
                "Tumor Characteristics",
                "Mol Subtype",
                "{0 = luminal-like,\n1 = ER/PR pos, HER2 pos,\n2 = her2,\n3 = trip neg}",
            ),
            "Date_of_Birth": ("Demographics", "Date of Birth (Days)", "(Taking date of diagnosis as day 0)"),
            "Menopause": ("Demographics", "Menopause (at diagnosis)", "{0 = pre,\n1 = post,\n2 = N/A}"),
            "Race_and_Ethnicity": (
                "Demographics",
                "Race and Ethnicity",
                "{0 = N/A\n1 = white,\n2 = black,\n3 = asian,\n4 = native,\n5 = hispanic,\n6 = multi,\n7 = hawa,    8 = amer indian}",
            ),
            "Nottingham_grade": ("Tumor Characteristics", "Nottingham grade", "1=low 2=intermediate 3=high\n"),
            "Recurrence_events": ("Recurrence", "Recurrence event(s)", "{0 = no, 1 = yes}"),
        }

        # Print debug information
        logger.debug(f"Created clinical data with columns: {df.columns.tolist()}")
        logger.debug(f"First row of clinical data: {df.iloc[0].to_dict()}")
        logger.debug(f"Patient IDs in clinical data: {df['Patient_ID'].tolist()}")
        logger.debug(f"Multi-level column mapping: {multi_level_columns}")

        return df, multi_level_columns  # Return both the DataFrame and the mapping

    @staticmethod
    def _create_test_dicom(filepath, instance_number, intensity_base):
        """Create a test DICOM file with minimal required attributes"""
        # Create test data with varying intensities
        x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        test_array = np.uint16((intensity_base + intensity_base * x + intensity_base * y).clip(0, 65535))

        # Create the FileDataset instance
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
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


class TestBreastMRIDatasetStructure(unittest.TestCase):
    """Test the dataset structure and initialization"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.clinical_data_path = os.path.join(cls.test_dir, "clinical_data.xlsx")
        TestDicomDataCreation.create_test_hierarchy(cls.test_dir)
        TestDicomDataCreation.create_test_clinical_data(cls.clinical_data_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_directory_validation(self):
        """Test directory structure validation"""
        # Test valid directory
        dataset = BreastMRIDataset(root_dir=self.test_dir)
        self.assertEqual(len(dataset), 3)

        # Test nonexistent directory
        with self.assertRaises(FileNotFoundError):
            BreastMRIDataset(root_dir="/nonexistent/path")

    def test_patient_indices(self):
        """Test patient index filtering"""
        # Test valid indices
        dataset = BreastMRIDataset(root_dir=self.test_dir, patient_indices=[1, 2])  # Using actual directory numbers
        self.assertEqual(len(dataset), 2)

        # Test invalid index
        with self.assertRaises(ValueError):
            BreastMRIDataset(root_dir=self.test_dir, patient_indices=[10])

    def test_clinical_data_loading(self):
        """Test clinical data loading and mapping"""
        # Define clinical features to extract
        clinical_features_columns = [
            ("Demographics", "Date of Birth (Days)", "(Taking date of diagnosis as day 0)"),
            ("Demographics", "Menopause (at diagnosis)", "{0 = pre,\n1 = post,\n2 = N/A}"),
            (
                "Demographics",
                "Race and Ethnicity",
                "{0 = N/A\n1 = white,\n2 = black,\n3 = asian,\n4 = native,\n5 = hispanic,\n6 = multi,\n7 = hawa,    8 = amer indian}",
            ),
            ("Tumor Characteristics", "Nottingham grade", "1=low 2=intermediate 3=high\n"),
            ("Recurrence", "Recurrence event(s)", "{0 = no, 1 = yes}"),
        ]

        # Test with clinical data and specified features
        dataset = BreastMRIDataset(
            root_dir=self.test_dir,
            clinical_data_path=self.clinical_data_path,
            clinical_features_columns=clinical_features_columns,
            patient_indices=[1],  # Test with first patient only
        )

        # Check first patient's data
        sample = dataset[0]
        self.assertIn("molecular_subtype", sample)
        self.assertIn("clinical_features", sample)

        # Verify molecular subtype mapping
        self.assertEqual(sample["molecular_subtype"], "luminal-like")

        # Verify clinical features
        expected_features = [
            "Demographics_Date of Birth (Days)",
            "Demographics_Menopause (at diagnosis)",
            "Demographics_Race and Ethnicity",
            "Tumor Characteristics_Nottingham grade",
            "Recurrence_Recurrence event(s)",
        ]
        for feature in expected_features:
            self.assertIn(feature, sample["clinical_features"])

        # Test with no clinical features specified
        dataset_no_features = BreastMRIDataset(
            root_dir=self.test_dir,
            clinical_data_path=self.clinical_data_path,
            patient_indices=[1],  # Test with first patient only
        )
        sample_no_features = dataset_no_features[0]
        self.assertEqual(sample_no_features["clinical_features"], {})

        # Test with invalid clinical feature
        invalid_features = [("Invalid", "Feature", "Description")]
        dataset_invalid = BreastMRIDataset(
            root_dir=self.test_dir,
            clinical_data_path=self.clinical_data_path,
            clinical_features_columns=invalid_features,
            patient_indices=[1],  # Test with first patient only
        )
        sample_invalid = dataset_invalid[0]
        self.assertIn("Invalid_Feature", sample_invalid["clinical_features"])
        self.assertIsNone(sample_invalid["clinical_features"]["Invalid_Feature"])

    def test_dynamic_sequence_requirements(self):
        """Test dynamic sequence validation"""
        # Test with valid directory structure
        dataset = BreastMRIDataset(root_dir=self.test_dir)
        self.assertEqual(len(dataset), 3)  # Should only include valid patients


class TestBreastMRIDatasetLoading(unittest.TestCase):
    """Test the dataset loading functionality"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.clinical_data_path = os.path.join(cls.test_dir, "clinical_data.xlsx")
        TestDicomDataCreation.create_test_hierarchy(cls.test_dir)
        cls.clinical_data, cls.column_mapping = TestDicomDataCreation.create_test_clinical_data(cls.clinical_data_path)
        cls.dataset = BreastMRIDataset(
            root_dir=cls.test_dir,
            clinical_data_path=cls.clinical_data_path,
            patient_indices=[1],  # Test with first patient only
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_data_loading(self):
        """Test basic data loading"""
        sample = self.dataset[0]

        # Check dictionary structure
        self.assertIsInstance(sample, dict)
        self.assertIn("images", sample)
        self.assertIn("patient_id", sample)
        self.assertIn("molecular_subtype", sample)
        self.assertIn("clinical_features", sample)

        # Check image dimensions
        images = sample["images"]
        self.assertEqual(len(images.shape), 4)  # [5, D, H, W]
        self.assertEqual(images.shape[0], 5)  # 5 time points
        self.assertEqual(images.shape[1], 3)  # 3 slices
        self.assertEqual(images.shape[2:], (10, 10))  # 10x10 images

    def test_data_types(self):
        """Test data types of loaded data"""
        sample = self.dataset[0]
        images = sample["images"]

        self.assertIsInstance(images, torch.Tensor)
        self.assertEqual(images.dtype, torch.float32)
        self.assertIsInstance(sample["patient_id"], str)
        self.assertIsInstance(sample["molecular_subtype"], str)
        self.assertIsInstance(sample["clinical_features"], dict)

    def test_molecular_subtype_mapping(self):
        """Test molecular subtype mapping for all patients"""
        dataset = BreastMRIDataset(
            root_dir=self.test_dir,
            clinical_data_path=self.clinical_data_path,
            patient_indices=[1, 2, 3],  # Test all patients in order
        )
        expected_subtypes = ["luminal-like", "ER/PR pos, HER2 pos", "her2"]

        for i, expected_subtype in enumerate(expected_subtypes):
            sample = dataset[i]
            self.assertEqual(sample["molecular_subtype"], expected_subtype)


class TestBreastMRIDatasetWithTransforms(unittest.TestCase):
    """Test the dataset with different transforms"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        TestDicomDataCreation.create_test_hierarchy(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_default_transform(self):
        """Test dataset with default normalization"""
        transform = MRITransformPipeline([Normalize(range_min=-1, range_max=1), ToTensor()])
        dataset = BreastMRIDataset(root_dir=self.test_dir, transform=transform)
        sample = dataset[0]

        # Check normalization range
        self.assertTrue(torch.all(sample["images"] >= -1))
        self.assertTrue(torch.all(sample["images"] <= 1))

    def test_custom_transform(self):
        """Test dataset with custom normalization range"""
        transform = MRITransformPipeline([Normalize(range_min=0, range_max=1), ToTensor()])
        dataset = BreastMRIDataset(root_dir=self.test_dir, transform=transform)
        sample = dataset[0]

        # Check normalization range
        self.assertTrue(torch.all(sample["images"] >= 0))
        self.assertTrue(torch.all(sample["images"] <= 1))


if __name__ == "__main__":
    unittest.main()
