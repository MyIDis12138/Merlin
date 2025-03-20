import concurrent.futures
import logging
import os
from collections import defaultdict

import pydicom

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_dicom_file(file_path):
    """
    Process a single DICOM file and return its shape if it has pixel data.

    Args:
        file_path (str): Path to the DICOM file

    Returns:
        tuple or None: Shape of the pixel array if available, None otherwise
    """
    try:
        ds = pydicom.dcmread(file_path)
        if "PixelData" in ds:
            return ds.pixel_array.shape
        return None
    except Exception as e:
        logger.error(f"Error reading file: {file_path}, Error: {e}")
        return None


def process_patient_directory(patient_dir_path):
    """
    Process all DICOM files in a patient directory.

    Args:
        patient_dir_path (str): Path to the patient directory

    Returns:
        tuple: (set of unique shapes, dict mapping sequence names to their shapes)
    """
    unique_shapes = set()
    has_pixel_data = False
    sequence_shapes = {}  # Will store shapes for each sequence

    for dynamic_sequence_dir in os.listdir(patient_dir_path):
        dynamic_sequence_path = os.path.join(patient_dir_path, dynamic_sequence_dir)
        if os.path.isdir(dynamic_sequence_path):
            dicom_files = [os.path.join(dynamic_sequence_path, f) for f in os.listdir(dynamic_sequence_path) if f.endswith(".dcm")]

            # Process DICOM files in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                shapes = list(executor.map(process_dicom_file, dicom_files))

            # Filter out None values and add shapes to the set
            valid_shapes = [s for s in shapes if s is not None]
            if valid_shapes:
                has_pixel_data = True
                unique_shapes.update(valid_shapes)

                # Find the most common shape for this sequence
                shape_counts = defaultdict(int)
                for shape in valid_shapes:
                    shape_counts[shape] += 1

                # Get the most common shape for this sequence
                if shape_counts:
                    most_common_shape = max(shape_counts.items(), key=lambda x: x[1])[0]
                    sequence_shapes[dynamic_sequence_dir] = most_common_shape

    return (unique_shapes if has_pixel_data else set(), sequence_shapes)


def analyze_patient_image_shapes(root_dir, max_workers=None):
    """
    Analyzes the image shapes of DICOM files at the patient level using parallel processing.

    Args:
        root_dir (str): The path to the root directory containing the Breast_MRI folders.
        max_workers (int, optional): Maximum number of worker processes/threads.

    Returns:
        tuple: (
            dict: A dictionary where keys are patient IDs and values are the common image shape,
                  or None if inconsistent or no images with PixelData,
            dict: A dictionary with detailed sequence shape information for inconsistent patients
        )
    """
    patient_data = []

    # Collect all patient directories
    for breast_mri_dir in os.listdir(root_dir):
        breast_mri_path = os.path.join(root_dir, breast_mri_dir)
        if os.path.isdir(breast_mri_path) and breast_mri_dir.startswith("Breast_MRI_"):
            patient_dirs = [d for d in os.listdir(breast_mri_path) if os.path.isdir(os.path.join(breast_mri_path, d))]

            if patient_dirs:
                patient_dir_path = os.path.join(breast_mri_path, patient_dirs[0])
                patient_id = breast_mri_dir
                patient_data.append((patient_id, patient_dir_path))
            else:
                logger.warning(f"No patient directory found in {breast_mri_path}")

    # Process patients in parallel
    patient_shapes = {}
    inconsistent_patient_details = {}  # Store detailed sequence shapes for inconsistent patients

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Start the processing tasks
        future_to_patient = {executor.submit(process_patient_directory, dir_path): patient_id for patient_id, dir_path in patient_data}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_patient):
            patient_id = future_to_patient[future]
            try:
                unique_shapes, sequence_shapes = future.result()
                if unique_shapes:
                    if len(unique_shapes) == 1:
                        patient_shapes[patient_id] = unique_shapes.pop()
                    else:
                        patient_shapes[patient_id] = None  # Inconsistent shapes
                        # Store detailed sequence shape information for this inconsistent patient
                        inconsistent_patient_details[patient_id] = sequence_shapes
                else:
                    patient_shapes[patient_id] = None  # No PixelData found
            except Exception as e:
                logger.error(f"Error processing patient {patient_id}: {e}")
                patient_shapes[patient_id] = None

    return patient_shapes, inconsistent_patient_details


def analyze_patient_shape_distribution(patient_shapes):
    """
    Analyzes the distribution of image shapes across patients.

    Args:
        patient_shapes (dict): A dictionary where keys are patient IDs and values
                              are their common image shapes (or None).

    Returns:
        dict: A dictionary where keys are image shapes and values are the number of patients with that shape.
    """
    shape_distribution = defaultdict(int)
    for shape in patient_shapes.values():
        shape_distribution[shape] += 1
    return dict(shape_distribution)  # Convert to regular dict for better display


def main():
    import argparse
    import json
    import multiprocessing

    parser = argparse.ArgumentParser(description="Analyze DICOM image shapes in a dataset.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing the Breast_MRI folders")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: number of CPU cores)")
    parser.add_argument("--output", type=str, default=None, help="Output file to save the detailed results (JSON format)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed sequence shape information for inconsistent patients")

    args = parser.parse_args()

    # If workers not specified, use CPU count
    if args.workers is None:
        args.workers = multiprocessing.cpu_count()

    logger.info(f"Starting analysis with {args.workers} workers")

    # Analyze patient shapes
    patient_image_shapes, inconsistent_patient_details = analyze_patient_image_shapes(args.root_dir, args.workers)
    patient_shape_distribution = analyze_patient_shape_distribution(patient_image_shapes)

    # Print results
    logger.info("\n--- Patient Level Image Shape Analysis ---")
    for patient, shape in patient_image_shapes.items():
        if shape:
            logger.info(f"Patient: {patient}, Common Shape: {shape}")
        else:
            logger.info(f"Patient: {patient}, Inconsistent or No PixelData found.")

    logger.info("\n--- Patient Shape Distribution ---")
    if patient_shape_distribution:
        for shape, count in sorted(patient_shape_distribution.items(), key=lambda item: item[1], reverse=True):
            if shape:
                logger.info(f"Shape: {shape}, Number of Patients: {count}")
            else:
                logger.info(f"Shape: Inconsistent or No PixelData, Number of Patients: {count}")
    else:
        logger.info("No patient shape data found.")

    # Print detailed sequence shape information for inconsistent patients if requested
    if args.detailed and inconsistent_patient_details:
        logger.info("\n--- Detailed Sequence Shape Analysis for Inconsistent Patients ---")
        for patient_id, sequence_shapes in inconsistent_patient_details.items():
            logger.info(f"\nPatient: {patient_id}")
            for sequence_name, shape in sequence_shapes.items():
                logger.info(f"  Sequence: {sequence_name}, Shape: {shape}")

    # Save detailed results to a JSON file if requested
    if args.output:
        logger.info(f"Saving detailed results to {args.output}")
        output_data = {
            "patient_shapes": {k: str(v) if v else None for k, v in patient_image_shapes.items()},
            "shape_distribution": {str(k) if k else "None": v for k, v in patient_shape_distribution.items()},
            "inconsistent_patients": {
                patient_id: {seq_name: str(shape) for seq_name, shape in seq_shapes.items()}
                for patient_id, seq_shapes in inconsistent_patient_details.items()
            },
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()
