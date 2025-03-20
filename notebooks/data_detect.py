import os
import warnings
from collections import Counter

warnings.filterwarnings("ignore")


def detect_dataset_structure(root_dir):
    """
    Explore the dataset structure and return summary statistics.
    """
    print(f"Analyzing dataset structure in: {root_dir}")

    # Dictionary to store structure information
    dataset_info = {"total_patients": 0, "total_studies": 0, "total_sequences": 0, "total_dcm_files": 0, "patient_ids": [], "patient_details": {}}

    # Check if root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: The directory {root_dir} does not exist.")
        return dataset_info

    # Loop through first level directories (Breast_MRI_XXX)
    for patient_folder in sorted(os.listdir(root_dir)):
        patient_path = os.path.join(root_dir, patient_folder)

        if not os.path.isdir(patient_path) or not patient_folder.startswith("Breast_MRI_"):
            continue

        dataset_info["total_patients"] += 1
        dataset_info["patient_ids"].append(patient_folder)

        patient_info = {"patient_directory": None, "sequences": [], "sequence_counts": {}, "total_files": 0}

        # Find patient_directory within Breast_MRI_XXX
        patient_subdirs = [d for d in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, d))]

        if len(patient_subdirs) > 0:
            patient_directory = patient_subdirs[0]  # Assuming there's only one directory per patient
            patient_info["patient_directory"] = patient_directory
            dataset_info["total_studies"] += 1

            patient_dir_path = os.path.join(patient_path, patient_directory)

            # Loop through sequences
            for sequence_folder in sorted(os.listdir(patient_dir_path)):
                sequence_path = os.path.join(patient_dir_path, sequence_folder)

                if os.path.isdir(sequence_path):
                    dataset_info["total_sequences"] += 1
                    if ("dyn" in sequence_folder or "Vibrant" in sequence_folder) and ("Ph1" in sequence_folder or "1st" in sequence_folder):
                        sequence_name = "Ph1"
                    elif ("dyn" in sequence_folder or "Vibrant" in sequence_folder) and ("Ph2" in sequence_folder or "2nd" in sequence_folder):
                        sequence_name = "Ph2"
                    elif ("dyn" in sequence_folder or "Vibrant" in sequence_folder) and ("Ph3" in sequence_folder or "3rd" in sequence_folder):
                        sequence_name = "Ph3"
                    elif ("dyn" in sequence_folder or "Vibrant" in sequence_folder) and ("Ph4" in sequence_folder or "4th" in sequence_folder):
                        sequence_name = "Ph4"
                    elif "t1" in sequence_folder or "T1" in sequence_folder:
                        sequence_name = "t1"
                    elif ("dyn" in sequence_folder or "Vibrant" in sequence_folder) and "pre" in sequence_folder:
                        sequence_name = "pre"
                    else:
                        sequence_name = sequence_folder

                    patient_info["sequences"].append(sequence_name)

                    # Count DICOM files
                    dcm_files = [f for f in os.listdir(sequence_path) if f.endswith(".dcm")]
                    file_count = len(dcm_files)
                    patient_info["sequence_counts"][sequence_folder] = file_count
                    patient_info["total_files"] += file_count
                    dataset_info["total_dcm_files"] += file_count

        dataset_info["patient_details"][patient_folder] = patient_info

    return dataset_info


def display_dataset_summary(dataset_info):
    """
    Display summary statistics of the dataset.
    """
    print("\n===== DATASET SUMMARY =====")
    print(f"Total patients (Breast_MRI_XXX folders): {dataset_info['total_patients']}")
    print(f"Total studies (patient directories): {dataset_info['total_studies']}")
    print(f"Total sequences: {dataset_info['total_sequences']}")
    print(f"Total DICOM files: {dataset_info['total_dcm_files']}")

    if dataset_info["total_patients"] > 0:
        avg_sequences = dataset_info["total_sequences"] / dataset_info["total_patients"]
        avg_files = dataset_info["total_dcm_files"] / dataset_info["total_patients"]
        print(f"Average sequences per patient: {avg_sequences:.2f}")
        print(f"Average DICOM files per patient: {avg_files:.2f}")

        # Collect sequence names across patients
        all_sequences = []
        for patient_id, details in dataset_info["patient_details"].items():
            all_sequences.extend(details["sequences"])
            if "Ph1" not in details["sequences"] or "Ph2" not in details["sequences"] or "Ph3" not in details["sequences"]:
                print(details["sequences"], " ", patient_id)

        sequence_counts = Counter(all_sequences)
        print("\nMost common sequence names:")
        for seq, count in sequence_counts.most_common(10):
            print(f"  - {seq}: {count} occurrences")

        # Sample of patients
        print("\nSample of patient IDs:")
        for patient_id in sorted(dataset_info["patient_ids"])[:5]:
            print(f"  - {patient_id}")
        if len(dataset_info["patient_ids"]) > 5:
            print(f"  - ... and {len(dataset_info['patient_ids']) - 5} more")


# Set the root directory path
root_dir = "/home/hice1/ygu367/ECE6780/project_group5/data/Duke-Breast-Cancer-MRI"  # Change this to your actual path

# Detect dataset structure
dataset_info = detect_dataset_structure(root_dir)

# Display summary
display_dataset_summary(dataset_info)
