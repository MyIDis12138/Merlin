import os
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

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

        patient_info = {
            "patient_directory": None,
            "sequences": [],
            "sequence_counts": {},
            "total_files": 0,
            "phase_sequence_files": {"Ph1": 0, "Ph2": 0, "Ph3": 0, "Ph4": 0},  # Track files in each phase
        }

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

                    # Determine sequence name
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

                    # Update phase sequence file counts if applicable
                    if sequence_name in ["Ph1", "Ph2", "Ph3", "Ph4"]:
                        patient_info["phase_sequence_files"][sequence_name] = file_count

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


def analyze_phase_sequences(dataset_info):
    """
    Analyze the distribution of images in Ph1, Ph2, and Ph3 sequences.
    """
    print("\n===== PHASE SEQUENCE ANALYSIS =====")

    # Extract phase sequence file counts for all patients
    ph1_counts = []
    ph2_counts = []
    ph3_counts = []
    patients_with_phases = []
    patient_phase_counts = {}

    for patient_id, details in dataset_info["patient_details"].items():
        ph1 = details["phase_sequence_files"]["Ph1"]
        ph2 = details["phase_sequence_files"]["Ph2"]
        ph3 = details["phase_sequence_files"]["Ph3"]

        # Only include patients that have all three phases
        if ph1 > 0 and ph2 > 0 and ph3 > 0:
            ph1_counts.append(ph1)
            ph2_counts.append(ph2)
            ph3_counts.append(ph3)
            patients_with_phases.append(patient_id)
            patient_phase_counts[patient_id] = {"Ph1": ph1, "Ph2": ph2, "Ph3": ph3}

    # Display statistics
    total_patients_with_phases = len(patients_with_phases)
    print(
        f"Patients with all three phases (Ph1, Ph2, Ph3): {total_patients_with_phases} out \
          of {dataset_info['total_patients']} ({total_patients_with_phases/dataset_info['total_patients']*100:.2f}%)"
    )

    if total_patients_with_phases > 0:
        # Calculate statistics
        print("\nStatistics for number of images in each phase:")
        for phase, counts in [("Ph1", ph1_counts), ("Ph2", ph2_counts), ("Ph3", ph3_counts)]:
            print(f"  {phase}:")
            print(f"    - Min: {min(counts)}")
            print(f"    - Max: {max(counts)}")
            print(f"    - Mean: {np.mean(counts):.2f}")
            print(f"    - Median: {np.median(counts):.2f}")
            print(f"    - Std Dev: {np.std(counts):.2f}")

        # Check consistency within patients
        print("\nConsistency check within patients:")
        consistent_patients = sum(
            1
            for p in patients_with_phases
            if len(set([patient_phase_counts[p]["Ph1"], patient_phase_counts[p]["Ph2"], patient_phase_counts[p]["Ph3"]])) == 1
        )
        print(
            f"Patients with same number of images across all phases: {consistent_patients} ({consistent_patients/total_patients_with_phases*100:.2f}%)"
        )

        # Show histogram of image counts for each phase
        plot_phase_distributions(ph1_counts, ph2_counts, ph3_counts)

        # Sample of patients with their phase counts
        print("\nSample of patients with phase images:")
        sample_count = min(5, len(patients_with_phases))
        for patient_id in patients_with_phases[:sample_count]:
            phases = patient_phase_counts[patient_id]
            print(f"  - {patient_id}: Ph1={phases['Ph1']}, Ph2={phases['Ph2']}, Ph3={phases['Ph3']}")
        if len(patients_with_phases) > sample_count:
            print(f"  - ... and {len(patients_with_phases) - sample_count} more")
    else:
        print("No patients found with all three phases.")


def plot_phase_distributions(ph1_counts, ph2_counts, ph3_counts):
    """
    Create histograms and boxplots to visualize the distribution of image counts.
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Histogram
    ax1.hist([ph1_counts, ph2_counts, ph3_counts], bins=20, alpha=0.7, label=["Ph1", "Ph2", "Ph3"])
    ax1.set_xlabel("Number of Images")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Image Counts for Phase Sequences")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Boxplot
    ax2.boxplot([ph1_counts, ph2_counts, ph3_counts], labels=["Ph1", "Ph2", "Ph3"], patch_artist=True)
    ax2.set_xlabel("Phase Sequence")
    ax2.set_ylabel("Number of Images")
    ax2.set_title("Boxplot of Image Counts for Phase Sequences")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("phase_sequence_distribution.png")
    print("\nDistribution plot saved as 'phase_sequence_distribution.png'")


# Set the root directory path
root_dir = "/home/hice1/ygu367/ECE6780/project_group5/data/Duke-Breast-Cancer-MRI"  # Change this to your actual path

# Detect dataset structure
dataset_info = detect_dataset_structure(root_dir)

# Display summary
display_dataset_summary(dataset_info)

# Analyze phase sequences
analyze_phase_sequences(dataset_info)
