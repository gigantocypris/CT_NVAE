# Created: 07.19.2023 16:00
# Purpose: Converting unorganized DICOM files to organized 3D numpy arrays

import argparse
import os
from utils import count_files_in_directory, count_unique_patient_ids, make_smaller_dataset, organize_dcm_files_by_patient_id, dicom_to_npy

def main(source_path, destination_path, smaller):
    # Set the path
    dcm_path = source_path
    dcm_organized_path = os.path.join(destination_path, "dcm_organized_by_patient")
    npy_organized_path = os.path.join(destination_path, "npy_organized_by_patient")
    smaller_organized_dcm_path = os.path.join(destination_path, "smaller_dcm_organized_by_patient")
    smaller_organized_npy_path = os.path.join(destination_path, "smaller_npy_organized_by_patient")

    # Make directories if they don't exist
    os.makedirs(dcm_organized_path, exist_ok=True)
    os.makedirs(npy_organized_path, exist_ok=True)
    os.makedirs(smaller_organized_dcm_path, exist_ok=True)
    os.makedirs(smaller_organized_npy_path, exist_ok=True)

#     # Count the number of files in a DICOM directory
#     num_files = count_files_in_directory(dcm_path)
#     print('Number of files:', num_files)

#     # Count how many patients there are
#     num_unique_patient_ids = count_unique_patient_ids(dcm_path)
#     print('Number of unique PatientIDs:', num_unique_patient_ids)

    # Organize DICOM files into patient folders
    print("Organizing DICOM files by patient)
    organize_dcm_files_by_patient_id(dcm_path, dcm_organized_path)

    # Make a dataset with 3D numpy arrays (can be smaller)
    if smaller:
        make_smaller_dataset(dcm_organized_path, smaller_organized_dcm_path, num_patient=100)
        # Convert organized DICOM files to 3D numpy arrays
        dicom_to_npy(smaller_organized_dcm_path, smaller_organized_npy_path)
        print("Converted to ",smaller_organized_npy_path)

    else:
        # Convert organized DICOM files to 3D numpy arrays
        dicom_to_npy(dcm_organized_path, npy_organized_path)
        print("Converted to ",npy_organized_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process brain DICOM files.')
    parser.add_argument('--source_path', type=str, required=True,
                        help='path to the source DICOM files')
    parser.add_argument('--destination_path', type=str, required=True,
                        help='path to the destination directory for the organized files and numpy arrays')
    parser.add_argument('--smaller', type=bool, default=False,
                        help='whether to create a smaller dataset')
    args = parser.parse_args()
    main(args.source_path, args.destination_path, args.smaller)