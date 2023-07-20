# Created by: Hojune Kim
# Date: July 19, 2023
# Purpose: Converting unorganized DICOM files to organized 3D numpy arrays and run create_sinogram.py on each .npy file

# Usage: python preprocess_brain_data.py <source_dir> <target_dir> -n <num_truncate> [-v]
# Example: python $CT_NVAE_PATH/computed_tomography/preprocess_brain_data.py $SOURCE_DIR $TARGET_DIR -small True -n 100 

from utils_real_data import create_sinogram_npy, visualize, make_small_dataset, dicom_to_npy
from utils_real_data import  organize_dcm_files_by_patient_id, count_files_in_directory, count_unique_patient_ids 

import os
import numpy as np
import argparse
from tqdm import tqdm

def preprocess(source_directory, destination_directory, small, num_truncate, visualize_output):
    theta = np.linspace(0, np.pi, 180, endpoint=False)

    # Set the path
    dcm_path = source_directory
    dcm_organized_path = os.path.join(destination_directory, "dcm_organized_by_patient")
    npy_organized_path = os.path.join(destination_directory, "npy_organized_by_patient")
    small_organized_dcm_path = os.path.join(destination_directory, "small_dcm_organized_by_patient")
    small_organized_npy_path = os.path.join(destination_directory, "small_npy_organized_by_patient")
    small_sinogram_path = os.path.join(destination_directory, "small_sinogram")
    sinogram_path = os.path.join(destination_directory, "sinogram")

    # Make directories if they don't exist
    os.makedirs(dcm_organized_path, exist_ok=True)
    os.makedirs(npy_organized_path, exist_ok=True)
    os.makedirs(small_organized_dcm_path, exist_ok=True)
    os.makedirs(small_organized_npy_path, exist_ok=True)
    os.makedirs(small_sinogram_path, exist_ok=True)
    os.makedirs(sinogram_path, exist_ok=True)

    # Save the projection angles
    theta_file_path = f"{destination_directory}/theta.npy"
    np.save(theta_file_path, theta)

    # # Start preprocessing 
    # # Count the number of files in a DICOM directory
    # print('Number of files:', count_files_in_directory(dcm_path))
    # # Count how many patients there are
    # print('Number of unique PatientIDs:', count_unique_patient_ids(dcm_path))

    # # Organize DICOM files into patient folders
    # print("Organizing DICOM files by patient")
    # organize_dcm_files_by_patient_id(dcm_path, dcm_organized_path)

    # Make a dataset with 3D numpy arrays
    if small:
        destination_directory = small_sinogram_path
        source_directory = small_organized_npy_path
        make_small_dataset(dcm_organized_path, small_organized_dcm_path, 100)

        # Convert organized DICOM files to 3D numpy arrays
        dicom_to_npy(small_organized_dcm_path, small_organized_npy_path)
        print("Converted to ",small_organized_npy_path)

        # Get the list of .npy files in the small_organized_npy_path directory
        npy_files = [filename for filename in os.listdir(small_organized_npy_path) if filename.endswith('.npy')]
        # In case num_truncate is larger than the number of .npy files in the small_organized_npy_path
        if num_truncate > len(npy_files):
            num_truncate = len(npy_files)

    else:
        destination_directory = sinogram_path
        source_directory = npy_organized_path
        # Convert organized DICOM files to 3D numpy arrays
        dicom_to_npy(dcm_organized_path, npy_organized_path)
        print("Converted to ",npy_organized_path)

        # Get the list of .npy files in the npy_organized_path directory
        npy_files = [filename for filename in os.listdir(npy_organized_path) if filename.endswith('.npy')]
        # In case num_truncate is larger than the number of .npy files in the npy_organized_path
        if num_truncate > len(npy_files):
            num_truncate = len(npy_files)


    # Run create_sinogram.py on each .npy file
    print("Now running create_sinogram.py on each .npy file...")

    # Iterate over files in the source_directory directory with progress
    for i, filename in enumerate(tqdm(npy_files[:num_truncate], desc="Processing files")):

        # Create the full path to the .npy file
        npy_file_path = os.path.join(source_directory, filename)

        # Call create_sinogram function with the specified parameters
        data, proj, file_name = create_sinogram_npy(npy_file_path, destination_directory, theta)

        if visualize_output:
            visualize(data, proj, file_name, destination_directory)

    print("All files preprocessed.")



if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Preprocess .gz files and run create_sinogram.py on each .nii file")
    parser.add_argument("source_directory", help="Directory containing .gz files")
    parser.add_argument("destination_directory", help="Directory to store the unzipped .nii files")
    parser.add_argument('-v', action='store_true', dest='visualize_output',
                        help='visualize images and sinograms')
    parser.add_argument('-n',dest='num_truncate', type=int, help='total number of nii files to pre-process', default=10)
    parser.add_argument('-small', type=bool, default=False,
                        help='whether to create a small dataset')
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Run preprocess function
    preprocess(args.source_directory, args.destination_directory, args.small, args.num_truncate, args.visualize_output)