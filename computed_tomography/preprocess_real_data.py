# Created by: Hojune Kim
# Date: July 5, 2023
# Purpose: Unpack .gz files and run create_sinogram.py on each .nii file
# Last update: July 11, 2023 by Gary Chen
# Updates: fixed typos in source and destination directory around lines 50-65 and also the terminal command examples

# Usage: python preprocess.py <source_dir> <target_dir>
# Example: python preprocess.py /global/cfs/cdirs/m3562/users/hkim/real_data/raw /global/cfs/cdirs/m3562/users/hkim/real_data/pre_processed
# Example: python $SCRATCH/CT_NVAE/computed_tomography/preprocess_real_data.py $SOURCE_DIR $TARGET_DIR -v

import os
import gzip
import shutil
from utils_real_data import create_sinogram_nib, visualize
import numpy as np
import argparse

def preprocess(source_directory, destination_directory, visualize_output):
    theta = np.linspace(0, np.pi, 180, endpoint=False)

    # Create the desination_directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    # Save the projection angles
    theta_file_path = f"{destination_directory}/theta.npy"
    np.save(theta_file_path, theta)

    # Get the list of .gz files in the source directory
    gz_files = [filename for filename in os.listdir(source_directory) if filename.endswith('.gz')]

    # Iterate over files in the source directory with progress
    for i, filename in enumerate(gz_files):
        # Create the paths for the source and destination files
        source_path = os.path.join(source_directory, filename)
        destination_path = os.path.join(destination_directory, os.path.splitext(filename)[0])

        # Open the .gz file and extract its contents
        with gzip.open(source_path, 'rb') as gz_file:
            with open(destination_path, 'wb') as extracted_file:
                shutil.copyfileobj(gz_file, extracted_file)

        # Show progress
        progress = (i + 1) / len(gz_files) * 100
        print("Unzipping progress: {:.2f}%".format(progress))
    print("All files unzipped. Now they are .nii files.")


    # Run create_sinogram.py on each .nii file
    print("Now running create_sinogram.py on each .nii file...")


    # Get the list of .nii files in the source_directory directory
    nii_files = [filename for filename in os.listdir(source_directory) if filename.endswith('.nii')]
    print(f'nii_files has shape {len(nii_files)}')

    # Iterate over files in the source_directory directory with progress
    for i, filename in enumerate(nii_files):
        # Create the full path to the .nii file
        nib_file_path = os.path.join(source_directory, filename)

        # Call create_sinogram function with the specified parameters
        data, proj, file_name = create_sinogram_nib(nib_file_path, destination_directory, theta)
        print(f'creating sinograms {i} in nii files')
        if visualize_output:
            visualize(data, proj, file_name, destination_directory)

        # Show progress
        progress = (i + 1) / len(nii_files) * 100
        print("Processing progress: {:.2f}%".format(progress))

    print("All files preprocessed.")



if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Preprocess .gz files and run create_sinogram.py on each .nii file")
    parser.add_argument("source_directory", help="Directory containing .gz files")
    parser.add_argument("destination_directory", help="Directory to store the unzipped .nii files")
    parser.add_argument('-v', action='store_true', dest='visualize_output',
                        help='visualize images and sinograms')
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Run preprocess function
    preprocess(args.source_directory, args.destination_directory, args.visualize_output)