# Created by: Hojune Kim
# Date: July 5, 2023
# Purpose: Unpack .gz files and run create_sinogram.py on each .nii file

# Usage: python preprocess.py <source_dir> <target_dir>
# Example: python preprocess.py /home/hojune/download/covid /home/hojune/real_data/raw
# Example: python $SCRATCH/CT_NVAE/computed_tomography/preprocess_real_data.py $SCRATCH/CT-Covid-19 $SCRATCH/CT-Covid-19-processed -v

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


    # Get the list of .nii files in the destination directory
    nii_files = [filename for filename in os.listdir(destination_directory) if filename.endswith('.nii')]

    # Iterate over files in the destination directory with progress
    for i, filename in enumerate(nii_files):
        # Create the full path to the .nii file
        nib_file_path = os.path.join(destination_directory, filename)

        # Call create_sinogram function with the specified parameters
        data, proj, file_name = create_sinogram_nib(nib_file_path, destination_directory, theta)
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