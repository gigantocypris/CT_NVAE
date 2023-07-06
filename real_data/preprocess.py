# Created by: Hojune Kim
# Date: July 5, 2023
# Purpose: Unpack .gz files and run create_sinogram.py on each .nii file

# Usage: python preprocess.py <source_dir> <target_dir>
# Example: python preprocess.py /home/hojune/download/covid /home/hojune/real_data/raw

import os
import gzip
import shutil
from create_sinogram import create_sinogram
import numpy as np
import argparse

def preprocess(source_directory, destination_directory):
    # Create the desination_directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

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
        print(f"Unzipping progress: {progress:.2f}%")
    print("All files unzipped. Now they are .nii files.")


    # Run create_sinogram.py on each .nii file
    print("Now running create_sinogram.py on each .nii file...")
    theta = np.arange(0, 180, 1)
    only_sinogram = True
    pad = True

    # Get the list of .nii files in the destination directory
    nii_files = [filename for filename in os.listdir(destination_directory) if filename.endswith('.nii')]

    # Iterate over files in the destination directory with progress
    for i, filename in enumerate(nii_files):
        # Create the full path to the .nii file
        nib_file_path = os.path.join(destination_directory, filename)

        # Call create_sinogram function with the specified parameters
        create_sinogram(nib_file_path, theta, only_sinogram, pad)

        # Show progress
        progress = (i + 1) / len(nii_files) * 100
        print(f"Processing progress: {progress:.2f}%")

    print("All files preprocessed.")



if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Preprocess .gz files and run create_sinogram.py on each .nii file")
    parser.add_argument("source_directory", help="Directory containing .gz files")
    parser.add_argument("destination_directory", help="Directory to store the unzipped .nii files")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Run preprocess function
    preprocess(args.source_directory, args.destination_directory)