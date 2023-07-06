import os
import numpy as np
import gzip
import shutil
from create_sinogram import create_sinogram
import numpy as np
import argparse

# Created by: Hojune Kim
# Date: July 5, 2023
# Purpose: Create sinogram from .nib file and save it as .npy file

# Usage: python create_sinogram.py <nib_file_path> <only_sinogram> <pad>
# Example: python data/create_sinogram.py data/raw/Covid_CT_1.nii False True
# If only_sinogram=True, then it will only create sinogram and save it as .npy file
# If only_sinogram=False, then it will save plot and .npy file for both sinogram and its original image

# Make sure to create data folder and figures folder within this directory before running this code
# Run this on the root directory of this project (Desktop for this case)

def preprocess(source_directory, destination_directory):
    # Create sinogram_npy directory if it doesn't exist
    sinogram_npy_directory = os.path.join(os.path.dirname(os.path.dirname(destination_directory)), "sinogram_npy")
    os.makedirs(sinogram_npy_directory, exist_ok=True)

    # Create input_npy directory if it doesn't exist
    input_npy_directory = os.path.join(os.path.dirname(os.path.dirname(destination_directory)), "input_npy")
    os.makedirs(input_npy_directory, exist_ok=True)

    # Create figures directory if it doesn't exist
    figures_directory = os.path.join(os.path.dirname(os.path.dirname(destination_directory)), "figures")
    os.makedirs(figures_directory, exist_ok=True)

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
