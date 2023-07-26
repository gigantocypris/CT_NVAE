import os
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import argparse

def generate_dicom_metadata_array(directory_path, csv_path):
    # Ignore specific pydicom warnings
    warnings.filterwarnings("ignore", "Invalid value for VR UI")

    # Get list of all files in the directory
    files = os.listdir(directory_path)

    # Filter out non-DICOM files
    dicom_files = [f for f in files if f.endswith('.dcm')]

    if not dicom_files:
        print("No DICOM files found in the directory.")
        return

    # Prepare a list to store metadata
    metadata = []

    # Iterate over each file in the directory
    for dicom_file in tqdm(dicom_files, desc="Processing DICOM files"):
        # Read the DICOM file
        dataset = pydicom.dcmread(os.path.join(directory_path, dicom_file))

        # Extract the specified metadata and add it to the list
        metadata.append([dicom_file, dataset.StudyInstanceUID, dataset.PatientID, str(dataset.ImagePositionPatient)])

    # Convert to DataFrame and save to .csv file
    df = pd.DataFrame(metadata, columns=['File Name', 'StudyInstanceUID', 'PatientID', 'ImagePositionPatient'])
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a CSV of metadata from DICOM files.')
    parser.add_argument('directory_path', type=str, help='Path to the directory containing DICOM files')
    parser.add_argument('csv_path', type=str, help='Path to save the output CSV file')

    args = parser.parse_args()

    generate_dicom_metadata_array(args.directory_path, args.csv_path)
