# Purpose: Creating cache that contains important metadata from DICOM files 
# Usage: python preprocessing/brain_create_CSV.py <source_dir> <temp_csv_dir> --start_file <start_file>
# Example: python $CT_NVAE_PATH/preprocessing/brain_create_CSV.py $SOURCE_DIR $TEMP_CSV_DIR  

import os
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import argparse


def generate_dicom_metadata_array(directory_path, base_csv_path, start_file=None):
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

    # Flag to indicate whether we should start processing
    start_processing = False if start_file else True

    # Counter for the number of files processed
    processed_files = 0

    # Batch number
    batch_num = 68 if start_file else 1 # Change batch_num accordingly if you have a start_file

    # Iterate over each file in the directory
    for i, dicom_file in enumerate(tqdm(dicom_files, desc="Processing DICOM files")):
        # If this is the start file, we start processing from now
        if dicom_file == start_file:
            start_processing = True

        if start_processing:
            # Read the DICOM file
            dataset = pydicom.dcmread(os.path.join(directory_path, dicom_file))

            # Extract the specified metadata and add it to the list
            metadata.append([dicom_file, dataset.StudyInstanceUID, dataset.PatientID, str(dataset.ImagePositionPatient)])

            processed_files += 1

            # Save to a new CSV every 10,000 DICOM files
            if processed_files % 10000 == 0:
                df = pd.DataFrame(metadata, columns=['File Name', 'StudyInstanceUID', 'PatientID', 'ImagePositionPatient'])
                csv_path = f"{base_csv_path}_part_{str(batch_num).zfill(3)}.csv"
                df.to_csv(csv_path, index=False)

                # Clear the metadata list for the next batch
                metadata = []
                batch_num += 1

    # Save remaining DICOM files
    if metadata:
        df = pd.DataFrame(metadata, columns=['File Name', 'StudyInstanceUID', 'PatientID', 'ImagePositionPatient'])
        csv_path = f"{base_csv_path}_part_{str(batch_num).zfill(3)}.csv"
        df.to_csv(csv_path, index=False)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a CSV of metadata from DICOM files.')
    parser.add_argument('source_dir', type=str, help='Path to the directory containing DICOM files')
    parser.add_argument('temp_csv_dir', type=str, help='Base path to save the output CSV files')
    parser.add_argument('--start_file', type=str, default=None, help='File to start processing from. This should end with .dcm. Also, change batch_num accordingly')

    args = parser.parse_args()

    generate_dicom_metadata_array(args.source_dir, args.temp_csv_dir, args.start_file)
