import pandas as pd
import pydicom
import numpy as np
import os
import argparse
from tqdm import tqdm


def create_dataset(csv_path, dicom_directory, output_path, thickness_path, num_instance):
    # Read the thickness CSV file
    df_thickness = pd.read_csv(thickness_path)

    # Select num_instance instances randomly
    selected_instances = df_thickness.sample(num_instance)['StudyInstanceUID']

    # Read the main CSV file
    df_main = pd.read_csv(csv_path)

    # Filter rows that have selected instances
    df_selected = df_main[df_main['StudyInstanceUID'].isin(selected_instances)]

    # Save the selected instances to a new CSV file
    df_selected.to_csv(os.path.join(output_path, f'{num_instance}_instances_selected.csv'), index=False)

    # Process the selected DICOM files
    process_dicom_files(os.path.join(output_path, f'{num_instance}_instances_selected.csv'), dicom_directory, output_path)



def process_dicom_files(csv_path, dicom_directory, output_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Initialize an empty list to store pixel arrays
    pixel_arrays = []

    # Initialize the current StudyInstanceUID
    current_study = None

    # Iterate over each row in the DataFrame
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing DICOM files"):
        # Read the DICOM file
        dicom_file = os.path.join(dicom_directory, row['File Name'])
        dataset = pydicom.dcmread(dicom_file)

        # If the StudyInstanceUID has changed, save the current pixel arrays to a file
        if current_study is not None and row['StudyInstanceUID'] != current_study:
            try:
                # Convert the list of pixel arrays to a 3D numpy array
                volume = np.stack(pixel_arrays, axis=2)

                # Normalize the volume
                volume = volume.astype(np.float32)
                volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

                # Save the normalized volume to a .npy file
                np.save(os.path.join(output_path, f"{current_study}.npy"), volume)
            except ValueError as ve:
                print(f"Error processing StudyInstanceUID {current_study}: {ve}")

            # Clear the pixel arrays for the next volume
            pixel_arrays = []

        # Add the pixel array from the current DICOM file to the list
        pixel_arrays.append(dataset.pixel_array)

        # Update the current StudyInstanceUID
        current_study = row['StudyInstanceUID']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DICOM files.')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('dicom_directory', type=str, help='Path to the directory containing DICOM files')
    parser.add_argument('output_path', type=str, help='Path to save the output .npy files')
    parser.add_argument('thickness_path', type=str, help='Path to the thickness CSV file')
    parser.add_argument('num_instance', type=int, help='Number of instances to select')

    args = parser.parse_args()

    create_dataset(args.csv_path, args.dicom_directory, args.output_path, args.thickness_path, args.num_instance)

