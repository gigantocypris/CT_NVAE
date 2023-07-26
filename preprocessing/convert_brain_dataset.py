# Purpose: Converting unorganized DICOM files to organized 3D numpy arrays
# Usage: python preprocess_brain_data.py <source_dir> <target_dir>
# Example: python $CT_NVAE_PATH/preprocessing/convert_brain_dataset.py $SOURCE_DIR $TARGET_DIR 

from helper import dicom_to_npy, organize_dcm_files_by_Instance
import os
import argparse

def preprocess(source_directory, destination_directory):

    # Set the path
    dcm_path = source_directory
    dcm_organized_path = os.path.join(destination_directory, "dcm_organized_by_instance")
    npy_organized_path = os.path.join(destination_directory, "npy_organized_by_instance")

    # Make directories if they don't exist
    os.makedirs(dcm_organized_path, exist_ok=True)
    os.makedirs(npy_organized_path, exist_ok=True)

    # Start preprocessing 
    # Organize DICOM files into instance folders
    print("Organizing DICOM files by instance")
    organize_dcm_files_by_Instance(dcm_path, dcm_organized_path)

    # Convert organized DICOM files to 3D numpy arrays
    dicom_to_npy(dcm_organized_path, npy_organized_path)
    print("All files converted to ",npy_organized_path)



if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Preprocess unorganized DICOM files into organized 3D numpy arrays")
    parser.add_argument("source_directory", help="Directory containing .DICOM files")
    parser.add_argument("destination_directory", help="Directory to store the organized DICOM files and 3D .npy files")
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Run preprocess function
    preprocess(args.source_directory, args.destination_directory)