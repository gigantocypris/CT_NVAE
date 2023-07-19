# Created: 07.19.2023 16:00
# Author: Hojune Kim
# Purpose: Converting unorganized DICOM files to organized 3D numpy arrays


import os
import random
import numpy as np
import shutil
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm

# Input: whether to make a smaller dataset or not
def convert_dcm_to_3D_npy(smaller = True):
    # Set the path
    dcm_path = "/global/cfs/cdirs/m3562/users/hkim/brain_data/raw/stage_2_test"
    dcm_organized_path = "/global/cfs/cdirs/m3562/users/hkim/brain_data/dcm_organized_by_patient"
    npy_organized_path = "/global/cfs/cdirs/m3562/users/hkim/brain_data/npy_organized_by_patient"
    smaller_organized_dcm_path = "/global/cfs/cdirs/m3562/users/hkim/brain_data/smaller_dcm_organized_by_patient"
    smaller_organized_npy_path = "/global/cfs/cdirs/m3562/users/hkim/brain_data/smaller_npy_organized_by_patient"
    # Make directories if they don't exist
    os.makedirs(dcm_organized_path, exist_ok=True)
    os.makedirs(npy_organized_path, exist_ok=True)
    os.makedirs(smaller_organized_dcm_path, exist_ok=True)
    os.makedirs(smaller_organized_npy_path, exist_ok=True)

    # # Count the number of files in a DICOM directory
    # num_files = count_files_in_directory(dcm_path)
    # print('Number of files:', num_files)

    # # Count how many patients there are
    # num_unique_patient_ids = count_unique_patient_ids(dcm_path)
    # print('Number of unique PatientIDs:', num_unique_patient_ids)

    # Organize DICOM files into patient folders
    organize_dcm_files_by_patient_id(dcm_path, dcm_organized_path)

    # Make a dataset with 3D numpy arrays (can be smaller)
    if smaller:
        make_smaller_dataset(dcm_organized_path, smaller_organized_dcm_path, num_patient=100)
        # Convert organized DICOM files to 3D numpy arrays
        dicom_to_npy(smaller_organized_dcm_path, smaller_organized_npy_path)

    else:
        # Convert organized DICOM files to 3D numpy arrays
        dicom_to_npy(dcm_organized_path, npy_organized_path)


# Organize DICOM files by PatientID
# Takes about 9 min for 26,296 patients
def organize_dcm_files_by_patient_id(input_directory, output_directory):
    patient_ids = set()
    # Filter only .dcm files
    files = [f for f in os.listdir(input_directory) if f.endswith(".dcm")]

    for filename in tqdm(files, desc="Organizing DICOM files by PatientID"):
        dicom_file_path = os.path.join(input_directory, filename)
        dicom_file = pydicom.dcmread(dicom_file_path)
            
        # Extract the PatientID from the DICOM file
        patient_id = dicom_file.PatientID

        if patient_id not in patient_ids:
            # If it's a new patient_id, create a new subfolder
            os.makedirs(os.path.join(output_directory, patient_id), exist_ok=True)
            patient_ids.add(patient_id)

        # Copy the DICOM file to the corresponding subfolder
        shutil.copy(dicom_file_path, os.path.join(output_directory, patient_id, filename))


# Input: path to each patient's folder
# Output: 3D numpy array of the patient's CT scan
def dcm_to_3D_npy_per_patient(input_folder, output_path):
    # Get all files in the provided folder
    files = os.listdir(input_folder)
    
    # Filter out non-DICOM files
    dicom_files = [file for file in files if file.endswith('.dcm')]
    
    # List to store the images
    images = []
    
    # Get the patient ID from the folder name
    patient_id = os.path.basename(input_folder)

    # Go through each DICOM file
    for dicom_file in dicom_files:
        # Load the DICOM file
        ds = pydicom.dcmread(os.path.join(input_folder, dicom_file))
        
        # Append a tuple of the last ImagePositionPatient value and the pixel array to the list
        images.append((ds.ImagePositionPatient[-1], ds.pixel_array))
    
    # Sort the images by the last ImagePositionPatient value (from high to low)
    images.sort(key=lambda x: x[0], reverse=True)
    
    # # Plot each image
    # for _, img in images:
    #     plt.figure()
    #     plt.imshow(img, cmap=plt.cm.bone)
    #     plt.show()
    
    # Create a 3D numpy array from the 2D images
    volume = np.stack([img for _, img in images])
    
    # Save the 3D array as a .npy file
    np.save(os.path.join(output_path, f"{patient_id}_volume.npy"), volume)


# Input: path to dcm parent folder
# Output: create full conversion of dcm files to npy files in output_path
def dicom_to_npy(input_folder, output_path):
    # Get all folders in the provided folder
    folders = os.listdir(input_folder)
    
    # Filter out non-folders
    patientIDs = [folder for folder in folders if os.path.isdir(os.path.join(input_folder, folder))]
    
    # Go through each patient folder
    for patientID in tqdm(patientIDs, desc="Converting DICOM files to numpy arrays"):
        # Convert the DICOM files to a 3D numpy array
        dcm_to_3D_npy_per_patient(os.path.join(input_folder, patientID), output_path)


# Count the number of files in a DICOM directory
def count_files_in_directory(directory_path):
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    count = 0
    for filename in tqdm(files, desc="Counting files"):
        count += 1
    return count


# Count the number of unique PatientIDs in a directory of DICOM files
def count_unique_patient_ids(input_directory):
    patient_ids = set()
    # Filter only .dcm files
    files = [f for f in os.listdir(input_directory) if f.endswith(".dcm")]

    for filename in tqdm(files, desc="Counting unique PatientIDs"):
        dicom_file_path = os.path.join(input_directory, filename)
        dicom_file = pydicom.dcmread(dicom_file_path)
            
        # Extract the PatientID from the DICOM file
        patient_id = dicom_file.PatientID

        # Add the PatientID to our set of unique PatientIDs
        patient_ids.add(patient_id)
            
    # Return the number of unique PatientIDs
    return len(patient_ids)


# Make a smaller dataset
def make_smaller_dataset(input_parent_directory, output_directory, num_patient=100):
    # Get a list of all patient directories in the input directory
    patient_dirs = [d for d in os.listdir(input_parent_directory) if os.path.isdir(os.path.join(input_parent_directory, d))]

    # Randomly select num_patient directories
    selected_dirs = random.sample(patient_dirs, num_patient)

    for dir_name in tqdm(selected_dirs, desc="Making smaller dataset"):
        input_dir_path = os.path.join(input_parent_directory, dir_name)
        output_dir_path = os.path.join(output_directory, dir_name)

        # Create the output directory
        os.makedirs(output_dir_path, exist_ok=True)

        # Copy all files in the input directory to the output directory
        for filename in os.listdir(input_dir_path):
            if filename.endswith(".dcm"):
                shutil.copy(os.path.join(input_dir_path, filename), os.path.join(output_dir_path, filename))


# Plot a 3D npy file for each patient
def plot_3d_npy(npy_file):
    # Load the 3D numpy array from the .npy file
    volume = np.load(npy_file)

    # Go through each slice in the volume
    for i in range(volume.shape[0]):
        # Create a new figure for each slice
        plt.figure()

        # Display the slice
        plt.imshow(volume[i], cmap=plt.cm.bone)

        # Show the plot
        plt.show()
