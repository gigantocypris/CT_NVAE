import os
import glob
import numpy as np
import pydicom
import shutil
from tqdm import tqdm


# Input: path to unorganized dcm files
# Output: create organized dcm files in output_directory
def organize_dcm_files_by_Instance(input_directory, output_directory):
    instance_ids = set()
    # Filter only .dcm files
    files = glob.glob(os.path.join(input_directory, "*.dcm"))

    for filename in tqdm(files, desc="Organizing DICOM files by StudyInstanceUID"):
        try:
            dicom_file = pydicom.dcmread(filename)
            
            # Extract the PatientID from the DICOM file
            instance_id = dicom_file.StudyInstanceUID

            if instance_id not in instance_ids:
                # If it's a new instance_id, create a new subfolder
                os.makedirs(os.path.join(output_directory, instance_id), exist_ok=True)
                instance_ids.add(instance_id)

            # Copy the DICOM file to the corresponding subfolder
            shutil.copy(filename, os.path.join(output_directory, instance_id, os.path.basename(filename)))
        except Exception as e:
            print(f"Error processing file {filename}: {e}")


# Input: path to organized dcm parent folder
# Output: create full conversion of dcm files to npy files in output_path
def dicom_to_npy(input_folder, output_path):
    try:
        # Get all folders in the provided folder
        folders = os.listdir(input_folder)
    except FileNotFoundError:
        print(f"Input folder {input_folder} does not exist.")
        return

    if not os.path.exists(output_path):
        print(f"Output path {output_path} does not exist.")
        return

    # Filter out non-folders
    patientIDs = [folder for folder in folders if os.path.isdir(os.path.join(input_folder, folder))]

    # Go through each patient folder
    for patientID in tqdm(patientIDs, desc="Converting DICOM files to numpy arrays"):
        # Convert the DICOM files to a 3D numpy array
        dcm_to_3D_npy_per_patient(os.path.join(input_folder, patientID), output_path)


# Input: path to each patient's folder
# Output: 3D numpy array of the patient's CT scan
def dcm_to_3D_npy_per_patient(input_folder, output_path):
    try:
        # Get all files in the provided folder
        files = os.listdir(input_folder)
    except FileNotFoundError:
        print(f"Input folder {input_folder} does not exist.")
        return

    # Filter out non-DICOM files
    dicom_files = [file for file in files if file.endswith('.dcm')]

    # List to store the images
    images = []

    # Get the patient ID from the folder name
    patient_id = os.path.basename(input_folder)

    # Go through each DICOM file
    for dicom_file in dicom_files:
        try:
            # Load the DICOM file
            ds = pydicom.dcmread(os.path.join(input_folder, dicom_file), force=True)

            # Append a tuple of the last ImagePositionPatient value and the pixel array to the list
            images.append((ds.ImagePositionPatient[-1], ds.pixel_array))
        except (AttributeError, FileNotFoundError):
            print(f"Error reading DICOM file {dicom_file}. Skipping this file.")
            continue

    # Sort the images by the last ImagePositionPatient value (from high to low)
    images.sort(key=lambda x: x[0], reverse=True)

    # Check if there are any images
    if not images:
        print(f"No valid DICOM files found in {input_folder}.")
        return

    # Get the shape of the first image
    first_image_shape = images[0][1].shape

    # Check if all images have the same shape
    if any(img.shape != first_image_shape for _, img in images):
        print(f"Not all DICOM files in {input_folder} have the same shape. Skipping this patient.")
        return

    try:
        # Create a 3D numpy array from the 2D images
        volume = np.stack([img for _, img in images])

        # Save the 3D array as a .npy file
        np.save(os.path.join(output_path, f"{patient_id}.npy"), volume)
    except Exception as e:
        print(f"Error saving numpy file for patient {patient_id}: {e}")
