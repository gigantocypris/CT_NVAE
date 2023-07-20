import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tomopy
import random
import shutil
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm
from utils import create_sinogram

def create_sinogram_nib(nib_file_path, target_dir, theta, pad = True, add_ring_artifact=False, save_numpy=True):
    img = nib.load(nib_file_path)
    data = img.get_fdata()
    data = data.transpose((2, 0, 1))
    file_name = os.path.splitext(os.path.basename(nib_file_path))[0]

    data += 2048
    data /= np.max(data)
    data[data < 0] = 0
    proj = create_sinogram(data, theta, pad=pad, add_ring_artifact=add_ring_artifact)

    sinogram_file_path = f"{target_dir}/{file_name}_sinogram.npy"
    np.save(sinogram_file_path, proj)

    if save_numpy:
        # Load .nib file and save it as .npy file
        npy_file_path = f"{target_dir}/{file_name}.npy"
        np.save(npy_file_path, data)
        # print(f"Successfully converted {nib_file_path} to {npy_file_path}.")
    return(data, proj, file_name)

def create_sinogram_npy(npy_file_path, target_dir, theta, pad = True, add_ring_artifact=False):
    data = np.load(npy_file_path).astype(float) # (num_images, x_size, y_size)
    file_name = os.path.splitext(os.path.basename(npy_file_path))[0]

    data += 2048
    data /= np.max(data)
    data[data < 0] = 0
    proj = create_sinogram(data, theta, pad=pad, add_ring_artifact=add_ring_artifact) # (num_images, num_angles, num_proj_pix)

    sinogram_file_path = f"{target_dir}/{file_name}_sinogram.npy"
    np.save(sinogram_file_path, proj)
    return(data, proj, file_name)

def visualize(data, proj, file_name, target_dir):
    """Visualize the output from create_sinogram_nib"""

    # Plot input images and save plot as PNG file
    num_images = data.shape[0]
    num_rows = num_images // 5 if num_images % 5 == 0 else (num_images // 5) + 1
    fig, axes = plt.subplots(num_rows, 5, figsize=(10, 2 * num_rows))
    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            break
        slice_img = data[i, :, :]
        ax.imshow(slice_img, cmap='gray')
        ax.set_title(f"Slice: {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{target_dir}/plot_{file_name}.png")
    # plt.show()
    
    # Plot sinogram images and save the plot as PNG file
    num_images_to_plot = 18
    interval = proj.shape[0] // num_images_to_plot

    if num_images_to_plot%3 == 0:
        rows = (num_images_to_plot // 3)
    else:
        rows = (num_images_to_plot // 3) + 1
    
    fig, axes = plt.subplots(rows, 3, figsize=(12, 6 * rows))

    for i, ax in enumerate(axes.flat):
        index = i * interval
        if index >= proj.shape[0]:
            break

        image = proj[index, :, :]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Index: {index}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{target_dir}/sinograms_{file_name}.png")
    # plt.show()



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
        ds = pydicom.dcmread(os.path.join(input_folder, dicom_file), force=True)
        
        # Append a tuple of the last ImagePositionPatient value and the pixel array to the list
        images.append((ds.ImagePositionPatient[-1], ds.pixel_array))
    
    # Sort the images by the last ImagePositionPatient value (from high to low)
    images.sort(key=lambda x: x[0], reverse=True)
    
    
    # Create a 3D numpy array from the 2D images
    volume = np.stack([img for _, img in images])
    
    # Save the 3D array as a .npy file
    np.save(os.path.join(output_path, f"{patient_id}.npy"), volume)


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


# Make a small dataset
def make_small_dataset(input_parent_directory, output_directory, num_patient=100):
    # Get a list of all patient directories in the input directory
    patient_dirs = [d for d in os.listdir(input_parent_directory) if os.path.isdir(os.path.join(input_parent_directory, d))]

    # Randomly select num_patient directories
    selected_dirs = random.sample(patient_dirs, num_patient)

    for dir_name in tqdm(selected_dirs, desc="Making small dataset"):
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
