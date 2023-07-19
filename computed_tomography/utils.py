import tomopy
import numpy as np
import os
from scipy.stats import truncnorm
import random
import shutil
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm


def create_folder(save_path=None,**kwargs):
    try: 
        os.makedirs(save_path)
    except OSError:
        if not os.path.isdir(save_path):
            raise

def create_sinogram(img_stack, theta, pad=True):
    """
    Dimensions of img_stack should be num_images x x_size x y_size
    Output dimensions of proj are num_images x num_angles x num_proj_pix
    """

    # multiprocessing.freeze_support()

    proj = tomopy.project(img_stack, theta, center=None, emission=True, pad=pad, sinogram_order=False)
    proj = np.transpose(proj, (1, 0, 2))
    return proj

def add_ring_artifact(proj, ring_artifact_strength=0.15):
    num_proj_pix = proj.shape[2]
    ring_artifact = truncnorm.rvs(-2,2,loc=1, scale=ring_artifact_strength, size=num_proj_pix)
    ring_artifact = np.expand_dims(np.expand_dims(ring_artifact, axis=0),axis=0)
    proj = proj*ring_artifact
    return proj

def get_images(rank, img_type = 'foam', dataset_type = 'train'):
    x_train = np.load(img_type + '_' + str(dataset_type) + '_' + str(rank) + '.npy')
    return(x_train)

def get_sparse_angles(random, num_angles, num_sparse_angles):
    if random:
        angle_array = np.arange(num_angles)
        np.random.shuffle(angle_array)
        sparse_angles = angle_array[:num_sparse_angles]
    else: 
        # uniformly distribute, but choose a random starting index
        start_ind = np.random.randint(0,num_angles)
        spacing = np.floor(num_angles/num_sparse_angles)
        end_ind = start_ind + spacing*num_sparse_angles
        all_inds = np.arange(start_ind,end_ind,spacing)
        sparse_angles = all_inds%num_angles
    sparse_angles = np.sort(sparse_angles).astype(np.int32)
    return(sparse_angles)

def process_sinogram(input_sinogram, random, num_sparse_angles, theta, 
                     poisson_noise_multiplier=1e3, remove_ring_artifact=False,
                     ring_artifact_strength=0.3):
    """
    process sinogram to make it artificially sparse and reconstruct with tomopy
    input sinogram is num_angles x num_z x num_proj_pix
    input sinogram is linearized by -log(x)
    """
    exp_sinogram = np.exp(-input_sinogram) # switch to raw data
    if ring_artifact_strength>0:
        exp_sinogram = add_ring_artifact(exp_sinogram, ring_artifact_strength=ring_artifact_strength)

    # add approximate Poisson noise with truncated normal (within 2 standard deviations)
    poisson_noise_std = np.sqrt(exp_sinogram/poisson_noise_multiplier)
    poisson_noise = truncnorm.rvs(-2,2,loc=0, scale=poisson_noise_std, size=exp_sinogram.shape)
    exp_sinogram = exp_sinogram + poisson_noise

    # remove angles
    num_angles = len(theta)
    sparse_angles = get_sparse_angles(random, num_angles, num_sparse_angles)
    sparse_sinogram = exp_sinogram[sparse_angles,:,:]

    # transform sinogram with tomopy
    # sinogram in tomopy.recon must be num_angles x num_z x num_proj_pix
    sparse_sinogram = -np.log(sparse_sinogram) # linearize the sinogram
    reconstruction = tomopy.recon(sparse_sinogram, theta[sparse_angles], center=None, sinogram_order=False, algorithm='gridrec')
    # reconstruction = tomopy.recon(sparse_sinogram, theta[sparse_angles], algorithm='sirt',center=None, 
    #                     sinogram_order=False, interpolation='LINEAR', num_iter=20)
    
    if remove_ring_artifact:
        reconstruction = tomopy.misc.corr.remove_ring(reconstruction)
    
    return sparse_angles, reconstruction, sparse_sinogram

def create_sparse_dataset(x_train_sinograms, 
                          theta,
                          poisson_noise_multiplier = 1e3, # poisson noise multiplier, higher value means higher SNR
                          num_sparse_angles = 10, # number of angles to image per sample (dose remains the same)
                          random = False, # If True, randomly pick angles
                          remove_ring_artifact = False, # If True, remove ring artifact with tomopy correction algorithm
                         ):
 
    """Artifically remove angles from the sinogram and reconstruct with tomopy to emulate a training dataset for CT_NVAE"""
    x_train_sinograms[x_train_sinograms<0]=0
    num_examples = len(x_train_sinograms)
    num_angles = x_train_sinograms.shape[1]
    
    assert num_angles == len(theta)

    # Create the masks and sparse sinograms
    all_mask_inds = []
    all_reconstructed_objects = []
    all_sparse_sinograms = []
    
    for ind in range(num_examples):
        input_sinogram = x_train_sinograms[ind,:,:]
        input_sinogram = np.expand_dims(input_sinogram, axis=1)
        sparse_angles, reconstruction, sparse_sinogram = process_sinogram(input_sinogram, random, 
                                                                          num_sparse_angles, theta, 
                                                                          poisson_noise_multiplier = poisson_noise_multiplier, 
                                                                          remove_ring_artifact = remove_ring_artifact)

        all_mask_inds.append(sparse_angles)
        all_reconstructed_objects.append(reconstruction)
        all_sparse_sinograms.append(np.squeeze(sparse_sinogram, axis=1))

    all_mask_inds = np.stack(all_mask_inds,axis=0)
    all_reconstructed_objects = np.concatenate(all_reconstructed_objects,axis=0)
    all_sparse_sinograms = np.stack(all_sparse_sinograms,axis=0)
    return(all_mask_inds, all_reconstructed_objects, all_sparse_sinograms)



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
