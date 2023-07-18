import tomopy
import numpy as np
import os
from scipy.stats import truncnorm

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
