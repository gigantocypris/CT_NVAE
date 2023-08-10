import tomopy
import numpy as np
from scipy.stats import truncnorm

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
    # truncate normal distribution to be within 2 standard deviations
    ring_artifact = truncnorm.rvs(-2,2,loc=1, scale=ring_artifact_strength, size=num_proj_pix)
    ring_artifact = np.expand_dims(np.expand_dims(ring_artifact, axis=0),axis=0)
    proj = proj*ring_artifact
    return proj

def get_sparse_angles(random, num_angles, num_sparse_angles, random_start_ind=True):
    if random:
        angle_array = np.arange(num_angles)
        np.random.shuffle(angle_array)
        sparse_angles = angle_array[:num_sparse_angles]
    else: 
        # uniformly distribute
        if random_start_ind: # choose a random starting index
            start_ind = np.random.randint(0,num_angles)
        else:
            start_ind = 0
        spacing = np.floor(num_angles/num_sparse_angles)
        end_ind = start_ind + spacing*num_sparse_angles
        all_inds = np.arange(start_ind,end_ind,spacing)
        sparse_angles = all_inds%num_angles
    sparse_angles = np.sort(sparse_angles).astype(np.int32)
    return(sparse_angles)

def reconstruct_sinogram(proj, theta, remove_ring_artifact=False, algorithm='gridrec'):
    """
    transform sinogram with tomopy
    proj in tomopy.recon must be num_angles x num_z x num_proj_pix
    """

    if algorithm=='gridrec':
        reconstruction = tomopy.recon(proj, theta, center=None, sinogram_order=False, algorithm='gridrec')
    elif algorithm=='sirt':
        reconstruction = tomopy.recon(proj, theta, algorithm='sirt',center=None, 
                            sinogram_order=False, interpolation='LINEAR', num_iter=20)
    elif algorithm=='tv':
        reconstruction = tomopy.recon(proj, theta, algorithm='tv',center=None, 
                            sinogram_order=False, num_iter=20, reg_par=1e-3)
    
    if remove_ring_artifact:
        reconstruction = tomopy.misc.corr.remove_ring(reconstruction)
    return reconstruction

def process_sinogram(input_sinogram, random, num_sparse_angles, theta, 
                     poisson_noise_multiplier=1e3, remove_ring_artifact=False,
                     ring_artifact_strength=0.3, random_start_ind=True,
                     algorithm='gridrec'):
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
    sparse_angles = get_sparse_angles(random, num_angles, num_sparse_angles, random_start_ind=random_start_ind)
    sparse_sinogram_raw = exp_sinogram[sparse_angles,:,:]

    
    sparse_sinogram = -np.log(sparse_sinogram_raw) # linearize the sinogram
    reconstruction = reconstruct_sinogram(sparse_sinogram, theta[sparse_angles], 
                                          remove_ring_artifact=remove_ring_artifact,
                                          algorithm=algorithm)

    
    return sparse_angles, reconstruction, sparse_sinogram_raw, sparse_sinogram
