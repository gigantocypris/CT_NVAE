import tomopy
import numpy as np
import os

def create_folder(save_path=None,**kwargs):
    try: 
        os.makedirs(save_path)
    except OSError:
        if not os.path.isdir(save_path):
            raise

def create_sinogram(img_stack, theta, pad=True, add_ring_artifact=False):
    """
    Dimensions of img_stack should be num_images x x_size x y_size
    Output dimensions of proj are num_images x num_angles x num_proj_pix
    """

    # multiprocessing.freeze_support()

    proj = tomopy.project(img_stack, theta, center=None, emission=True, pad=pad, sinogram_order=False)

    # add ring artifact
    if add_ring_artifact:
        num_proj_pix = proj.shape[2]
        ring_artifact = np.random.rand(num_proj_pix)*0.3+0.7
        ring_artifact = np.expand_dims(np.expand_dims(ring_artifact, axis=0),axis=0)
        proj = proj*ring_artifact
    proj = np.transpose(proj, (1, 0, 2))
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

def create_sparse_dataset(x_train_sinograms, 
                          theta,
                          poisson_noise_multiplier = 1e3, # poisson noise multiplier, higher value means higher SNR
                          num_sparse_angles = 10, # number of angles to image per sample (dose remains the same)
                          random = False, # If True, randomly pick angles
                          remove_ring_artifact = False, # If True, remove ring artifact with tomopy correction algorithm
                         ):
 
    x_train_sinograms[x_train_sinograms<0]=0
    num_examples = len(x_train_sinograms)
    num_angles = x_train_sinograms.shape[1]
    
    assert num_angles == len(theta)

    # Create the masks and sparse sinograms
    all_mask_inds = []
    all_reconstructed_objects = []
    all_sparse_sinograms = []
    
    for ind in range(num_examples):
        sparse_angles = get_sparse_angles(random, num_angles, num_sparse_angles)
        sparse_sinogram = x_train_sinograms[ind,sparse_angles,:]

        # add approximate Poisson noise with numpy
        sparse_sinogram = sparse_sinogram + np.sqrt(sparse_sinogram/poisson_noise_multiplier)*np.random.randn(sparse_sinogram.shape[0],sparse_sinogram.shape[1])
        sparse_sinogram[sparse_sinogram<0]=0
        # transform sinogram with tomopy
        # sinogram in tomopy.recon must be num_angles x num_z x num_proj_pix
        reconstruction = tomopy.recon(np.expand_dims(sparse_sinogram, axis=1), theta[sparse_angles], center=None, sinogram_order=False, 
                                      algorithm='gridrec', filter_name='hann')
        if remove_ring_artifact:
            reconstruction = tomopy.misc.corr.remove_ring(reconstruction)

        all_mask_inds.append(sparse_angles)
        all_reconstructed_objects.append(reconstruction)
        all_sparse_sinograms.append(sparse_sinogram)

    all_mask_inds = np.stack(all_mask_inds,axis=0)
    all_reconstructed_objects = np.concatenate(all_reconstructed_objects,axis=0)
    all_sparse_sinograms = np.stack(all_sparse_sinograms,axis=0)

    return(all_mask_inds, all_reconstructed_objects, all_sparse_sinograms)
