import tomopy
import numpy as np
import os

def create_folder(save_path=None,**kwargs):
    try: 
        os.makedirs(save_path)
    except OSError:
        if not os.path.isdir(save_path):
            raise

def create_sinogram(img_stack, theta, pad=True):
    # multiprocessing.freeze_support()
    proj = tomopy.project(img_stack, theta, center=None, emission=True, pad=pad, sinogram_order=False)
    proj = np.transpose(proj, (1, 0, 2))
    return proj

def get_images(img_type = 'foam', dataset_type = 'train'):
    x_train = np.load(img_type + '_' + str(dataset_type) + '.npy')
    return(x_train)

def create_sparse_dataset(x_train_sinograms, 
                          theta,
                          poisson_noise_multiplier = 1e3, # poisson noise multiplier, higher value means higher SNR
                          num_sparse_angles = 10, # number of angles to image per sample (dose remains the same)
                          random = False, # If True, randomly pick angles
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
        if random:
            sparse_angles = np.random.shuffle(np.range(num_angles))[:num_sparse_angles]
        else: 
            # uniformly distribute, but choose a random starting index
            start_ind = np.random.randint(0,num_angles)
            spacing = np.floor(num_angles/num_sparse_angles)
            end_ind = start_ind + spacing*num_sparse_angles
            all_inds = np.arange(start_ind,end_ind,spacing)
            sparse_angles = all_inds%num_angles
        sparse_angles = np.sort(sparse_angles).astype(np.int32)
        sparse_sinogram = x_train_sinograms[ind,sparse_angles,:]

        # add approximate Poisson noise with numpy
        sparse_sinogram = sparse_sinogram + np.sqrt(sparse_sinogram/poisson_noise_multiplier)*np.random.randn(sparse_sinogram.shape[0],sparse_sinogram.shape[1])
        sparse_sinogram[sparse_sinogram<0]=0
        
        # transform sinogram with tomopy
        reconstruction = tomopy.recon(np.expand_dims(sparse_sinogram, axis=1), theta[sparse_angles], center=None, sinogram_order=False, algorithm='gridrec')

        all_mask_inds.append(sparse_angles)
        all_reconstructed_objects.append(reconstruction)
        all_sparse_sinograms.append(sparse_sinogram)

    all_mask_inds = np.stack(all_mask_inds,axis=0)
    all_reconstructed_objects = np.concatenate(all_reconstructed_objects,axis=0)
    all_sparse_sinograms = np.stack(all_sparse_sinograms,axis=0)

    return(all_mask_inds, all_reconstructed_objects, all_sparse_sinograms)
