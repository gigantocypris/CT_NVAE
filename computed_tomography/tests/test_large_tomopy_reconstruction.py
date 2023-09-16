"""
Test different reconstruction methods using tomopy
"""

import tomopy
import numpy as np
import torch
import matplotlib.pyplot as plt
from computed_tomography.utils import create_sinogram, process_sinogram
from computed_tomography.forward_physics import pad_phantom
import glob
from metrics.utils import compare

if __name__ == '__main__':

    # inputs
    img_folder_path = 'images_covid_100ex'
    img_ind = 0
    slice_ind = 0
    random = True
    force_angle_array = None
    num_sparse_angles = 180 # angles in the sparse sinogram
    num_angles = 180 # angles in the full sinogram
    pnm = np.floor(1000000/num_sparse_angles)
    ring_artifact_strength = 0
    algorithm = 'tv'

    # start processing

    sinogram_files = np.sort(glob.glob(img_folder_path + '/*_sinogram.npy'))
    filepath_sino = sinogram_files[img_ind] # sinogram filepath
    filepath_gt = ''.join(filepath_sino.split('_sinogram')) # ground truth filepath

    x_train = np.load(filepath_gt)
    theta = np.load(img_folder_path + '/theta.npy')
    # reconstruction with noise
    x_train_sinogram = np.load(filepath_sino)
    sparse_angles, sparse_reconstruction, sparse_sinogram_raw, sparse_sinogram = \
        process_sinogram(np.transpose(x_train_sinogram,axes=[1,0,2]), random, force_angle_array, num_sparse_angles, theta, 
                        poisson_noise_multiplier = pnm, remove_ring_artifact = False, 
                        ring_artifact_strength = ring_artifact_strength, algorithm=algorithm)
    sparse_reconstruction = sparse_reconstruction[slice_ind]

    # reconstruction without noise
    img = x_train[slice_ind]
    img = np.expand_dims(img, axis=0)
    theta = np.linspace(0, np.pi, num_angles, endpoint=False) # projection angles

    # get the sinogram with tomopy
    proj_full = create_sinogram(img, theta, pad=True) # z_slices x num_angles x num_proj_pix

    # recontruct with tomopy
    proj_full = np.transpose(proj_full, axes=[1,0,2]) # num_angles x z_slices x num_proj_pix

    if algorithm == 'gridrec':
        rec_full = tomopy.recon(proj_full, theta, algorithm='gridrec',center=None, 
                            sinogram_order=False)
    elif algorithm == 'tv':    
        rec_full = tomopy.recon(proj_full, theta, algorithm='tv',center=None, 
                            sinogram_order=False,reg_par=1e-5,num_iter=1)
    elif algorithm == 'sirt':
        rec_full = tomopy.recon(proj_full, theta, algorithm='sirt',center=None, 
                            sinogram_order=False, interpolation='LINEAR', num_iter=1)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    # change fig relative size
    fig.set_size_inches(4, 2)
    fig.suptitle('Ground Truth vs Reconstruction')
    img_0 = np.squeeze(pad_phantom(torch.Tensor(np.expand_dims(img,axis=3))).numpy(),axis=-1)
 
    img_0 = img_0[0,:,:]
    rec_full = rec_full[0,:,:]

    err_vec_NVAE, err_string_NVAE = compare(img_0, sparse_reconstruction, verbose=True)


    vmax = np.max(img_0)
    ax1.title.set_text('GT')
    ax1.imshow(img_0,vmin=0,vmax=vmax)
    # remove all axes
    ax1.axis('off')

    ax2.title.set_text('No Noise, Full')
    ax2.imshow(rec_full,vmin=0,vmax=vmax)
    ax2.axis('off')

    ax3.title.set_text('Diff')
    ax3.imshow(np.abs(rec_full-img_0))
    ax3.axis('off')

    ax4.title.set_text('Noisy, Sparse')
    ax4.imshow(sparse_reconstruction,vmin=0,vmax=vmax)
    ax4.axis('off')


    plt.savefig('large_reconstruction.png', dpi=300, bbox_inches='tight')
