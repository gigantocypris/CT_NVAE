"""
Test different reconstruction methods using tomopy
"""

import tomopy
import numpy as np
import torch
import matplotlib.pyplot as plt
from computed_tomography.utils import create_sinogram, get_sparse_angles
from computed_tomography.forward_physics import pad_phantom
from preprocessing.create_images import create_foam_example
import glob
from metrics.utils import compare

if __name__ == '__main__':

    # inputs
    img_folder_path = 'images_brain_100ex'
    img_ind = 0
    slice_ind = 0
    
    num_angles = 180 # total angles

    # start processing

    sinogram_files = np.sort(glob.glob(img_folder_path + '/*_sinogram.npy'))
    filepath_sino = sinogram_files[img_ind] # sinogram filepath
    filepath_gt = ''.join(filepath_sino.split('_sinogram')) # ground truth filepath

    img = np.load(filepath_gt)[slice_ind]
    img = np.expand_dims(img, axis=0)

    theta = np.linspace(0, np.pi, num_angles, endpoint=False) # projection angles

    # get the sinogram with tomopy
    proj_full = create_sinogram(img, theta, pad=True)

    # recontruct with tomopy
    proj_full = np.transpose(proj_full, axes=[1,0,2])

    rec_full = tomopy.recon(proj_full, theta, algorithm='gridrec',center=None, 
                         sinogram_order=False)

    # rec_full = tomopy.recon(proj_full, theta, algorithm='tv',center=None, 
    #                     sinogram_order=False,reg_par=1e-5,num_iter=100)

    # rec_full = tomopy.recon(proj_full, theta, algorithm='sirt',center=None, 
    #                     sinogram_order=False, interpolation='LINEAR', num_iter=10)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # change fig relative size
    fig.set_size_inches(4, 2)
    fig.suptitle('Ground Truth vs Reconstruction')
    img_0 = np.squeeze(pad_phantom(torch.Tensor(np.expand_dims(img,axis=3))).numpy(),axis=-1)
 
    img_0 = img_0[0,:,:]
    rec_full = rec_full[0,:,:]

    err_vec_NVAE, err_string_NVAE = compare(img_0, rec_full, verbose=True)


    vmax = np.max(img_0)
    ax1.title.set_text('GT')
    ax1.imshow(img_0,vmin=0,vmax=vmax)
    # remove all axes
    ax1.axis('off')

    ax2.title.set_text('Full')
    ax2.imshow(rec_full,vmin=0,vmax=vmax)
    ax2.axis('off')

    ax3.title.set_text('Diff')
    ax3.imshow(np.abs(rec_full-img_0))
    ax3.axis('off')

    plt.savefig('large_reconstruction.png', dpi=300, bbox_inches='tight')
