"""
Test different reconstruction methods using tomopy for obtaining sparse_reconstruction_mask
"""

import tomopy
import numpy as np
import torch
import matplotlib.pyplot as plt
from computed_tomography.utils import create_sinogram, get_sparse_angles
from computed_tomography.forward_physics import pad_phantom
from preprocessing.create_images import create_foam_example

if __name__ == '__main__':
    # make a phantom
    img = create_foam_example()[0][0:3] # exact values of phantom are unused
    num_angles = 180
    num_sparse_angles = 10 # number of angles to image per sample
    random = True

    theta = np.linspace(0, np.pi, num_angles, endpoint=False) # projection angles
    sparse_angles = get_sparse_angles(random, num_angles, num_sparse_angles)
    theta_sparse = theta[sparse_angles]

    # get the sinogram with tomopy
    proj_0 = create_sinogram(img, theta_sparse, pad=True)

    # get sparse_sinogram_mask
    sparse_sinogram_mask = np.ones_like(proj_0)

    # recontruct with tomopy
    proj_0 = np.transpose(proj_0, axes=[1,0,2])
    sparse_sinogram_mask = np.transpose(sparse_sinogram_mask, axes=[1,0,2])

    rec_0 = tomopy.recon(sparse_sinogram_mask, theta_sparse, algorithm='gridrec',center=None, 
                         sinogram_order=False)

    rec_1 = tomopy.recon(sparse_sinogram_mask, theta_sparse, algorithm='gridrec',center=None, 
                         sinogram_order=False,filter_name='hann')
    
    rec_2 = tomopy.recon(sparse_sinogram_mask, theta_sparse, algorithm='fbp',center=None, 
                         sinogram_order=False,filter_name='none')
    
    rec_3 = tomopy.recon(sparse_sinogram_mask, theta_sparse, algorithm='tv',center=None, 
                        sinogram_order=False,reg_par=1e-5,num_iter=100)
    
    rec_3_ref = tomopy.recon(proj_0, theta_sparse, algorithm='tv',center=None, 
                    sinogram_order=False,reg_par=1e-5,num_iter=100)
    
    rec_4 = tomopy.recon(sparse_sinogram_mask, theta_sparse, algorithm='sirt',center=None, 
                        sinogram_order=False, interpolation='LINEAR', num_iter=100)
    # plot reconstruction

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5)
    # change fig relative size
    fig.set_size_inches(9, 2)
    img_0 = np.squeeze(pad_phantom(torch.Tensor(np.expand_dims(img,axis=3))).numpy(),axis=-1)

    ax0.title.set_text('GridRec1')
    ax0.imshow(rec_0[0,:,:])
    ax0.axis('off')

    ax1.title.set_text('GridRec2')
    ax1.imshow(rec_1[0,:,:])
    ax1.axis('off')

    ax2.title.set_text('FBP')
    ax2.imshow(rec_2[0,:,:])
    ax2.axis('off')

    ax3.title.set_text('TV')
    ax3.imshow(rec_3[0,:,:])
    ax3.axis('off')

    ax4.title.set_text('SIRT')
    ax4.imshow(rec_4[0,:,:])
    ax4.axis('off')

    plt.savefig('reconstruction_mask.png')

    plt.figure()
    plt.imshow(rec_3[0,:,:])
    plt.colorbar()
    plt.savefig('reconstruction_mask_TV.png')

    plt.figure()
    plt.imshow(rec_3_ref[0,:,:])
    plt.colorbar()
    plt.savefig('reconstruction_obj_TV.png')




