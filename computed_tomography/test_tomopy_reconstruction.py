"""
Test different reconstruction methods using tomopy
"""

import tomopy
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import create_sinogram, get_sparse_angles
from forward_physics import pad_phantom

if __name__ == '__main__':
    # get a phantom
    img = np.load('foam_train.npy')[0:3]

    num_angles = 180
    num_sparse_angles = 45 # number of angles to image per sample
    random = False

    theta = np.linspace(0, np.pi, num_angles, endpoint=False) # projection angles
    sparse_angles = get_sparse_angles(random, num_angles, num_sparse_angles)
    theta_sparse = theta[sparse_angles]

    # get the sinogram with tomopy
    proj_0 = create_sinogram(img, theta_sparse, pad=True)

    # recontruct with tomopy
    proj_0 = np.transpose(proj_0, axes=[1,0,2])

    rec_0 = tomopy.recon(proj_0, theta_sparse, algorithm='gridrec',center=None, 
                         sinogram_order=False)

    rec_1 = tomopy.recon(proj_0, theta_sparse, algorithm='gridrec',center=None, 
                         sinogram_order=False,filter_name='hann')
    
    rec_2 = tomopy.recon(proj_0, theta_sparse, algorithm='gridrec',center=None, 
                         sinogram_order=False,filter_name='hamming')
    
    rec_3 = tomopy.recon(proj_0, theta_sparse, algorithm='tv',center=None, 
                        sinogram_order=False,reg_par=1e-5,num_iter=100)
    
    rec_4 = tomopy.recon(proj_0, theta_sparse, algorithm='sirt',center=None, 
                        sinogram_order=False, interpolation='LINEAR', num_iter=100)
    # plot reconstruction

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6)
    # change fig relative size
    fig.set_size_inches(9, 2)
    fig.suptitle('Ground Truth vs Reconstruction')
    img_0 = np.squeeze(pad_phantom(torch.Tensor(np.expand_dims(img,axis=3))).numpy(),axis=-1)
 
    ax1.title.set_text('Ground Truth')
    ax1.imshow(img_0[0,:,:],vmin=0,vmax=1)
    # remove all axes
    ax1.axis('off')

    ax2.title.set_text('GridRec1')
    ax2.imshow(rec_0[0,:,:],vmin=0,vmax=1)
    ax2.axis('off')

    ax3.title.set_text('GridRec2')
    ax3.imshow(rec_1[0,:,:],vmin=0,vmax=1)
    ax3.axis('off')

    ax4.title.set_text('GridRec3')
    ax4.imshow(rec_2[0,:,:],vmin=0,vmax=1)
    ax4.axis('off')

    ax5.title.set_text('TV')
    ax5.imshow(rec_3[0,:,:],vmin=0,vmax=1)
    ax5.axis('off')

    ax6.title.set_text('SIRT')
    ax6.imshow(rec_4[0,:,:],vmin=0,vmax=1)
    ax6.axis('off')

    plt.savefig('reconstruction.png')
    plt.show()