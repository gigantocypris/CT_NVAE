"""
Add a ring artifact to a sinogram
Reconstruct with and without ring artifact correction
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

    # get the sinogram with tomopy, without ring artifact
    proj_0 = create_sinogram(img, theta_sparse, pad=True, add_ring_artifact=False)

    # get the sinogram with tomopy, adding ring artifact
    proj_0_ring = create_sinogram(img, theta_sparse, pad=True, add_ring_artifact=True)

    # plot sinograms
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Sinograms')

    ax1.title.set_text('NoRing')
    ax1.imshow(proj_0[0,:,:],vmin=np.min(proj_0),vmax=np.max(proj_0))
    # remove all axes
    ax1.axis('off')

    ax2.title.set_text('Ring')
    ax2.imshow(proj_0_ring[0,:,:],vmin=np.min(proj_0),vmax=np.max(proj_0))
    ax2.axis('off')

    plt.savefig('ring_sinograms.png')

    # recontruct with tomopy
    proj_0 = np.transpose(proj_0, axes=[1,0,2])
    rec_0 = tomopy.recon(proj_0, theta_sparse, algorithm='gridrec',center=None,
                         sinogram_order=False)
    
    proj_0_ring = np.transpose(proj_0_ring, axes=[1,0,2])
    rec_0_ring = tomopy.recon(proj_0_ring, theta_sparse, algorithm='gridrec',center=None, 
                         sinogram_order=False)
    rec_0_corr = tomopy.misc.corr.remove_ring(rec_0_ring)

    # plot reconstruction

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    # change fig relative size
    fig.set_size_inches(5, 2)
    fig.suptitle('Ground Truth vs Reconstruction')
    img_0 = np.squeeze(pad_phantom(torch.Tensor(np.expand_dims(img,axis=3))).numpy(),axis=-1)
 
    ax1.title.set_text('Ground Truth')
    ax1.imshow(img_0[0,:,:],vmin=0,vmax=1)
    # remove all axes
    ax1.axis('off')

    ax2.title.set_text('NoRing')
    ax2.imshow(rec_0[0,:,:],vmin=0,vmax=1)
    ax2.axis('off')

    ax3.title.set_text('Ring1')
    ax3.imshow(rec_0_ring[0,:,:],vmin=0,vmax=1)
    ax3.axis('off')

    ax4.title.set_text('Ring2')
    ax4.imshow(rec_0_corr[0,:,:],vmin=0,vmax=1)
    ax4.axis('off')

    plt.savefig('ring_reconstruction.png')
