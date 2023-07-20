"""
Add a ring artifact to a sinogram
Reconstruct with and without ring artifact correction
"""

import tomopy
import numpy as np
import torch
import matplotlib.pyplot as plt
from computed_tomography.utils import create_sinogram, process_sinogram
from computed_tomography.forward_physics import pad_phantom
from preprocessing.create_images import create_foam_example

if __name__ == '__main__':
    ring_artifact_strength = 0.05
    # make a phantom
    img = create_foam_example()[0][0:3]

    # full set of projection angles
    num_angles = 180
    theta = np.linspace(0, np.pi, num_angles, endpoint=False) # projection angles

    # sparse sinogram parameters
    num_sparse_angles = 45 # number of angles to image per sample
    random = False # If True, randomly pick angles

    # create the sinogram with tomopy
    proj_0 = create_sinogram(img, theta, pad=True)
    proj_0 = np.transpose(proj_0, axes=[1,0,2])

    # no ring artifact
    sparse_angles, reconstruction, sparse_sinogram_0, sparse_sinogram_raw = \
        process_sinogram(proj_0, random, num_sparse_angles, theta, 
                            poisson_noise_multiplier=1e3, remove_ring_artifact=False,
                            ring_artifact_strength=0)
    
    # with ring artifact
    sparse_angles, reconstruction, sparse_sinogram_ring, sparse_sinogram_raw = \
        process_sinogram(proj_0, random, num_sparse_angles, theta, 
                            poisson_noise_multiplier=1e3, remove_ring_artifact=False,
                            ring_artifact_strength=ring_artifact_strength)

    # plot sinograms
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Sinograms')

    ax1.title.set_text('NoRing')
    ax1.imshow(sparse_sinogram_0[0,:,:],vmin=np.min(sparse_sinogram_0),vmax=np.max(sparse_sinogram_0))
    # remove all axes
    ax1.axis('off')

    ax2.title.set_text('Ring')
    ax2.imshow(sparse_sinogram_ring[0,:,:],vmin=np.min(sparse_sinogram_0),vmax=np.max(sparse_sinogram_0))
    ax2.axis('off')

    plt.savefig('ring_sinograms.png')


    random = False
    # no ring artifact
    sparse_angles, reconstruction_0, sparse_sinogram, sparse_sinogram_raw = \
        process_sinogram(proj_0, random, num_sparse_angles, theta, 
                            poisson_noise_multiplier=1e3, remove_ring_artifact=False,
                            ring_artifact_strength=0, random_start_ind=False)

    # with ring artifact 
    sparse_angles, reconstruction_ring, sparse_sinogram, sparse_sinogram_raw = \
        process_sinogram(proj_0, random, num_sparse_angles, theta, 
                            poisson_noise_multiplier=1e3, remove_ring_artifact=False,
                            ring_artifact_strength=ring_artifact_strength, random_start_ind=False)

    # with ring artifact and removed in reconstruction
    sparse_angles, reconstruction_ring_remove, sparse_sinogram, sparse_sinogram_raw = \
        process_sinogram(proj_0, random, num_sparse_angles, theta, 
                            poisson_noise_multiplier=1e3, remove_ring_artifact=True,
                            ring_artifact_strength=ring_artifact_strength, random_start_ind=False)

    # plot reconstruction

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    # change fig relative size
    fig.set_size_inches(5, 2)
    fig.suptitle('Ground Truth vs Reconstruction')
    img_0 = np.squeeze(pad_phantom(torch.Tensor(np.expand_dims(img,axis=3))).numpy(),axis=-1)
 
    ax1.title.set_text('Ground Truth')
    ax1.imshow(img_0[0,:,:],vmin=np.min(img_0),vmax=np.max(img_0))
    # remove all axes
    ax1.axis('off')

    ax2.title.set_text('NoRing')
    ax2.imshow(reconstruction_0[0,:,:],vmin=np.min(img_0),vmax=np.max(img_0))
    ax2.axis('off')

    ax3.title.set_text('Ring1')
    ax3.imshow(reconstruction_ring[0,:,:],vmin=np.min(img_0),vmax=np.max(img_0))
    ax3.axis('off')

    ax4.title.set_text('Ring2')
    ax4.imshow(reconstruction_ring_remove[0,:,:],vmin=np.min(img_0),vmax=np.max(img_0))
    ax4.axis('off')

    plt.savefig('ring_reconstruction.png')

    # get sinogram from reconstruction

    random = False
    # no ring artifact
    sparse_angles, reconstruction_0, sparse_sinogram_0, sparse_sinogram_raw = \
        process_sinogram(proj_0, random, num_sparse_angles, theta, 
                            poisson_noise_multiplier=1e3, remove_ring_artifact=False,
                            ring_artifact_strength=0, random_start_ind=False)

    # with ring artifact 
    sparse_angles, reconstruction_ring, sparse_sinogram_ring, sparse_sinogram_raw = \
        process_sinogram(proj_0, random, num_sparse_angles, theta, 
                            poisson_noise_multiplier=1e3, remove_ring_artifact=False,
                            ring_artifact_strength=ring_artifact_strength, random_start_ind=False)


    theta = np.linspace(0, np.pi, num_sparse_angles, endpoint=False) # projection angles

    proj_1 = create_sinogram(reconstruction_0, theta, pad=False)
    proj_1 = np.transpose(proj_1, axes=[1,0,2])

    proj_1_ring = create_sinogram(reconstruction_ring, theta, pad=False)
    proj_1_ring = np.transpose(proj_1_ring, axes=[1,0,2])

    # compare to original sinogram
    error_no_ring_artifact = np.sum((sparse_sinogram_0-proj_1)**2)
    error_ring_artifact = np.sum((sparse_sinogram_ring-proj_1_ring)**2)
    print(error_ring_artifact/error_no_ring_artifact)