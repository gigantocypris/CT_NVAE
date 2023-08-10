"""
Compare tomopy and forward_physics.py
"""

import tomopy
import numpy as np
import torch
import matplotlib.pyplot as plt
from computed_tomography.utils import create_sinogram
from computed_tomography.forward_physics import project_torch
from preprocessing.create_images import create_foam_example

if __name__ == '__main__':
    theta = np.linspace(0, np.pi, 180, endpoint=False) # projection angles

    # get a phantom
    img = create_foam_example()[0][0:3]

    # get the sinogram with tomopy
    proj_0 = create_sinogram(img, theta, pad=True)

    # get the sinogram with forward_physics.py
    phantom = torch.Tensor(img)

    theta_degrees = theta*180/np.pi
    theta_degrees = torch.stack([torch.Tensor(theta_degrees)]*phantom.shape[0], axis=0)
    proj_1 = project_torch(phantom[:,:,:,None], theta_degrees, pad=True)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Sinogram Comparison')
    ax1.imshow(proj_0[0,:,:])
    ax2.imshow(proj_1[0,:,:])
    plt.savefig('sinogram_comparison.png')

    plt.figure()
    plt.title('Difference Map')
    plt.imshow(proj_0[0,:,:]-proj_1.numpy()[0,:,:])
    plt.savefig('sinogram_difference.png')

    print('Max Absolute Difference: ' + str(np.max(np.abs(proj_0-proj_1.numpy()))))