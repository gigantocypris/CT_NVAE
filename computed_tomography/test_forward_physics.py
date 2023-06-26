"""
Compare tomopy and forward_physics.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import create_sinogram
from forward_physics import project_torch

if __name__ == '__main__':
    theta = np.linspace(0, np.pi, 180, endpoint=False) # projection angles

    # get a phantom
    img = np.load('foam_train.npy')[0:3]

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
    plt.show()

    plt.figure()
    plt.title('Difference Map')
    plt.imshow(proj_0[0,:,:]-proj_1.numpy()[0,:,:])
    plt.show()

    print('Max Absolute Difference: ' + str(np.max(np.abs(proj_0-proj_1.numpy()))))