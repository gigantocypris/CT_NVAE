"""
PyTorch implementation of the Radon transform
A real space stack of 2D images are each converted to a sinogram (line projections)
"""
import numpy as np
import torch
import kornia

def pad_phantom(phantom,
                ):
    '''
    phantom is batch_size x img_size_x x img_size_y x 1
    '''
    img_size_x = phantom.shape[1]
    img_size_y = phantom.shape[2]

    num_proj_pix = np.sqrt(img_size_x**2 + img_size_y**2) + 2
    num_proj_pix = np.ceil(num_proj_pix / 2.) * 2
    
    odd_x = int((num_proj_pix-img_size_x)%2)
    odd_y = int((num_proj_pix-img_size_y)%2)
        
    padx = int((num_proj_pix-img_size_x)//2)
    pady = int((num_proj_pix-img_size_y)//2)
    
    paddings = (0,0,pady,pady+odd_y,padx,padx+odd_x,0,0)
    phantom = torch.nn.functional.pad(phantom, paddings, 'constant',value=0)
    return(phantom)


def project_torch(phantom, theta_degrees, pad=True, 
                  ):
    '''
    phantom is batch_size x img_size_x x img_size_y x 1
    theta_degrees is batch_size x num_angles
    rotation tutorial: https://kornia-tutorials.readthedocs.io/en/latest/_nbs/rotate_affine.html
    rotation reference: https://kornia.readthedocs.io/en/latest/geometry.transform.html
    '''

    num_angles = theta_degrees.shape[1]
    if pad:
        phantom = pad_phantom(phantom)
    
    phantom = phantom.repeat(1,1,1,num_angles)
    phantom = torch.transpose(phantom, 2,3)
    phantom = torch.transpose(phantom, 1,2)
    phantom = torch.transpose(phantom, 0,1)
    
    # imgs_rot = kornia.geometry.rotate(phantom, -theta_degrees)

    '''
    def rot_no_batch_dim(phantom, theta_degrees): 
        phantom_expand = torch.unsqueeze(phantom, 1)
        rot_angles = -theta_degrees[0]
        return kornia.geometry.rotate(phantom_expand, rot_angles)
    
    rot_no_batch_dim_1 = torch.func.functionalize(rot_no_batch_dim)

    imgs_rot = torch.vmap(rot_no_batch_dim_1,in_dims=1,out_dims=1)(phantom, -theta_degrees[None,:,:])
    '''
    imgs_rot = []
    for i in range(theta_degrees.shape[0]):
        imgs_rot.append(kornia.geometry.rotate(torch.unsqueeze(phantom[:,i,:,:],1), -theta_degrees[i]))
    imgs_rot = torch.cat(imgs_rot, dim=1)
    sino = torch.sum(imgs_rot, 2)
    
    # transpose back
    sino = torch.transpose(sino, 0, 1)

    return(sino)