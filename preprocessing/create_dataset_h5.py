"""
Process sinograms into a dataset for CT_NVAE training.
"""

import argparse
import numpy as np
from computed_tomography.utils import process_sinogram
from computed_tomography.forward_physics import pad_phantom
import time
import glob
import torch
import h5py

def main(args, dataset_type):

    ### INPUT ###
    sub_dir = args.dir + '/' + dataset_type
    theta = np.load(args.dir + '/theta.npy') # projection angles
    num_sparse_angles = args.num_sparse_angles # number of angles to image per sample (dose remains the same)
    random = args.random # If True, randomly pick angles
    #############

    print(f'sub_dir is {sub_dir}')

    sinogram_files = np.sort(glob.glob(sub_dir + '/*_sinogram.npy'))
    print(f'Total number of sinograms found: {len(sinogram_files)}')

    h5_filename = args.dir + '/' + dataset_type + '.h5'

    with h5py.File(h5_filename, 'w') as h5_file:
        total_slices = 0
        for i in range(len(sinogram_files)):
            print(f'processing sinogram {i} of {len(sinogram_files)}')
            filepath_sino = sinogram_files[i] # sinogram filepath
            print(f'filepath_sino is {filepath_sino}')
            filepath_gt = ''.join(filepath_sino.split('_sinogram')) # ground truth filepath
            print(f'filepath_gt is {filepath_gt}')

            x_train = np.load(filepath_gt)
            x_train_sinogram = np.load(filepath_sino)
            print(f'x_train has shape {x_train.shape}')
            print(f'x_train_sinogram has shape {x_train_sinogram.shape}')

            # make sparse sinogram and reconstruct
            sparse_angles, reconstruction, sparse_sinogram_raw, sparse_sinogram = \
                process_sinogram(np.transpose(x_train_sinogram,axes=[1,0,2]), random, num_sparse_angles, theta, 
                                poisson_noise_multiplier = args.pnm, remove_ring_artifact = False, 
                                ring_artifact_strength = args.ring_artifact_strength, algorithm=args.algorithm)

            
            x_train_pad = pad_phantom(torch.Tensor(np.expand_dims(x_train.astype(np.float32), axis=-1)))
            x_train_pad = np.squeeze(x_train_pad.numpy(),axis=-1)
            mask_inds = np.repeat(np.expand_dims(sparse_angles,axis=0),x_train.shape[0],axis=0)
            sparse_sinogram = np.transpose(sparse_sinogram,axes=[1,0,2])
            sparse_sinogram_raw = np.transpose(sparse_sinogram_raw,axes=[1,0,2])
            object_id_3d = np.repeat(np.expand_dims(np.sum(reconstruction,axis=0),axis=0),x_train.shape[0],axis=0)

            # save each slice to h5 file
            for slice_ind in range(x_train_pad.shape[0]):
                slice_group = h5_file.create_group(f'slice_{total_slices}')
                slice_group.create_dataset('x_train_img', data=x_train_pad[slice_ind])
                slice_group.create_dataset('x_train_sinogram', data=x_train_sinogram[slice_ind])
                slice_group.create_dataset('mask_inds', data=mask_inds[slice_ind])
                slice_group.create_dataset('reconstructed_object', data=reconstruction[slice_ind])
                slice_group.create_dataset('sparse_sinogram', data=sparse_sinogram[slice_ind])
                slice_group.create_dataset('sparse_sinogram_raw', data=sparse_sinogram_raw[slice_ind])
                slice_group.create_dataset('object_id_3d', data=object_id_3d[slice_ind])
                total_slices += 1

        num_proj_pix = x_train_sinogram.shape[-1]
        x_size = x_train_pad.shape[1]
        y_size = x_train_pad.shape[2]
        h5_file.create_dataset('theta', data=theta)
        h5_file.create_dataset('num_proj_pix', data=num_proj_pix)
        h5_file.create_dataset('x_size', data=x_size)
        h5_file.create_dataset('y_size', data=y_size)
        h5_file.create_dataset('total_slices', data=total_slices)
        h5_file.create_dataset('total_objects', data=len(sinogram_files))
        np.save(args.dir + '/' + dataset_type + '_num_proj_pix.npy', num_proj_pix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('--dir', dest='dir', help='directory containing splits of ground truth/sinograms and theta')
    parser.add_argument('--pnm', dest='pnm', type=float, help='poisson noise multiplier, higher value means higher SNR', default=1e3)
    parser.add_argument('--sparse', dest='num_sparse_angles', type=int, help='number of angles to image per sample (dose remains the same)', default=10)
    parser.add_argument('--random', dest='random', type=bool, help='If True, randomly pick angles', default=True)
    parser.add_argument('--ring', dest='ring_artifact_strength', type=float, help='if >0, add ring artifact to sinograms', default=0.0)
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='algorithm to use for reconstruction', default='gridrec', 
                        choices=['gridrec', 'sirt', 'tv'])
    args = parser.parse_args()


    start_time = time.time()
    main(args, 'train')
    main(args, 'valid')
    main(args, 'test')
    end_time = time.time()

    print('Total time was ' + str((end_time-start_time)/60) + ' minutes.')