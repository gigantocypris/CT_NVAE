"""
Process sinograms into a dataset for CT_NVAE training.
"""

import argparse
import numpy as np
from computed_tomography.utils import process_sinogram
import time
import glob

def main(args, dataset_type):

    ### INPUT ###
    sub_dir = args.dir + '/' + dataset_type
    theta = np.load(args.dir + '/theta.npy') # projection angles
    num_sparse_angles = args.num_sparse_angles # number of angles to image per sample (dose remains the same)
    random = args.random # If True, randomly pick angles
    #############
    x_train_imgs = [] # num_images x x_size x y_size
    x_train_sinograms = [] # num_images x num_angles x num_proj_pix
    all_mask_inds = []
    all_reconstructed_objects = []
    all_sparse_sinograms = []
    all_3d_object_ids = []
    all_sparse_sinograms_raw = []

    print(f'sub_dir is {sub_dir}')

    sinogram_files = np.sort(glob.glob(sub_dir + '/*_sinogram.npy'))
    print(f'Total number of sinograms found: {len(sinogram_files)}')

    for i in range(len(sinogram_files)):
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
                             poisson_noise_multiplier = args.pnm, remove_ring_artifact = False, ring_artifact_strength = args.ring_artifact_strength)

        # append to lists

        x_train_imgs.append(x_train)
        x_train_sinograms.append(x_train_sinogram)
        all_mask_inds.append(np.repeat(np.expand_dims(sparse_angles,axis=0),x_train.shape[0],axis=0))
        all_reconstructed_objects.append(reconstruction)
        all_sparse_sinograms.append(np.transpose(sparse_sinogram,axes=[1,0,2]))
        all_sparse_sinograms_raw.append(np.transpose(sparse_sinogram_raw,axes=[1,0,2]))
        all_3d_object_ids.append(np.repeat(np.expand_dims(np.sum(reconstruction,axis=0),axis=0),x_train.shape[0],axis=0))

    x_train_imgs = np.concatenate(x_train_imgs, axis=0)
    x_train_sinograms = np.concatenate(x_train_sinograms, axis=0)
    all_mask_inds = np.concatenate(all_mask_inds, axis=0)
    all_reconstructed_objects = np.concatenate(all_reconstructed_objects, axis=0)
    all_sparse_sinograms = np.concatenate(all_sparse_sinograms, axis=0)
    all_sparse_sinograms_raw = np.concatenate(all_sparse_sinograms_raw, axis=0)
    all_3d_object_ids = np.concatenate(all_3d_object_ids, axis=0)

    num_proj_pix = x_train_sinograms.shape[-1]
    
    np.save(args.dir + '/' + dataset_type + '_sinograms.npy', x_train_sinograms)
    np.save(args.dir + '/' + dataset_type + '_ground_truth.npy', x_train_imgs)
    np.save(args.dir + '/' + dataset_type + '_theta.npy', theta)
    np.save(args.dir + '/' + dataset_type + '_num_proj_pix.npy', num_proj_pix)

    np.save(args.dir + '/' + dataset_type + '_x_size.npy', x_train_imgs.shape[1]) # size of original image
    np.save(args.dir + '/' + dataset_type + '_y_size.npy', x_train_imgs.shape[2]) # size of original image

    np.save(args.dir + '/' + dataset_type + '_masks.npy', all_mask_inds)
    np.save(args.dir + '/' + dataset_type + '_reconstructions.npy', all_reconstructed_objects)
    np.save(args.dir + '/' + dataset_type + '_sparse_sinograms.npy', all_sparse_sinograms)
    np.save(args.dir + '/' + dataset_type + '_sparse_sinograms_raw.npy', all_sparse_sinograms_raw)
    np.save(args.dir + '/' + dataset_type + '_3d_object_ids.npy', all_3d_object_ids)

    print("Shape of sinograms: ", x_train_sinograms.shape)
    print("Shape of original training images: ", x_train_imgs.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('--dir', dest='dir', help='directory containing splits of ground truth/sinograms and theta')
    parser.add_argument('--pnm', dest='pnm', type=float, help='poisson noise multiplier, higher value means higher SNR', default=1e3)
    parser.add_argument('--sparse', dest='num_sparse_angles', type=int, help='number of angles to image per sample (dose remains the same)', default=10)
    parser.add_argument('--random', dest='random', type=bool, help='If True, randomly pick angles', default=True)
    parser.add_argument('--ring', dest='ring_artifact_strength', type=float, help='if >0, add ring artifact to sinograms', default=0.0)
    args = parser.parse_args()


    start_time = time.time()
    main(args, 'train')
    main(args, 'valid')
    main(args, 'test')
    end_time = time.time()

    print('Total time was ' + str((end_time-start_time)/60) + ' minutes.')