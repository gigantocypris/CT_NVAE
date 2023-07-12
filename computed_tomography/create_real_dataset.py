"""
Process sinograms from the real data into a dataset for CT_NVAE training.
"""

import argparse
import numpy as np
from utils import process_sinogram, create_sinogram, get_images, create_folder, create_sparse_dataset
import time
import glob
from mpi4py import MPI

def main(rank):
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('--dir', help='base directory containing .npy files of the ground truth, sinograms, and theta')
    parser.add_argument('-n',dest='num_truncate', type=int, help='number of 3D examples', default=2)
    parser.add_argument('-d',dest='dataset_type', type=str, help='dataset type, either train or valid')
    parser.add_argument('--pnm', dest='pnm', type=float, help='poisson noise multiplier, higher value means higher SNR', default=1e3)
    args = parser.parse_args()

    ### INPUT ###
    sub_dir = args.dir + '/' + args.dataset_type + '/' + str(rank)
    theta = np.load(args.dir + '/theta.npy') # projection angles
    truncate_dataset = args.num_truncate
    num_sparse_angles = 10 # number of angles to image per sample (dose remains the same)
    random = True # If True, randomly pick angles
    #############
    x_train_imgs = [] # num_images x x_size x y_size
    x_train_sinograms = [] # num_images x num_angles x num_proj_pix
    all_mask_inds = []
    all_reconstructed_objects = []
    all_sparse_sinograms = []

    sinogram_files = np.sort(glob.glob(sub_dir + '/*_sinogram.npy'))
    print(f'sub_dir is {sub_dir}')
    #print(f'sinogram_files has {sinogram_files}')

    for i in range(truncate_dataset):
        filepath_sino = sinogram_files[i] # sinogram filepath
        print(f'filepath_sino is {filepath_sino}')
        filepath_gt = sinogram_files[i][:-13] + '.npy' # ground truth filepath
        print(f'filepath_gt is {filepath_gt}')
        x_train = np.load(filepath_gt)
        x_train_sinogram = np.load(filepath_sino)
        print(f'x_train_sinogram has shape {x_train_sinogram.shape}')

        # make sparse sinogram and reconstruct
        sparse_angles, reconstruction, sparse_sinogram = process_sinogram(np.transpose(x_train_sinogram,axes=[1,0,2]), random, num_sparse_angles, theta, 
                                                                          poisson_noise_multiplier = args.pnm, 
                                                                          remove_ring_artifact = False)

        # append to lists

        x_train_imgs.append(x_train)
        x_train_sinograms.append(x_train_sinogram)
        all_mask_inds.append(np.repeat(np.expand_dims(sparse_angles,axis=0),x_train.shape[0],axis=0))
        all_reconstructed_objects.append(reconstruction)
        all_sparse_sinograms.append(np.transpose(sparse_sinogram,axes=[1,0,2]))


    x_train_imgs = np.concatenate(x_train_imgs, axis=0)
    x_train_sinograms = np.concatenate(x_train_sinograms, axis=0)
    all_mask_inds = np.concatenate(all_mask_inds, axis=0)
    all_reconstructed_objects = np.concatenate(all_reconstructed_objects, axis=0)
    all_sparse_sinograms = np.concatenate(all_sparse_sinograms, axis=0)


    num_proj_pix = x_train_sinograms.shape[-1]
    
    np.save(args.dir + '/' + args.dataset_type + '_sinograms_' + str(rank) + '.npy', x_train_sinograms)
    np.save(args.dir + '/' + args.dataset_type + '_ground_truth_' + str(rank) + '.npy', x_train_imgs)
    np.save(args.dir + '/' + args.dataset_type + '_theta_' + str(rank) + '.npy', theta)
    np.save(args.dir + '/' + args.dataset_type + '_num_proj_pix_' + str(rank) + '.npy', num_proj_pix)

    np.save(args.dir + '/' + args.dataset_type + '_x_size_' + str(rank) + '.npy', x_train_imgs.shape[1]) # size of original image
    np.save(args.dir + '/' + args.dataset_type + '_y_size_' + str(rank) + '.npy', x_train_imgs.shape[2]) # size of original image

    np.save(args.dir + '/' + args.dataset_type + '_masks_' + str(rank) + '.npy', all_mask_inds)
    np.save(args.dir + '/' + args.dataset_type + '_reconstructions_' + str(rank) + '.npy', all_reconstructed_objects)
    np.save(args.dir + '/' + args.dataset_type + '_sparse_sinograms_' + str(rank) + '.npy', all_sparse_sinograms)

    print("Shape of sinograms: ", x_train_sinograms.shape)
    print("Shape of original training images: ", x_train_imgs.shape)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD

    # check rank
    rank = comm.rank
    print('Hello from rank: ' + str(rank))

    start_time = time.time()
    main(rank)
    end_time = time.time()
    print('Total time was ' + str((end_time-start_time)/60) + ' minutes.')