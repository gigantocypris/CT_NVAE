"""
Preprocess images into sinograms
"""

import argparse
import numpy as np
from utils import create_sinogram, get_images, create_folder, create_sparse_dataset
import time
from mpi4py import MPI

def main(rank):
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('-n',dest='num_truncate', type=int, help='number of 2D examples', default=64)
    parser.add_argument('-d',dest='dataset_type', type=str, help='dataset type, either train or valid')
    parser.add_argument('--pnm', dest='pnm', type=float, help='poisson noise multiplier, higher value means higher SNR', default=1e3)
    args = parser.parse_args()

    ### INPUT ###

    theta = np.linspace(0, np.pi, 180, endpoint=False) # projection angles
    img_type = 'foam' # 'mnist' or 'foam'
    pad = True
    truncate_dataset = args.num_truncate
    num_sparse_angles = 10 # number of angles to image per sample (dose remains the same)
    random = True # If True, randomly pick angles
    #############
    
    save_path = 'dataset_' + img_type
    create_folder(save_path)
    
    # pull images that are normalized from 0 to 1
    # 0th dimension should be the batch dimension
    # 1st and 2nd dimensions should be spatial x and y coords, respectively
    x_train_imgs = get_images(rank, img_type = img_type, dataset_type=args.dataset_type)
    x_train_imgs = x_train_imgs[0:truncate_dataset]
    
    # Create sinograms all at once
    x_train_sinograms = create_sinogram(x_train_imgs, theta, pad=pad) # shape is truncate_dataset x num_angles x num_proj_pix
   

    num_proj_pix = x_train_sinograms.shape[-1]
    
    x_train_sinograms[x_train_sinograms<0]=0
    
    np.save(save_path + '/' + args.dataset_type + '_sinograms_' + str(rank) + '.npy', x_train_sinograms)
    np.save(save_path + '/' + args.dataset_type + '_theta_' + str(rank) + '.npy', theta)
    np.save(save_path + '/' + args.dataset_type + '_num_proj_pix_' + str(rank) + '.npy', num_proj_pix)

    np.save(save_path + '/' + args.dataset_type + '_x_size_' + str(rank) + '.npy', x_train_imgs.shape[1]) # size of original image
    np.save(save_path + '/' + args.dataset_type + '_y_size_' + str(rank) + '.npy', x_train_imgs.shape[2]) # size of original image
    
    print("Shape of sinograms: ", x_train_sinograms.shape)
    print("Shape of original training images: ", x_train_imgs.shape)
    
    all_mask_inds, all_reconstructed_objects, all_sparse_sinograms = \
    create_sparse_dataset(x_train_sinograms, 
                          theta,
                          args.pnm, # poisson noise multiplier, higher value means higher SNR
                          num_sparse_angles = num_sparse_angles, # number of angles to image per sample (dose remains the same)
                          random = random, # If True, randomly pick angles
                         )

    np.save(save_path + '/' + args.dataset_type + '_masks_' + str(rank) + '.npy', all_mask_inds)
    np.save(save_path + '/' + args.dataset_type + '_reconstructions_' + str(rank) + '.npy', all_reconstructed_objects)
    np.save(save_path + '/' + args.dataset_type + '_sparse_sinograms_' + str(rank) + '.npy', all_sparse_sinograms)
    
if __name__ == '__main__':
    comm = MPI.COMM_WORLD

    # check rank
    rank = comm.rank
    print('Hello from rank: ' + str(rank))

    start_time = time.time()
    main(rank)
    end_time = time.time()
    print('Total time was ' + str((end_time-start_time)/60) + ' minutes.')