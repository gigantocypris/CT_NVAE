"""
Preprocess images into sinograms
"""

import argparse
import numpy as np
from utils import create_sinogram, get_images, create_folder, create_sparse_dataset
import time

def main():
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('-n',dest='num_truncate', type=int, help='number of points', default=64)
    parser.add_argument('-d',dest='dataset_type', type=str, help='dataset type')
    parser.add_argument('--pnm', dest='pnm', type=float, help='poisson noise multiplier, higher value means higher SNR', default=1e3)
    args = parser.parse_args()

    ### INPUT ###

    theta = np.linspace(0, np.pi, 180, endpoint=False) # projection angles
    img_type = 'foam' # 'mnist' or 'foam'
    pad = True
    truncate_dataset = args.num_truncate
    num_sparse_angles = 10 # number of angles to image per sample (dose remains the same)
    random = False # If True, randomly pick angles
    #############
    
    save_path = 'dataset_' + img_type
    create_folder(save_path)
    
    # pull images that are normalized from 0 to 1
    # 0th dimension should be the batch dimension
    # 1st and 2nd dimensions should be spatial x and y coords, respectively
    x_train_imgs = get_images(img_type = img_type, dataset_type=args.dataset_type)
    x_train_imgs = x_train_imgs[0:truncate_dataset]
    
    # Create sinograms all at once
    x_train_sinograms = create_sinogram(x_train_imgs, theta, pad=pad) # shape is truncate_dataset x num_angles x num_proj_pix
   
    """
    # Create sinograms one at a time
    x_train_sinograms = []
    for b in range(x_train_imgs.shape[0]):
        print(b)
        img = x_train_imgs[b]
        img = np.expand_dims(img, axis=0)
        sinogram = create_sinogram(img, theta, pad=pad)
        x_train_sinograms.append(np.expand_dims(sinogram, axis=0))
    x_train_sinograms = np.concatenate(x_train_sinograms, axis=0) # shape is truncate_dataset x num_angles x num_proj_pix
    """

    num_proj_pix = x_train_sinograms.shape[-1]
    
    x_train_sinograms[x_train_sinograms<0]=0
    
    np.save(save_path + '/' + args.dataset_type + '_sinograms.npy', x_train_sinograms)
    np.save(save_path + '/' + args.dataset_type + '_theta.npy', theta)
    np.save(save_path + '/' + args.dataset_type + '_num_proj_pix.npy', num_proj_pix)

    np.save(save_path + '/' + args.dataset_type + '_x_size.npy', x_train_imgs.shape[1]) # size of original image
    np.save(save_path + '/' + args.dataset_type + '_y_size.npy', x_train_imgs.shape[2]) # size of original image
    
    print("Shape of sinograms: ", x_train_sinograms.shape)
    print("Shape of original training images: ", x_train_imgs.shape)
    
    all_mask_inds, all_reconstructed_objects, all_sparse_sinograms = \
    create_sparse_dataset(x_train_sinograms, 
                          theta,
                          args.pnm, # poisson noise multiplier, higher value means higher SNR
                          num_sparse_angles = num_sparse_angles, # number of angles to image per sample (dose remains the same)
                          random = random, # If True, randomly pick angles
                         )

    np.save(save_path + '/' + args.dataset_type + '_masks.npy', all_mask_inds)
    np.save(save_path + '/' + args.dataset_type + '_reconstructions.npy', all_reconstructed_objects)
    np.save(save_path + '/' + args.dataset_type + '_sparse_sinograms.npy', all_sparse_sinograms)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Total time was ' + str((end_time-start_time)/60) + ' minutes.')