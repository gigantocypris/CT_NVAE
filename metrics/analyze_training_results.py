import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
from metrics.utils import analyze_single_slice, visualize_single_slice, visualize_phantom_samples

parser = argparse.ArgumentParser(description='Get command line args')
parser.add_argument('--expr_id', type=str,
                    help='experiment id')
parser.add_argument('--rank', type=int, default=0,
                    help='rank')
parser.add_argument('--original_size', type=int,
                    help='original size')
parser.add_argument('--algorithm', type=str, default='gridrec', choices=['gridrec', 'sirt', 'tv'],
                    help='algorithm')
parser.add_argument('--checkpoint_dir', type=str,
                    help='checkpoint directory')
parser.add_argument('--dataset_type', type=str, default='valid',
                    help='dataset type')
parser.add_argument('--epoch', type=int,
                    help='epoch to evaluate at')
args = parser.parse_args()

# get working_dir from the environment variable
working_dir = os.environ['WORKING_DIR']

results_path = args.checkpoint_dir + '/eval-' + args.expr_id

h5_filename = results_path + '/eval_dataset_' + args.dataset_type + '_epoch_' + str(args.epoch) + '_rank_' + str(args.rank) + '.h5'

with h5py.File(h5_filename, 'r') as h5_file:
    num_steps = len(h5_file)

    all_err_vec = []
    all_err_vec_ring = []
    all_err_vec_NVAE = []

    for step in range(num_steps):
        example = h5_file[f'example_{step}']
        final_phantom = example['phantom'][:]
        sparse_sinogram = example['sparse_sinogram'][:]
        ground_truth = example['ground_truth'][:]
        theta = example['theta'][:]

        for img_ind in range(final_phantom.shape[0]):
            ground_truth_i = ground_truth[img_ind]
            sparse_sinogram_i = np.expand_dims(sparse_sinogram[img_ind], axis=1)
            theta_i= theta[img_ind]
            final_phantom_i = final_phantom[img_ind]
            # visualize_phantom_samples(final_phantom_i, results_path)

            final_phantom_i = final_phantom_i[:,:,0] # 0th sample

            ground_truth_crop, final_phantom_crop, reconstruct, reconstruct_ring, err_vec, err_vec_ring, err_vec_NVAE = \
                analyze_single_slice(sparse_sinogram_i, theta_i, ground_truth_i, final_phantom_i, 
                                     args.original_size, args.algorithm, verbose=True)
            all_err_vec.append(err_vec)
            all_err_vec_ring.append(err_vec_ring)
            all_err_vec_NVAE.append(err_vec_NVAE)
            visualize_single_slice(ground_truth_crop, reconstruct, reconstruct_ring, final_phantom_crop, 
                                results_path, args.dataset_type, args.epoch, args.rank, step, img_ind)
            plt.close('all')

    np.save(results_path + '/err_vec_' + args.dataset_type + '_rank_' + str(args.rank) + '.npy', all_err_vec)
    np.save(results_path + '/err_vec_ring_' + args.dataset_type + '_rank_' + str(args.rank) + '.npy', all_err_vec_ring)
    np.save(results_path + '/err_vec_NVAE_' + args.dataset_type + '_rank_' + str(args.rank) + '.npy', all_err_vec_NVAE)
