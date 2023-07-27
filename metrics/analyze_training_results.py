import os
import argparse
import numpy as np
from metrics.utils import analyze_single_slice, visualize_single_slice

parser = argparse.ArgumentParser(description='Get command line args')
parser.add_argument('--dataset', type=str, default='foam_ring',
                    help='dataset name')
parser.add_argument('--expr_id', type=str, default='test_0000_foam2_ring',
                    help='experiment id')
parser.add_argument('--rank', type=int, default=0,
                    help='rank')
parser.add_argument('--original_size', type=int, default=128,
                    help='original size')
parser.add_argument('--algorithm', type=str, default='gridrec', choices=['gridrec', 'sirt', 'tv'],
                    help='algorithm')
parser.add_argument('--checkpoint_dir', type=str, default='checkpts',
                    help='checkpoint directory')
parser.add_argument('--dataset_type', type=str, default='valid',
                    help='dataset type')
args = parser.parse_args()

# get working_dir from the environment variable
working_dir = os.environ['WORKING_DIR']

results_path = working_dir + '/' + args.checkpoint_dir + '/eval-' + args.expr_id
dataset_path = working_dir + '/dataset_' + args.dataset

# Load the final validation *.npy files
final_phantoms = np.load(results_path + '/final_phantoms_' + args.dataset_type + '_rank_' + str(args.rank) + '.npy')
ground_truths = np.load(results_path + '/final_ground_truth_' + args.dataset_type + '_rank_' + str(args.rank) + '.npy')
sparse_sinograms = np.load(results_path + '/final_sparse_sinograms_' + args.dataset_type + '_rank_' + str(args.rank) + '.npy')
theta = np.load(results_path + '/final_theta_' + args.dataset_type + '_rank_' + str(args.rank) + '.npy')

num_z_slices = final_phantoms.shape[0]

all_err_vec = []
all_err_vec_ring = []
all_err_vec_NVAE = []

for img_ind in range(num_z_slices):
    ground_truth = ground_truths[img_ind]
    sparse_sinogram = np.expand_dims(sparse_sinograms[img_ind], axis=1)
    theta_0 = theta[img_ind]
    final_phantom = final_phantoms[img_ind]

    ground_truth, reconstruct, reconstruct_ring, final_phantom, err_vec, err_vec_ring, err_vec_NVAE = \
        analyze_single_slice(sparse_sinogram, theta_0, ground_truth, final_phantom, args.original_size, args.algorithm, verbose=True)
    all_err_vec.append(err_vec)
    all_err_vec_ring.append(err_vec_ring)
    all_err_vec_NVAE.append(err_vec_NVAE)


np.save(results_path + '/err_vec_' + args.dataset_type + '_rank_' + str(args.rank) + '.npy', all_err_vec)
np.save(results_path + '/err_vec_ring_' + args.dataset_type + '_rank_' + str(args.rank) + '.npy', all_err_vec_ring)
np.save(results_path + '/err_vec_NVAE_' + args.dataset_type + '_rank_' + str(args.rank) + '.npy', all_err_vec_NVAE)
visualize_single_slice(ground_truth, reconstruct, reconstruct_ring, final_phantom, 
                           results_path, args.dataset_type, args.rank, img_ind)