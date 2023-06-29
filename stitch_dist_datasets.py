"""Stitch datasets generated from different ranks together"""

import argparse

# get number of ranks from command line
parser = argparse.ArgumentParser()
parser.add_argument('--num_ranks', type=int, default=1)
parser.add_argument('--dataset_type', type=str, default='train')
parser.add_argument('--img_type', type=str, default='foam')

args = parser.parse_args()
num_ranks = args.num_ranks
img_type = args.img_type
dataset_type = args.dataset_type

import numpy as np

dataset_folder = 'dataset_' + img_type

ground_truth = []
masks = []
sparse_sinograms = []
reconstructions = []
num_proj_pix = []
sinograms = []
theta = []
x_size = []
y_size = []

for rank in range(num_ranks):
    ground_truth.append(np.load(img_type + '_' + dataset_type + '_' + str(rank) + '.npy'))
    masks.append(np.load(dataset_folder + '/' + dataset_type + '_masks' + '_' + str(rank) + '.npy'))
    sparse_sinograms.append(np.load(dataset_folder + '/'+ dataset_type + '_sparse_sinograms' + '_' + str(rank) + '.npy'))
    reconstructions.append(np.load(dataset_folder + '/'+ dataset_type + '_reconstructions' + '_' + str(rank) + '.npy'))
    num_proj_pix.append(np.load(dataset_folder + '/'+ dataset_type + '_num_proj_pix' + '_' + str(rank) + '.npy'))
    sinograms.append(np.load(dataset_folder + '/'+ dataset_type + '_sinograms' + '_' + str(rank) + '.npy'))
    theta.append(np.load(dataset_folder + '/'+ dataset_type + '_theta' + '_' + str(rank) + '.npy'))
    x_size.append(np.load(dataset_folder + '/'+ dataset_type + '_x_size' + '_' + str(rank) + '.npy'))
    y_size.append(np.load(dataset_folder + '/'+ dataset_type + '_y_size' + '_' + str(rank) + '.npy'))

ground_truth = np.concatenate(ground_truth,axis=0)
masks = np.concatenate(masks,axis=0)
sparse_sinograms = np.concatenate(sparse_sinograms,axis=0)
reconstructions = np.concatenate(reconstructions,axis=0)
sinograms = np.concatenate(sinograms,axis=0)

num_proj_pix = np.array(num_proj_pix)
theta = np.array(theta)
x_size = np.array(x_size)
y_size = np.array(y_size)

assert np.all(num_proj_pix == num_proj_pix[0])
assert np.all(theta == theta[0])
assert np.all(x_size == x_size[0])
assert np.all(y_size == y_size[0])

np.save(str(img_type)  + '_' + str(dataset_type) + '.npy',ground_truth)
np.save(dataset_folder + '/' + str(dataset_type) + '_masks.npy',masks)
np.save(dataset_folder + '/' + str(dataset_type) + '_sparse_sinograms.npy',sparse_sinograms)
np.save(dataset_folder + '/' + str(dataset_type) + '_reconstructions.npy',reconstructions)
np.save(dataset_folder + '/' + str(dataset_type) + '_sinograms.npy',sinograms)

np.save(dataset_folder + '/' + str(dataset_type) + '_num_proj_pix.npy',num_proj_pix[0])
np.save(dataset_folder + '/' + str(dataset_type) + '_theta.npy',theta[0])
np.save(dataset_folder + '/' + str(dataset_type) + '_x_size.npy',x_size[0])
np.save(dataset_folder + '/' + str(dataset_type) + '_y_size.npy',y_size[0])
