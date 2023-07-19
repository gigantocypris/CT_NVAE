# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
#
# Modified July 18, 2023
# ---------------------------------------------------------------

"""Code for getting the data loaders."""

import numpy as np
from PIL import Image
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
import urllib
from lmdb_datasets import LMDBDataset
from thirdparty.lsun import LSUN

def get_loaders(args):
    """Get data loaders for required dataset."""
    return get_loaders_eval(args.dataset, args)

class CT_Dataset(Dataset):
    def __init__(self, ground_truth, reconstruction, sparse_sinogram, sparse_sinogram_raw,\
                 object_ids, mask, x_size, y_size, num_proj_pix, theta):
        self.sparse_recons = reconstruction[:,None,:,:]
        self.sparse_sino = sparse_sinogram
        self.sparse_sino_raw  = sparse_sinogram_raw
        self.object_ids = object_ids
        self.ground_truth = ground_truth
        self.masks = mask
        self.theta = theta
        self.x_size = x_size
        self.y_size = y_size
        self.num_proj_pix = num_proj_pix
        self.index = [i for i in range(len(self.sparse_recons))]

    def __getitem__(self, index):
        # d = self.transform(self.sparse_recons[index])
        sparse_reconstruction = torch.from_numpy(self.sparse_recons[index]).float()
        sparse_sinogram = torch.from_numpy(self.sparse_sino[index]).float()
        angles = torch.from_numpy(self.theta[self.masks[index]]).float()
        x_size = torch.from_numpy(self.x_size).float()
        y_size = torch.from_numpy(self.y_size).float()
        num_proj_pix = torch.from_numpy(self.num_proj_pix).float()
        ground_truth = torch.from_numpy(self.ground_truth[index]).float()
        sparse_sinogram_raw = torch.from_numpy(self.sparse_sino_raw[index]).float()
        object_id = torch.from_numpy(self.object_ids[index]).float()
        return (sparse_reconstruction, sparse_sinogram, sparse_sinogram_raw, object_id,
                angles, x_size, y_size, num_proj_pix, ground_truth)

    def __len__(self):
        return len(self.sparse_recons)

def load_data(dataset, dataset_dir, dataset_type='train'):
    ground_truth = np.load(dataset_dir + '/dataset_' + dataset + '/' + dataset_type + '_ground_truth.npy')
    reconstruction = np.load(dataset_dir + '/dataset_' + dataset + '/' + dataset_type + '_reconstructions.npy')
    sparse_sinogram = np.load(dataset_dir + '/dataset_' + dataset + '/' + dataset_type + '_sparse_sinograms.npy')
    sparse_sinogram_raw = np.load(dataset_dir + '/dataset_' + dataset + '/' + dataset_type + '_sparse_sinograms_raw.npy')
    object_ids = np.load(dataset_dir + '/dataset_' + dataset + '/' + dataset_type + '_3d_object_ids.npy')
    mask = np.load(dataset_dir + '/dataset_' + dataset + '/' + dataset_type + '_masks.npy')
    x_size = np.load(dataset_dir + '/dataset_' + dataset + '/' + dataset_type + '_x_size.npy')
    y_size = np.load(dataset_dir + '/dataset_' + dataset + '/' + dataset_type + '_y_size.npy')
    num_proj_pix = np.load(dataset_dir +  '/dataset_' + dataset + '/' + dataset_type + '_num_proj_pix.npy')
    theta = np.load(dataset_dir + '/dataset_' + dataset + '/' + dataset_type + '_theta.npy')
    return (ground_truth, reconstruction, sparse_sinogram, sparse_sinogram_raw,\
        object_ids, mask, x_size, y_size, num_proj_pix, theta)

def get_loaders_eval(dataset, args):
    """Get train and valid loaders for cifar10/tiny imagenet."""

    # train_transform, valid_transform = _data_transforms_foam(args)
    dataset_dir = os.environ['DATASET_DIR']

    train_data = CT_Dataset(*load_data(dataset, dataset_dir, dataset_type='train'))
    valid_data = CT_Dataset(*load_data(dataset, dataset_dir, dataset_type='valid'))

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=4, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=1, drop_last=False)

    return train_queue, valid_queue
