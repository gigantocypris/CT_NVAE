"""
Split objects and corresponding sinograms into train, valid, and test sets.
Reads in the sinogram and object .npy files from the specified directory and saves to
a new directory.
Usage:
python create_splits.py --src <source_dir> --dest <dest_dir> --train <train_ratio> --valid <valid_ratio> --test <test_ratio>

Example for foam images:
cd $WORKING_DIR
python $CT_NVAE_PATH/scripts/create_splits.py --src images_foam --dest dataset_foam2 --train 0.7 --valid 0.2 --test 0.1 -n 64

Example for covid images:
cd $WORKING_DIR
python $CT_NVAE_PATH/scripts/create_splits.py --src images_covid --dest dataset_covid2 --train 0.7 --valid 0.2 --test 0.1 -n 64
"""

import numpy as np
import os
import argparse
import glob
import shutil

def create_dirs(dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    train_dir = os.path.join(dest_dir, 'train')
    valid_dir = os.path.join(dest_dir, 'valid')
    test_dir = os.path.join(dest_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

def split_data(filenames, train_fraction, valid_fraction, test_fraction):
    # Splitting the data into train, valid, and train
    num_files = len(filenames)
    ratio = np.array([train_fraction, valid_fraction, test_fraction])

    cum_ratio = np.cumsum(ratio)
    split_indices = cum_ratio[:-1] * num_files
    split_indices = split_indices.astype(int)
    train, valid, test = np.split(filenames,split_indices)
    return train, valid, test

def copy_paste_files(sinogram_filenames, dest_dir, data_type):
    for src_file in sinogram_filenames:
        target_dir = os.path.join(dest_dir, data_type)
        shutil.copy(src_file, target_dir)
        shutil.copy(''.join(src_file.split('_sinogram')), target_dir)

def main(args):
    # Normalize the ratios
    total = args.train_fraction + args.valid_fraction + args.test_fraction
    args.train_fraction = args.train_fraction / total
    args.valid_fraction = args.valid_fraction / total
    args.test_fraction = args.test_fraction / total

    # Load ground truth, sinograms, and hyperparameters
    sinogram_filenames = np.sort(glob.glob(args.source_dir + '/*_sinogram.npy'))
    sinogram_filenames = sinogram_filenames[0:args.num_truncate] # truncate the dataset
    # shuffle the filenames
    np.random.shuffle(sinogram_filenames)

    # Create all the necessary directories to store the split arrays
    create_dirs(args.dest_dir)

    # Split the files into train, valid, and test
    train_files, valid_files, test_files = split_data(sinogram_filenames, args.train_fraction, args.valid_fraction, args.test_fraction)

    # Copy and paste the .npy files to from source dir to the appropriate dir
    copy_paste_files(train_files, args.dest_dir, 'train')
    copy_paste_files(valid_files, args.dest_dir, 'valid')
    copy_paste_files(test_files, args.dest_dir, 'test')

    shutil.copy(args.source_dir + '/theta.npy', args.dest_dir)
    print(f'Successfully split and moved all the files')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('--src', dest = 'source_dir', help='source directory where ground truth and sinogram npy are')
    parser.add_argument('--dest', dest = 'dest_dir', help='dest directory where split data are')
    parser.add_argument('--train', dest = 'train_fraction', type=float, help='train fraction', default=0.7)
    parser.add_argument('--valid', dest = 'valid_fraction', type=float, help='valid fraction', default=0.2)
    parser.add_argument('--test', dest = 'test_fraction', type=float, help='test fraction', default=0.1)
    parser.add_argument('-n',dest='num_truncate', type=int, help='number of 2D examples')
    args = parser.parse_args()
    main(args)