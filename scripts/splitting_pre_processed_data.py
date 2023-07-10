# Created by: Gary Chen
# Date: July 10, 2023
# Purpose: split 650 preprocessed sinograms to into 4 ranks and each rank contains a test and a train group
# output: split 650 sinograms into 4 groups of 162, 162, 162, 164.
# for the group with 162 sinograms, they are split into 130 train sinograms and 32 test sinograms.
# for the group with 164 sinograms, they are split into 131 train sinograms and 33 test sinograms.
import numpy as np
import os
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('--dir', help='base directory where ground truth and sinogram npy stay')
    parser.add_argument('--dest', help='dest directory where split data stay')
    parser.add_argument('--rank_num', type=int, help='number of ranks', default=4)
    parser.add_argument('--train', type=float, help='train ratio', default=0.8)
    parser.add_argument('--test', type=float, help='test ratio', default=0.2)
    args = parser.parse_args()

    # load ground truth, sinograms, and hyperparameters
    ground_truth_dir = os.path.join(args.dir, 'input_npy')
    sinogram_dir = os.path.join(args.dir, 'sinogram_npy')
    dest_dir = args.dest
    #ground_truth_npys = [filename for filename in os.listdir(ground_truth_dir) if filename.endswith('.npy')]
    #sinogram_npys = [filename for filename in os.listdir(sinogram_dir) if filename.endswith('.npy')]
    sinogram_files = np.sort(glob.glob(sinogram_dir + '/*_sinogram.npy'))
    ground_truth_files = np.sort(glob.glob(ground_truth_dir + '/*.npy'))
    print(f'sino with glob has {len(sinogram_files)}')
    print(sinogram_files.shape)
    print(f'grouth with glob has {len(ground_truth_files)}')
    print(ground_truth_files.shape)

    rank_num = args.rank_num
    assert rank_num == 4
    train = args.train
    test = args.test
    assert train == 0.8
    assert test == 0.2

    # Create all the necessary directories to store the splitted arrays
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for i in range(rank_num):
        train_rank_dir = os.path.join(train_dir, f'{i}')
        test_rank_dir = os.path.join(test_dir, f'{i}')
        os.makedirs(train_rank_dir, exist_ok=True)
        os.makedirs(test_rank_dir, exist_ok=True)

    # Splitting the array into 4 groups
    sino_groups = np.split(sinogram_files,rank_num)
    gt_groups = np.split(ground_truth_files,rank_num)

    idx = int(len(sino_groups)/rank_num*train)
    train, test = np.split(sino_groups, [idx])
    for i, group in enumerate(sino_groups):
        pass
        


main()