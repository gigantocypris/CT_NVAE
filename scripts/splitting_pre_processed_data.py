# Created by: Gary Chen
# Date: July 11, 2023
# Purpose: evenly split preprocessed sinogramsm and ground truth to into a test and a train group; each group contains 4 ranks
# input directories to the preprocessed sinograms and ground truths
# Process: 1) remove the outlier files and the duplicates and 2) split the cleaned up files into proper folders

import numpy as np
import os
import argparse
import glob
import shutil

def main():
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('--dir', help='base directory where ground truth and sinogram npy stay')
    parser.add_argument('--dest', help='dest directory where split data stay')
    parser.add_argument('--rank_num', type=int, help='number of ranks', default=4)
    parser.add_argument('--train', type=float, help='train ratio', default=0.8)
    parser.add_argument('--test', type=float, help='test ratio', default=0.2)
    args = parser.parse_args()

    # Load ground truth, sinograms, and hyperparameters
    ground_truth_dir = os.path.join(args.dir, 'input_npy')
    sinogram_dir = os.path.join(args.dir, 'sinogram_npy')
    dest_dir = args.dest
    sinogram_file_dirs = np.sort(glob.glob(sinogram_dir + '/*_sinogram.npy'))
    ground_truth_file_dirs = np.sort(glob.glob(ground_truth_dir + '/*.npy'))
    assert len(sinogram_file_dirs[0]) == 94
    assert len(ground_truth_file_dirs[0]) == 82

    # Remove files with unusual file names
    sinogram_file_dirs = np.array([filename for filename in sinogram_file_dirs if len(filename) == 94])
    ground_truth_file_dirs = np.array([filename for filename in ground_truth_file_dirs if len(filename) == 82])

    # Identify missing files in the sinogram set and the groud truth set
    sinogram_ids = np.array([filename[-17:-13] for filename in sinogram_file_dirs if len(filename) == 94])
    ground_truth_ids = np.array([filename[-8:-4] for filename in ground_truth_file_dirs if len(filename) == 82])
    sino_set = set(sinogram_ids)
    gt_set = set(ground_truth_ids)

    # Ensure there is no duplicates
    assert len(list(sino_set)) == len(sinogram_ids)
    assert len(list(gt_set)) == len(ground_truth_ids)

    # Get the differences
    diff1 = sino_set - gt_set  # Elements in set1 but not in set2
    diff2 = gt_set - sino_set  # Elements in set2 but not in set1
    diff1 = np.array(list(diff1))
    diff2 = np.array(list(diff2))
    print("File id in sino_set but not in gt_set:", diff1)
    print("File id in gt_set but not in sino_set:", diff2)

    # Remove the differences
    assert len(sinogram_file_dirs) == len(list(sino_set))
    assert len(ground_truth_file_dirs) == len(list(gt_set))
    for diff in diff1:
        mask = np.vectorize(lambda sinogram_file_dirs: diff not in sinogram_file_dirs)(sinogram_file_dirs)
        sinogram_file_dirs = sinogram_file_dirs[mask]
    for diff in diff2:
        mask = np.vectorize(lambda ground_truth_file_dirs: diff not in ground_truth_file_dirs)(ground_truth_file_dirs)
        ground_truth_file_dirs = ground_truth_file_dirs[mask]
    
    print('Successfully cleanup the pre-processed data')
    
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

    # Splitting the sinograms and gt into train and test group based on 8:2 ratio
    # 382 ground truths and 382 sinograms into 4 files (sino_train, sino_test, gt_train, and gt_test)
    # train set has 306 ground truths and 306 sinograms
    # test set has 76 ground truths and 76 sinograms

    cleaned_file_num = len(sinogram_file_dirs)
    idx = int(cleaned_file_num*train)
    sino_train, sino_test = np.split(sinogram_file_dirs, [idx])
    gt_train, gt_test = np.split(ground_truth_file_dirs, [idx])

    train_group_size = len(sino_train)//rank_num
    test_group_size = len(sino_test)//rank_num

    sino_train = sino_train[:train_group_size*rank_num]
    sino_test = sino_test[:test_group_size*rank_num]
    gt_train = gt_train[:train_group_size*rank_num]
    gt_test = gt_test[:test_group_size*rank_num]

    sino_train_groups = np.split(sino_train,rank_num)
    gt__train_groups = np.split(gt_train,rank_num)
    sino_test_groups = np.split(sino_test,rank_num)
    gt_test_groups = np.split(gt_test,rank_num)

    # Copy and paste the .npy files to from source dir to the appropriate dir
    for i in range(4):
        for src_file in sino_train_groups[i]:
            target_dir = os.path.join(dest_dir, 'train', f'{i}')
            shutil.copy(src_file, target_dir)
        for src_file in gt__train_groups[i]:
            target_dir = os.path.join(dest_dir, 'train', f'{i}')
            shutil.copy(src_file, target_dir)
        for src_file in sino_test_groups[i]:
            target_dir = os.path.join(dest_dir, 'test', f'{i}')
            shutil.copy(src_file, target_dir)
        for src_file in gt_test_groups[i]:
            target_dir = os.path.join(dest_dir, 'test', f'{i}')
            shutil.copy(src_file, target_dir)

    print('Successfully moved all the pre-processed .npy files')

main()