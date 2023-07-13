# Created by: Gary Chen
# Date: July 11, 2023
# Purpose: evenly split preprocessed sinogramsm and ground truth to into a valid and a train group; each group contains num_ranks ranks
# input directories to the preprocessed sinograms and ground truths
# Process: 1) remove the outlier files and the duplicates and 2) split the cleaned up files into proper folders

import numpy as np
import os
import argparse
import glob
import shutil

def main():
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('--dir', help='base directory where ground truth and sinogram npy are')
    parser.add_argument('--dest', help='dest directory where split data are')
    parser.add_argument('--num_ranks', type=int, help='number of ranks', default=4)
    parser.add_argument('--train', type=float, help='train ratio', default=0.8)
    parser.add_argument('--valid', type=float, help='valid ratio', default=0.2)
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

    # Identify missing files in the sinogram set and the ground truth set
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
    
    num_ranks = args.num_ranks
    train = args.train
    valid = args.valid

    # Create all the necessary directories to store the splitted arrays
    train_dir = os.path.join(dest_dir, 'train')
    valid_dir = os.path.join(dest_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    for i in range(num_ranks):
        train_rank_dir = os.path.join(train_dir, f'{i}')
        valid_rank_dir = os.path.join(valid_dir, f'{i}')
        os.makedirs(train_rank_dir, exist_ok=True)
        os.makedirs(valid_rank_dir, exist_ok=True)

    # Splitting the sinograms and gt into train and valid group based on 8:2 ratio
    # 382 ground truths and 382 sinograms into 4 files (sino_train, sino_valid, gt_train, and gt_valid)
    # train set has 306 ground truths and 306 sinograms
    # valid set has 76 ground truths and 76 sinograms

    cleaned_file_num = len(sinogram_file_dirs)
    idx = int(cleaned_file_num*train)
    sino_train, sino_valid = np.split(sinogram_file_dirs, [idx])
    gt_train, gt_valid = np.split(ground_truth_file_dirs, [idx])

    train_group_size = len(sino_train)//num_ranks
    valid_group_size = len(sino_valid)//num_ranks

    sino_train = sino_train[:train_group_size*num_ranks]
    sino_valid = sino_valid[:valid_group_size*num_ranks]
    gt_train = gt_train[:train_group_size*num_ranks]
    gt_valid = gt_valid[:valid_group_size*num_ranks]

    sino_train_groups = np.split(sino_train,num_ranks)
    gt__train_groups = np.split(gt_train,num_ranks)
    sino_valid_groups = np.split(sino_valid,num_ranks)
    gt_valid_groups = np.split(gt_valid,num_ranks)

    # Copy and paste the .npy files to from source dir to the appropriate dir
    for i in range(num_ranks):
        for src_file in sino_train_groups[i]:
            target_dir = os.path.join(dest_dir, 'train', f'{i}')
            shutil.copy(src_file, target_dir)
        for src_file in gt__train_groups[i]:
            target_dir = os.path.join(dest_dir, 'train', f'{i}')
            shutil.copy(src_file, target_dir)
        for src_file in sino_valid_groups[i]:
            target_dir = os.path.join(dest_dir, 'valid', f'{i}')
            shutil.copy(src_file, target_dir)
        for src_file in gt_valid_groups[i]:
            target_dir = os.path.join(dest_dir, 'valid', f'{i}')
            shutil.copy(src_file, target_dir)

    print('Successfully moved all the pre-processed .npy files')

if __name__ == "__main__":
    main()