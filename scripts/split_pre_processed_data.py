# Created by: Gary Chen
# Date: July 13, 2023
# Purpose: evenly split preprocessed sinogramsm and ground truth to into train, valid, and test groups in a specfic ratio
# input directories to the preprocessed sinograms and ground truths
# Process: 1) remove the outlier files and the duplicates and 2) split the cleaned up files into proper folders
# output: files are split into the desired directory strucutre under the specific target directory

import numpy as np
import os
import argparse
import glob
import shutil

def clean_up_preprocessed(sinogram_file_dirs, ground_truth_file_dirs):
    # Remove files with unusual file names
    sino_filename_length = len(sinogram_file_dirs[0])
    gt_filename_length = len(ground_truth_file_dirs[0])
    sinogram_file_dirs = np.array([filename for filename in sinogram_file_dirs if len(filename) == sino_filename_length])
    ground_truth_file_dirs = np.array([filename for filename in ground_truth_file_dirs if len(filename) == gt_filename_length])
    assert len(sinogram_file_dirs) > 50, "Something wrong about loading the sinograms"
    assert len(ground_truth_file_dirs) > 50, "Something wrong about loading the ground truth"

    # Identify missing files in the sinogram set and the groud truth set
    sinogram_ids = np.array([filename[-17:-13] for filename in sinogram_file_dirs if len(filename) == sino_filename_length])
    ground_truth_ids = np.array([filename[-8:-4] for filename in ground_truth_file_dirs if len(filename) == gt_filename_length])
    sino_set = set(sinogram_ids)
    gt_set = set(ground_truth_ids)

    # Ensure there is no duplicates
    assert len(list(sino_set)) == len(sinogram_ids), 'Duplicate sinograms are not cleaned successfully'
    assert len(list(gt_set)) == len(ground_truth_ids), 'Duplicate ground truths are not cleaned successfully'

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
    return sinogram_file_dirs, ground_truth_file_dirs

def create_dirs(dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    train_dir = os.path.join(dest_dir, 'train')
    valid_dir = os.path.join(dest_dir, 'valid')
    test_dir = os.path.join(dest_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

def split_data(file_dirs,train,valid,test):
    # Splitting the data into train, valid, and train
    cleaned_file_num = len(file_dirs)
    ratio = np.array([train,valid,test])
    assert np.ceil(np.sum(ratio)) == 1

    cum_ratio = np.cumsum(ratio)
    split_indices = cum_ratio[:-1] * cleaned_file_num
    split_indices = split_indices.astype(int)
    train, valid, test = np.split(file_dirs,split_indices)
    if len(train)+len(valid)+len(test)==cleaned_file_num:
        print(f'data is successfully split')
    else:
        print(f'failure in splitting')
    return train, valid, test

def copy_paste_files(file_dirs, dest_dir, data_type):
    for src_file in file_dirs:
        target_dir = os.path.join(dest_dir, data_type)
        shutil.copy(src_file, target_dir)

def main():
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('--dir', help='base directory where ground truth and sinogram npy are')
    parser.add_argument('--dest', help='dest directory where split data are')
    parser.add_argument('--train', type=float, help='train ratio', default=0.7)
    parser.add_argument('--valid', type=float, help='valid ratio', default=0.2)
    parser.add_argument('--test', type=float, help='test ratio', default=0.1)
    args = parser.parse_args()

    # Load ground truth, sinograms, and hyperparameters
    ground_truth_dir = os.path.join(args.dir, 'input_npy')
    sinogram_dir = os.path.join(args.dir, 'sinogram_npy')
    dest_dir = args.dest
    train = args.train
    valid = args.valid
    test = args.test
    sinogram_file_dirs = np.sort(glob.glob(sinogram_dir + '/*_sinogram.npy'))
    ground_truth_file_dirs = np.sort(glob.glob(ground_truth_dir + '/*.npy'))

    # Clean up the preproccessed data
    sinogram_file_dirs, ground_truth_file_dirs = clean_up_preprocessed(sinogram_file_dirs, ground_truth_file_dirs)
    
    # Create all the necessary directories to store the split arrays
    create_dirs(dest_dir)

    # Split the cleanup data into train, valid, and test
    sino_train, sino_valid, sino_test = split_data(sinogram_file_dirs,train,valid,test)
    gt_train, gt_valid, gt_test = split_data(ground_truth_file_dirs,train,valid,test)

    # Copy and paste the .npy files to from source dir to the appropriate dir
    copy_paste_files(sino_train, dest_dir, 'train')
    copy_paste_files(sino_valid, dest_dir, 'valid')
    copy_paste_files(sino_test, dest_dir, 'test')
    copy_paste_files(gt_train, dest_dir, 'train')
    copy_paste_files(gt_valid, dest_dir, 'valid')
    copy_paste_files(gt_test, dest_dir, 'test')
    print(f'Successfully split and moved all the files')

if __name__ == "__main__":
    main()