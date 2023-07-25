# Purpose: Create a small dataset for testing pipeline
# Usage: python make_small_dataset.py <input_dir> <average_num_slice> <total_slice> <output_dir>
# Example: python preprocessing/make_small_dataset.py /global/cfs/cdirs/m3562/users/hkim/brain_data/npy_organized_by_patient 25 1000 /pscratch/sd/h/hojunek/output_brain/dataset_small_1000

import shutil
import os
import random
from tqdm import tqdm
import argparse
import numpy as np

def copy_random_files(input_dir, average_num_slice, total_slice, output_dir):
    # Get a list of all .npy files in the input directory
    filenames = [file for file in os.listdir(input_dir) if file.endswith('.npy')]

    # Initialize a list to store selected filenames
    filenames_to_copy = []
    accumulated_slices = 0

    # Randomly select files until accumulated slices reach total_slice
    while accumulated_slices < total_slice:
        filename = random.choice(filenames)
        data = np.load(os.path.join(input_dir, filename))
        num_slices = data.shape[0]
        
        if average_num_slice - 5 <= num_slices <= average_num_slice + 5:
            accumulated_slices += num_slices
            filenames_to_copy.append(filename)
            filenames.remove(filename)  # To prevent re-selecting the same file

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Copy the selected files to the output directory
    for filename in tqdm(filenames_to_copy, desc="Copying files"):
        shutil.copy(os.path.join(input_dir, filename), output_dir)

def main():
    parser = argparse.ArgumentParser(description='Copy a random selection of .npy files from one directory to another based on their shape.')
    parser.add_argument('input_dir', type=str, help='The directory to copy files from.')
    parser.add_argument('average_num_slice', type=int, help='The average number of slices in selected .npy files.')
    parser.add_argument('total_slice', type=int, help='The total number of slices in all selected .npy files.')
    parser.add_argument('output_dir', type=str, help='The directory to copy files to.')
    
    args = parser.parse_args()

    copy_random_files(args.input_dir, args.average_num_slice, args.total_slice, args.output_dir)

if __name__ == '__main__':
    main()
