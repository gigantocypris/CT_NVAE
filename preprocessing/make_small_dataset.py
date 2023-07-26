# Purpose: Create a small dataset for testing pipeline
# Usage: python make_small_dataset.py <input_dir> <average_num_slice> <total_slice> <output_dir> 
# Example: python preprocessing/make_small_dataset.py /global/cfs/cdirs/m3562/users/hkim/brain_data/npy_organized_by_patient 25 750 /pscratch/sd/h/hojunek/output_brain/small_750

import shutil
import os
import random
from tqdm import tqdm
import argparse
import numpy as np
import random

def create_small_dataset(input_dir, average_num_slice, total_slice, output_dir):
    # Get a list of all .npy files in the input directory
    all_filenames = [file for file in os.listdir(input_dir) if file.endswith('.npy')]

    # Initialize a list to store selected filenames
    filenames_to_copy = []
    accumulated_slices = 0
    window = 5

    # Select files that are within the window span - Takes about 15 minutes
    filenames = []
    for file in tqdm(all_filenames, desc="Selecting files within +-5 window"):
        data_shape = np.load(os.path.join(input_dir, file)).shape[0]
        if average_num_slice - window <= data_shape <= average_num_slice + window:
            filenames.append(file)
    print("Number of files within +-5 window: ", len(filenames))

    # Increase window size until the condition is satisfied
    pbar = tqdm(total=total_slice, desc="Selecting files with larger window")
    while len(filenames) * (average_num_slice - window) < total_slice:
        window += 10
        print("Increasing window size to ", window)
        filenames = []
        for file in all_filenames:
            data_shape = np.load(os.path.join(input_dir, file)).shape[0]
            if average_num_slice - window <= data_shape <= average_num_slice + window:
                filenames.append(file)
        print("Number of files within +_", window, " window: ", len(filenames))
    pbar.close()

    pbar = tqdm(total=total_slice, desc="Accumulating slices")
    # Randomly select files until accumulated slices reach total_slice
    while accumulated_slices < total_slice:
        filename = random.choice(filenames)
        data = np.load(os.path.join(input_dir, filename))
        num_slices = data.shape[0]

        accumulated_slices += num_slices
        pbar.update(num_slices)
        filenames_to_copy.append(filename)
        filenames.remove(filename)  # To prevent re-selecting the same file
    pbar.close()

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

    create_small_dataset(args.input_dir, args.average_num_slice, args.total_slice, args.output_dir)

if __name__ == '__main__':
    main()
