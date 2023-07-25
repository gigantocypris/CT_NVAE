# Purpose: Create a small dataset for testing pipeline
# Usage: python make_small_dataset.py <input_dir> <average_num_slice> <total_slice> <output_dir> <fast_ver>
# Example: python preprocessing/make_small_dataset.py /global/cfs/cdirs/m3562/users/hkim/brain_data/npy_organized_by_patient 25 1000 /pscratch/sd/h/hojunek/output_brain/dataset_small_1000 False

import shutil
import os
import random
from tqdm import tqdm
import argparse
import numpy as np
import random

def slow_create_small_dataset(input_dir, avg_num_slice, total_slice, output_dir):
    # Get a list of all .npy files in the input directory
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    # Initialize the accumulated number of slices and the slice window
    accum_slices = 0
    slice_window = 0
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize a progress bar
    pbar = tqdm(total=total_slice)

    # Loop until the total number of slices reaches the target
    while accum_slices < total_slice and filenames:
        # Shuffle the list of filenames to ensure a random selection
        random.shuffle(filenames)
        
        # Try to find a file with an acceptable number of slices
        for filename in filenames:
            # Load the file and get the number of slices
            data = np.load(os.path.join(input_dir, filename))
            num_slices = data.shape[0]
            
            # If the number of slices is within the window, copy the file and update accum_slices
            if avg_num_slice - slice_window <= num_slices <= avg_num_slice + slice_window:
                shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, filename))
                accum_slices += num_slices
                filenames.remove(filename)  # remove the selected file from the list

                # Update the progress bar
                pbar.update(num_slices)
                break
        else:
            # If no file with an acceptable number of slices was found, increase the window
            slice_window += 1

    # Close the progress bar
    pbar.close()


def fast_create_small_dataset(input_dir, average_num_slice, total_slice, output_dir):
    # Get a list of all .npy files in the input directory
    filenames = [file for file in os.listdir(input_dir) if file.endswith('.npy')]

    # Initialize a list to store selected filenames
    filenames_to_copy = []
    accumulated_slices = 0
    window = 5

    # Randomly select files until accumulated slices reach total_slice
    while accumulated_slices < total_slice:
        if not filenames:  # If we exhausted the list, extend the window and reload the filenames
            window += 5
            filenames = [file for file in os.listdir(input_dir) if file.endswith('.npy')]

        filename = random.choice(filenames)
        data = np.load(os.path.join(input_dir, filename))
        num_slices = data.shape[0]

        if average_num_slice - window <= num_slices <= average_num_slice + window:
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
    parser.add_argument('fast_ver', type=bool, help='.')
    
    args = parser.parse_args()

    if args.fast_ver:
        fast_create_small_dataset(args.input_dir, args.average_num_slice, args.total_slice, args.output_dir)
    else:
        slow_create_small_dataset(args.input_dir, args.average_num_slice, args.total_slice, args.output_dir)

if __name__ == '__main__':
    main()
