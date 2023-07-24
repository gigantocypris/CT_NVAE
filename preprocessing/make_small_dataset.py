# Created July 24, 2023 by Hojune Kim
# Purpose: Create a small dataset for testing pipeline
# Usage: python make_small_dataset.py <input_dir> <output_dir> --num_files <num_files>
# Example: python $CT_NVAE_PATH/preprocessing/make_small_dataset.py $SOURCE_DIR $SMALL_TARGET_DIR --num_files 1000

import shutil
import os
import random
from tqdm import tqdm
import argparse

def copy_random_files(input_dir, output_dir, num_files):
    # Get a list of all files in the input directory
    filenames = os.listdir(input_dir)

    # Randomly select num_files files
    filenames_to_copy = random.sample(filenames, num_files)

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Copy the selected files to the output directory
    for filename in tqdm(filenames_to_copy, desc="Copying files"):
        shutil.copy(os.path.join(input_dir, filename), output_dir)

def main():
    parser = argparse.ArgumentParser(description='Copy a random selection of files from one directory to another.')
    parser.add_argument('input_dir', type=str, help='The directory to copy files from.')
    parser.add_argument('output_dir', type=str, help='The directory to copy files to.')
    parser.add_argument('--num_files', type=int, default=1000, help='The number of files to copy.')
    
    args = parser.parse_args()

    copy_random_files(args.input_dir, args.output_dir, args.num_files)

if __name__ == '__main__':
    main()
