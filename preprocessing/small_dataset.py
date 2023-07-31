import os
import sys
import shutil
import random
from tqdm import tqdm

def copy_random_npy_files(input_dir, output_dir, num_files):
    # Get a list of all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    # Select num_files files randomly
    selected_files = random.sample(npy_files, num_files)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the selected files to the output directory
    for file in tqdm(selected_files, desc="Copying files"):
        shutil.copy(os.path.join(input_dir, file), os.path.join(output_dir, file))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: python preprocessing/small_dataset.py $NPY_3D_PATH $SMALL_NPY_3D_PATH 50')
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_files = int(sys.argv[3])

    copy_random_npy_files(input_dir, output_dir, num_files)

