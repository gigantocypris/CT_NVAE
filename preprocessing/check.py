import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import argparse

def read_and_analyze_npy_files(directory):
    # List of known irregular files
    irregular_files = ["ID_ce763e71.npy", "ID_6a01c035.npy", "ID_6c3ed7f0.npy", "ID_2b9e0826.npy", "ID_796b5208.npy", "ID_a540b550.npy", "ID_5d81e0ab.npy"]
    irregular_filepaths = [os.path.join(directory, filename) for filename in irregular_files]

    # Get all .npy files in the directory, excluding known irregular files
    filepaths = [path for path in glob.glob(os.path.join(directory, "*.npy")) if path not in irregular_filepaths]
    
    slice_counts = []

    for filepath in tqdm(filepaths, desc="Processing .npy files"):
        try:
            # Load the .npy file
            volume = np.load(filepath)

            # Check the shape of the 3D array
            if len(volume.shape) == 3 and volume.shape[1:] == (512, 512):
                slice_counts.append(volume.shape[0])
        except Exception as e:
            print(f"Error reading .npy file {filepath}: {e}")
    
    # Plot a histogram of the number of slices for each .npy file
    plt.hist(slice_counts, bins='auto')
    plt.title("Histogram of number of slices in .npy files")
    plt.xlabel("Number of slices")
    plt.ylabel("Number of .npy files")
    plt.savefig("/global/cfs/cdirs/m3562/users/hkim/brain_data/slice_histogram.png")  # Save the plot as a .png file in the directory

    print("Histogram saved as slice_histogram.png in the input directory.")


def main():
    parser = argparse.ArgumentParser(description='Process and analyze .npy files in a directory.')
    parser.add_argument('directory', type=str, help='The directory containing the .npy files.')
    
    args = parser.parse_args()

    read_and_analyze_npy_files(args.directory)

if __name__ == '__main__':
    main()
