"""
Processes each .npy object file into a sinogram and saves it as a .npy file.
"""

import os
import argparse
import numpy as np
from mpi4py import MPI
import glob
from computed_tomography.utils import create_sinogram

def main(rank, world_size, dir, theta):
    file_list = [f for f in np.sort(glob.glob(dir + '/*.npy')) if not (f.endswith("_label.npy") or f.endswith("_sinogram.npy") or  f.endswith("theta.npy"))]
    total_files = len(file_list)
    error_messages = []  # List to store error messages
    
    for example_index in range(total_files):
        if example_index % int(world_size) == rank: # distribute work across ranks
            img_stack = np.load(file_list[example_index])
            
            # Check if img_stack is a 3D array
            if len(img_stack.shape) != 3:
                error_msg = f"[Rank {rank}] Skipped file {file_list[example_index]} due to unexpected shape {img_stack.shape}."
                error_messages.append(error_msg)
                continue

            proj = create_sinogram(img_stack, theta, pad=True)
            filename = os.path.splitext(os.path.basename(file_list[example_index]))[0]
            np.save(dir + '/' + filename + '_sinogram.npy', proj)
            # Print progress
            print(f"[Rank {rank}] Processed file {example_index + 1} of {total_files}")
    
    # Print error messages at the end
    if error_messages:
        print("\n--- Errors encountered during processing ---")
        for msg in error_messages:
            print(msg)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('--dir', dest = 'dir', type=str, help='where the numpy object files are saved')
    args = parser.parse_args()
    theta = np.linspace(0, 2*np.pi, 180, endpoint=False)

    comm = MPI.COMM_WORLD
    world_size = os.environ['SLURM_NTASKS']
    print('World size: ' + str(world_size))

    # check rank
    rank = comm.rank
    print('Hello from rank: ' + str(rank))

    np.random.seed(0)

    main(rank, world_size, args.dir, theta)
    
    if rank==0:
        np.save(args.dir + '/theta.npy', theta)