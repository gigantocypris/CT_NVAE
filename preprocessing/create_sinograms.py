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
    file_list = np.sort(glob.glob(dir + '/*[!_sinogram].npy'))
    for example_index in range(len(file_list)):
        if example_index % int(world_size) == rank: # distribute work across ranks
            img_stack = np.load(file_list[example_index])
            proj = create_sinogram(img_stack, theta, pad=True)
            filename = os.path.splitext(os.path.basename(file_list[example_index]))[0]
            np.save(dir + '/' + filename + '_sinogram.npy', proj)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('--dir', dest = 'dir', type=str, help='where the numpy object files are saved')
    parser.add_argument('--type', dest = 'type', type=str, help='type of data that was created', default='foam', 
                    choices=['foam', 'covid', 'toy'])
    
    args = parser.parse_args()
    if args.type=='toy':
        theta = np.linspace(0, np.pi, 2, endpoint=False)
    else:
        theta = np.linspace(0, np.pi, 180, endpoint=False)

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

