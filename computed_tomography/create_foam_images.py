"""
Creates synthetic dataset of foam objects
Usage:
srun -n NUM_RANKS python create_foam_images.py -n NUM_EXAMPLES
"""

import os
import argparse
import numpy as np
import xdesign as xd 
from mpi4py import MPI


def create_3d_example(SIZE_UPPER, SIZE_LOWER, N_PIXEL, GAP, Z_SLICES):
    example = []
    for z_index in range(Z_SLICES):
        phantom = xd.Foam(size_range=[SIZE_UPPER, SIZE_LOWER], gap=GAP, porosity=np.random.rand())
        discrete = xd.discrete_phantom(phantom, N_PIXEL)
        example.append(discrete)
    example = np.stack(example, axis=0) # shape is Z_SLICES x N_PIXEL x N_PIXEL
    return example

def create_dataset(num_examples, rank, world_size,
                   SIZE_UPPER, SIZE_LOWER, N_PIXEL, GAP, Z_SLICES,
                   save_prefix, save_dir):
    os.system('mkdir -p ' + save_dir)
    for example_index in range(num_examples):
        if example_index % world_size == rank:
            example = create_3d_example(SIZE_UPPER, SIZE_LOWER, N_PIXEL, GAP, Z_SLICES)
            np.save(save_prefix + '_' + str(example_index) + '.npy', example)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('-n', dest = 'num_examples', type=int, help='number of total examples', default=64)
    args = parser.parse_args()
    
    ### INPUTS ###

    N_PIXEL = 128 # size of each phantom is N_PIXEL x N_PIXEL

    # parameters to generate the foam phantoms
    SIZE_LOWER = 0.01
    SIZE_UPPER = 0.2
    GAP = 0
    Z_SLICES = 32

    num_examples= args.num_examples # number of 3D phantoms created

    ### END OF INPUTS ###

    comm = MPI.COMM_WORLD
    world_size = comm.get_size()
    print('World size: ' + str(world_size))

    # check rank
    rank = comm.rank
    print('Hello from rank: ' + str(rank))

    np.random.seed(0)

    save_prefix = 'foam_3D_'
    create_dataset(num_examples, rank, world_size,
                    SIZE_UPPER, SIZE_LOWER, N_PIXEL, GAP, Z_SLICES,
                    save_prefix)


    