"""
Processes each .npy object file into a sinogram and saves it as a .npy file.
Usage:
python create_sinograms.py --dir <dir> -n <num_examples>

Example for foam images:
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/computed_tomography/create_sinograms.py --dir images_foam

Example for covid images:
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/computed_tomography/create_sinograms.py --dir images_covid
"""

import os
import argparse
import numpy as np
from mpi4py import MPI
import glob
from utils import create_sinogram

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

