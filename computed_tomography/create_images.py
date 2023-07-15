"""
Creates dataset of 3D objects
Usage:
srun -n NUM_RANKS python create_images.py -n NUM_EXAMPLES --dest SAVE_DIR --type IMG_TYPE

Example for foam images:
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/computed_tomography/create_images.py -n 64 --dest images_foam --type foam

Example for covid images:
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/computed_tomography/create_images.py -n 64 --dest images_covid --type covid
"""

import os
import argparse
import numpy as np
import xdesign as xd 
from mpi4py import MPI
import nibabel as nib
import glob

def create_foam_example(N_PIXEL=128, SIZE_LOWER = 0.01, SIZE_UPPER = 0.2, GAP = 0, Z_SLICES = 32):
    """Creates a single 3D example of a foam phantom"""
    example = []
    for z_index in range(Z_SLICES):
        phantom = xd.Foam(size_range=[SIZE_UPPER, SIZE_LOWER], gap=GAP, porosity=np.random.rand())
        discrete = xd.discrete_phantom(phantom, N_PIXEL)
        example.append(discrete)
    example = np.stack(example, axis=0) # shape is Z_SLICES x N_PIXEL x N_PIXEL
    return example, None

def create_covid_example(nib_file_path):
    """Get a single 3D example of a covid patient lung scan"""
    img = nib.load(nib_file_path)
    example = img.get_fdata()
    example = example.transpose((2, 0, 1))
    filename = os.path.splitext(os.path.basename(nib_file_path))[0]

    example += 2048
    example /= np.max(example)
    example[example < 0] = 0

    return example, filename

def create_dataset(num_examples, rank, world_size, dest_dir, type):
    os.system('mkdir -p ' + dest_dir)
    if type=='covid':
        covid_list = np.sort(glob.glob('/global/cfs/cdirs/m3562/users/hkim/real_data/raw/*.nii'))

    for example_index in range(num_examples):
        if example_index % int(world_size) == rank: # distribute work across ranks
            if type=='foam':
                example, filename = create_foam_example()
            elif type=='covid':
                example, filename = create_covid_example(covid_list[example_index])
            else:
                raise NotImplementedError('image type not implemented')
            if filename is None:
                filename = 'example_' + str(example_index)

            np.save(dest_dir + '/' + filename + '.npy', example)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('-n', dest = 'num_examples', type=int, help='number of total examples', default=64)
    parser.add_argument('--dest', dest = 'dest_dir', type=str, help='where the numpy files are saved')
    parser.add_argument('--type', dest = 'type', type=str, help='type of data to create', default='foam', 
                        choices=['foam', 'covid'])
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    world_size = os.environ['SLURM_NTASKS']
    print('World size: ' + str(world_size))

    # check rank
    rank = comm.rank
    print('Hello from rank: ' + str(rank))

    np.random.seed(0)

    create_dataset(args.num_examples, rank, world_size, args.dest_dir, args.type)


    