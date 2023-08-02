"""
Creates dataset of 3D objects
"""

import os
import argparse
import numpy as np
import xdesign as xd 
from mpi4py import MPI
import nibabel as nib
import glob
import gzip
import shutil

def create_foam_example(N_PIXEL=128, SIZE_LOWER = 0.01, SIZE_UPPER = 0.2, GAP = 0, Z_SLICES = 32):
    """Creates a single 3D example of a foam phantom"""
    example = []
    for z_index in range(Z_SLICES):
        phantom = xd.Foam(size_range=[SIZE_UPPER, SIZE_LOWER], gap=GAP, porosity=np.random.rand())
        discrete = xd.discrete_phantom(phantom, N_PIXEL)
        example.append(discrete)
    example = np.stack(example, axis=0) # shape is Z_SLICES x N_PIXEL x N_PIXEL
    example = example/N_PIXEL
    return example, None

def create_covid_example(nib_file_path):
    """Get a single 3D example of a covid patient lung scan"""

    if nib_file_path[-3:]=='.gz':
        destination_path = os.path.splitext(nib_file_path)[0]

        # Open the .gz file and extract its contents
        with gzip.open(nib_file_path, 'rb') as gz_file:
            with open(destination_path, 'wb') as extracted_file:
                shutil.copyfileobj(gz_file, extracted_file)

    img = nib.load(nib_file_path)
    example = img.get_fdata()
    example = example.transpose((2, 0, 1))
    filename = os.path.splitext(os.path.basename(nib_file_path))[0]



    example += 2048
    example /= 2000 # globally normalize such that the sinogram norm is approximately 1
    example[example < 0] = 0

    print(nib_file_path)
    print('example min is ' + str(np.min(example)))
    print('example max is ' + str(np.max(example)))
    print('example median is ' + str(np.median(example)))
    print('example background is ' + str(example[0,0,0]))

    example /= example.shape[1]

    return example, filename

def main(num_examples, rank, world_size, dest_dir, type):
    os.system('mkdir -p ' + dest_dir)
    if type=='covid':
        # get from environment variable
        covid_raw_data = os.environ['COVID_RAW_DATA'] 
        covid_list = np.sort(glob.glob(covid_raw_data + '/*.nii'))

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

    main(args.num_examples, rank, world_size, args.dest_dir, args.type)


    