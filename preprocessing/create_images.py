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
from PIL import Image
import random


def create_foam_example(N_PIXEL=128, SIZE_LOWER = 0.01, SIZE_UPPER = 0.2, GAP = 0, Z_SLICES = 32):
    """Creates a single 3D example of a foam phantom"""
    example = []
    for z_index in range(Z_SLICES):
        phantom = xd.Foam(size_range=[SIZE_UPPER, SIZE_LOWER], gap=GAP, porosity=np.random.rand())
        discrete = xd.discrete_phantom(phantom, N_PIXEL)
        example.append(discrete)
    example = np.stack(example, axis=0)/N_PIXEL # shape is Z_SLICES x N_PIXEL x N_PIXEL
    return example, None

# This is for 3D Covid data
def create_covid3D_example(nib_file_path):
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
    example /= 2000
    example[example < 0] = 0

    print(nib_file_path)
    print('example min is ' + str(np.min(example)))
    print('example max is ' + str(np.max(example)))
    print('example median is ' + str(np.median(example)))
    print('example background is ' + str(example[0,0,0]))

    example /= example.shape[1]

    return example, filename

def create_covid2D_example(origin_dir):
    # Get a single 2D example of a covid patient lung scan
    # Return a single 2D slice of the 3D scan with the height of Z_SLICES
    filename = os.listdir(origin_dir)[0]  # Use the first image in the directory
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(os.path.join(origin_dir, filename)).convert('L')
        img = np.array(img)

        # Normalize the pixel values
        img = img / 255.0

        # Reshape the image to have a single channel
        img = img.reshape((1,) + img.shape)

        print('exaplme shape is ' + str(img.shape))
        print('example min is ' + str(np.min(img)))
        print('example max is ' + str(np.max(img)))
        print('example median is ' + str(np.median(img)))
        print('example background is ' + str(img[0,0,0]))
    return img, filename


def create_brain_example(nib_file_path):
    # TODO
    """Get a single 3D example of a single patient brain scan"""
    return



def main(num_examples, rank, world_size, dest_dir, type):
    os.system('mkdir -p ' + dest_dir)
    if type=='covid3D':
        covid3D_list = np.sort(glob.glob('/global/cfs/cdirs/m3562/users/hkim/real_data/raw/*.nii'))
        random.shuffle(covid3D_list)
    if type=='covid2D':
        covid2D_list = np.sort(glob.glob("/global/cfs/cdirs/m3562/users/lchien/Contrastive-COVIDNet/data/SARS-Cov-2/COVID/*.png"))
        random.shuffle(covid2D_list)
    if type=='brain':
        brain_list = None # TODO

    for example_index in range(num_examples):
        if example_index % int(world_size) == rank: # distribute work across ranks
            if type=='foam':
                example, filename = create_foam_example()
            elif type=='covid3D':
                example, filename = create_covid3D_example(covid3D_list[example_index])
            elif type=='covid2D':
                example, filename = create_covid2D_example(covid2D_list[example_index])

            elif type=='brain':
                continue # TODO
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
                        choices=['foam', 'covid3D', 'covid2D', 'brain'])
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    world_size = os.environ['SLURM_NTASKS']
    print('World size: ' + str(world_size))

    # check rank
    rank = comm.rank
    print('Hello from rank: ' + str(rank))

    np.random.seed(0)

    main(args.num_examples, rank, world_size, args.dest_dir, args.type)


    