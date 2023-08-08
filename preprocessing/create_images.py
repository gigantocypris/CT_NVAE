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
from skimage.transform import resize


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


def create_covid2D_example(origin_file_path):
    # Get a single 2D example of a covid patient lung scan
    # Return a single 2D slice of the 3D scan with the height of Z_SLICES
    filename = os.path.basename(origin_file_path)  # Get the filename from the path
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(origin_file_path).convert('L')  # Open the image directly from the path

        # Normalize the pixel values using min-max normalization
        img = np.array(img, dtype=np.float32)
        img = img - 167.97
        img = img / 224
        img[img < 0] = 0

        # Resize the image to (224, 224)
        img = resize(img, (224, 224))
        # Reshape the image to have a single channel
        img = img.reshape((1,) + img.shape)
        # print('example shape is ' + str(img.shape))
        print('example min is ' + str(np.min(img)))
        print('example max is ' + str(np.max(img)))
        print('example mean is ' + str(np.mean(img)))
        print('example background is ' + str(img[0,0,0]))

    # Remove the file extension from the filename - so that it's not img1.png.npy
    filename = os.path.splitext(filename)[0]
    filename = filename.replace(' ', '')

    # Search for the path in 'merged.txt' and extract the label
    with open('/global/cfs/cdirs/m3562/users/hkim/2DCovid/SARS/merged.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if origin_file_path in line:
                label = line.strip().split()[-1]  # Extract the last word as the label
                break
        else:
            label = "Unknown"  # If the path is not found, set the label to "Unknown"

    return img, filename, label



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
        covid2D_list = np.sort(glob.glob("/global/cfs/cdirs/m3562/users/hkim/2DCovid/SARS/merged/*.png"))
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
                example, filename, label = create_covid2D_example(covid2D_list[example_index])
                # np.save(dest_dir + '/' + filename + "_label.npy", label)
            elif type=='brain':
                continue # TODO
            else:
                raise NotImplementedError('image type not implemented')
            if filename is None:
                filename = 'example_' + str(example_index)

            np.save(dest_dir + '/' + filename + '.npy', example)
    print('Rank ' + str(rank) + ' finished creating ' + str(num_examples) + ' examples of type ' + type)


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