import h5py
import os

# Get working directory environment variable
WORKING_DIR = os.environ['WORKING_DIR']
filename = WORKING_DIR + '/bread_clean.h5'

# Function to recursively print the hierarchy
def print_hierarchy(group, level=0):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print("  " * level + f"Group: {key}")
            print_hierarchy(item, level + 1)
        elif isinstance(item, h5py.Dataset):
            print("  " * level + f"Dataset: {key}")

# Open the HDF5 file using a with statement
with h5py.File(filename, 'r') as file:
    print("Hierarchy of the HDF5 file:")
    print_hierarchy(file)
    print(file["exchange"]["data"].shape) # dimensions are num_proj_pix x num_angles x z_slices
    print(file["exchange"]["rot_center"][()]) # not the center of num_proj_pix
    print(file["exchange"]["theta"][()].shape)
    print(file["exchange"]["theta"][()])

    data = file["exchange"]["data"][()]  # dimensions are num_proj_pix x num_angles x z_slices
    rot_center = file["exchange"]["rot_center"][()]
    theta = file["exchange"]["theta"][()]
# The file is automatically closed when the 'with' block is exited

