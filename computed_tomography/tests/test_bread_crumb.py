import h5py
import os
import numpy as np
import tomopy
import matplotlib.pyplot as plt

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

    data = file["exchange"]["data"][()]  # dimensions are  z_slices x num_angles  x num_proj_pix, (2160, 1313, 3620) for bread_clean.h5
    rot_center = file["exchange"]["rot_center"][()]
    theta = file["exchange"]["theta"][()]

# Reconstruct the bread crumb

# recontruct with tomopy
proj_full = np.transpose(data, axes=[1,0,2])  # num_angles x z_slices x num_proj_pix

# Trim the data to a single slice
z_ind = 500

plt.figure()
plt.imshow(proj_full[:,z_ind, :])
plt.colorbar()
plt.savefig('bread_crumb_sinogram.png', dpi=300, bbox_inches='tight')


proj_full = np.expand_dims(proj_full[:, z_ind, :], axis=1)

# linearize the data
# proj_full = -np.log(proj_full)
# proj_full = tomopy.minus_log(proj_full)

print('calculating gridrec')
rec_gridrec = tomopy.recon(proj_full, theta, algorithm='gridrec',center=rot_center, 
                           sinogram_order=False)
rec_gridrec = np.squeeze(rec_gridrec)
plt.figure()
plt.imshow(rec_gridrec)
plt.colorbar()
plt.savefig('bread_crumb_gridrec.png', dpi=300, bbox_inches='tight')

print('calculating tv')
rec_tv = tomopy.recon(proj_full, theta, algorithm='tv',center=rot_center, 
                      sinogram_order=False,reg_par=1e-5,num_iter=10)
rec_tv = np.squeeze(rec_tv)

plt.figure()
plt.imshow(rec_tv)
plt.colorbar()
plt.savefig('bread_crumb_tv.png', dpi=300, bbox_inches='tight')


print('calculating sirt')
rec_sirt = tomopy.recon(proj_full, theta, algorithm='sirt',center=rot_center, 
                        sinogram_order=False, interpolation='LINEAR', num_iter=10)
rec_sirt = np.squeeze(rec_sirt)
plt.figure()
plt.imshow(rec_sirt)
plt.colorbar()
plt.savefig('bread_crumb_sirt.png', dpi=300, bbox_inches='tight')
breakpoint()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# change fig relative size
fig.set_size_inches(3, 2)
fig.suptitle('Bread Crumb Reconstruction')

vmax = np.max(rec_gridrec)
ax1.title.set_text('gridrec')
ax1.imshow(rec_gridrec,vmin=0,vmax=vmax)
# remove all axes
ax1.axis('off')

ax2.title.set_text('TV')
ax2.imshow(rec_tv,vmin=0,vmax=vmax)
ax2.axis('off')

ax3.title.set_text('SIRT')
ax3.imshow(rec_sirt)
ax3.axis('off')



plt.savefig('bread_crumb.png', dpi=300, bbox_inches='tight')
