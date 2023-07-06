import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tomopy
from utils import create_sinogram

def create_sinogram_nib(nib_file_path, target_dir, theta, pad = True, add_ring_artifact=False, save_numpy=False):
    img = nib.load(nib_file_path)
    data = img.get_fdata()
    data = data.transpose((2, 0, 1))
    file_name = os.path.splitext(os.path.basename(nib_file_path))[0]

    data += 2048
    data /= np.max(data)
    data[data < 0] = 0
    proj = create_sinogram(data, theta, pad=pad, add_ring_artifact=add_ring_artifact)

    sinogram_file_path = f"{target_dir}/{file_name}_sinogram.npy"
    np.save(sinogram_file_path, proj)

    if save_numpy:
        # Load .nib file and save it as .npy file
        npy_file_path = f"{target_dir}/{file_name}.npy"
        np.save(npy_file_path, data)
        # print(f"Successfully converted {nib_file_path} to {npy_file_path}.")
    return(data, proj, file_name)

def visualize(data, proj, file_name, target_dir):
    """Visualize the output from create_sinogram_nib"""

    # Plot input images and save plot as PNG file
    num_images = data.shape[0]
    num_rows = num_images // 5 if num_images % 5 == 0 else (num_images // 5) + 1
    fig, axes = plt.subplots(num_rows, 5, figsize=(10, 2 * num_rows))
    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            break
        slice_img = data[i, :, :]
        ax.imshow(slice_img, cmap='gray')
        ax.set_title(f"Slice: {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{target_dir}/plot_{file_name}.png")
    # plt.show()
    
    # Plot sinogram images and save the plot as PNG file
    num_images_to_plot = 18
    interval = proj.shape[0] // num_images_to_plot

    if num_images_to_plot%3 == 0:
        rows = (num_images_to_plot // 3)
    else:
        rows = (num_images_to_plot // 3) + 1
    
    fig, axes = plt.subplots(rows, 3, figsize=(12, 6 * rows))

    for i, ax in enumerate(axes.flat):
        index = i * interval
        if index >= proj.shape[0]:
            break

        image = proj[index, :, :]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Index: {index}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{target_dir}/sinograms_{file_name}.png")
    # plt.show()
