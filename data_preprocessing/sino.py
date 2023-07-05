import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tomopy

# Created by: Hojune Kim
# Date: May 20, 2021
# Purpose: Create sinogram from .nib file and save it as .npy file

# Usage: python sino.py <nib_file_path> <only_sinogram> <pad>
# Example: python sino.py data/Covid_CT_1.nii False True
# If only_sinogram=True, then it will only create sinogram and save it as .npy file
# If only_sinogram=False, then it will save plot and .npy file for both sinogram and its original image

# Make sure to create data folder and figures folder within this directory before running this code

def create_sinogram(nib_file_path, theta, only_sinogram, pad=True):
    
    img = nib.load(nib_file_path)
    data = img.get_fdata()
    data = data.transpose((2, 0, 1))
    file_name = os.path.splitext(os.path.basename(nib_file_path))[0]

    if only_sinogram:
        print("only_sinogram is True. Only sinogram will be created and saved as .npy file.")
        proj = tomopy.project(data, theta, center=None, emission=True, pad=pad, sinogram_order=False)
        sino = proj.transpose((0, 1, 2))  # Adjust the transpose order
        sinogram_file_path = f"data/{file_name}_sinogram.npy"
        np.save(sinogram_file_path, sino)

    else:
        print("only_sinogram is False. Sinogram and its original image will be created and saved as .npy file.")
        # Load .nib file and save it as .npy file
        npy_file_path = f"data/{file_name}.npy"
        np.save(npy_file_path, data)
        print(f"Successfully converted {nib_file_path} to {npy_file_path}.")
    

        # Plot input imgaes and save plot as PNG file
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
        file_name = os.path.splitext(os.path.basename(npy_file_path))[0]
        plt.savefig(f"figures/plot of {file_name}.png")
        # plt.show()


        # Create sinogram and save it as .npy file
        print("input shape:", data.shape)  # Input shape: (512, 512, 70)
        proj = tomopy.project(data, theta, center=None, emission=True, pad=pad, sinogram_order=False)
        sino = proj.transpose((0, 1, 2))  # Adjust the transpose order
        print("output shape:", sino.shape)  # Output shape: (180, 70, 512)
        sinogram_file_path = f"data/{file_name}_sinogram.npy"
        np.save(sinogram_file_path, sino)
        

        # Plot sinogram images and save the plot as PNG file
        num_images_to_plot = 18
        interval = sino.shape[0] // num_images_to_plot

        if num_images_to_plot%3 == 0:
            rows = (num_images_to_plot // 3)
        else:
            rows = (num_images_to_plot // 3) + 1
        
        fig, axes = plt.subplots(rows, 3, figsize=(12, 6 * rows))

        for i, ax in enumerate(axes.flat):
            index = i * interval
            if index >= sino.shape[0]:
                break

            image = sino[index, :, :]
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Index: {index}")
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"figures/sinogram of {file_name}.png")
        # plt.show()


if __name__ == "__main__":
    # Extract command-line arguments
    if len(sys.argv) < 4:
        print("Usage: python sino.py <nib_file_path> <only_sinogram> <pad>")
        sys.exit(1)

    # Assign command-line arguments to variables
    nib_file_path = sys.argv[1]
    only_sinogram = sys.argv[2].lower() == "true"
    pad = sys.argv[3].lower() == "true"

    # Additional code remains the same
    theta = np.arange(0, 180, 1)
    create_sinogram(nib_file_path, theta, only_sinogram, pad)