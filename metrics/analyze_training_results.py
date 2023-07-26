import numpy as np
from computed_tomography.utils import reconstruct_sinogram
import matplotlib.pyplot as plt
from metrics.utils import crop, compare

# TODO make command line arguments
WORKING_DIR = '/pscratch/sd/v/vidyagan/output_CT_NVAE'
DATASET = 'foam_ring'
CHECKPOINT_DIR = 'checkpts'
EXPR_ID = 'test_0000_foam2_ring'
results_path = WORKING_DIR + '/' + CHECKPOINT_DIR + '/eval-' + EXPR_ID
dataset_path = WORKING_DIR + '/dataset_' + DATASET
dataset_type = 'valid'
rank = 0
img_ind = 0
algorithm = 'sirt'
original_size = 128

# Load the final validation *.npy files
final_phantoms = np.load(results_path + '/final_phantoms_' + dataset_type + '_rank_' + str(rank) + '.npy')
ground_truths = np.load(results_path + '/final_ground_truth_' + dataset_type + '_rank_' + str(rank) + '.npy')
sparse_sinograms = np.load(results_path + '/final_sparse_sinograms_' + dataset_type + '_rank_' + str(rank) + '.npy')
theta = np.load(results_path + '/final_theta_' + dataset_type + '_rank_' + str(rank) + '.npy')

ground_truth = ground_truths[img_ind]
sparse_sinogram = np.expand_dims(sparse_sinograms[img_ind], axis=1)
theta_0 = theta[img_ind]
final_phantom = final_phantoms[img_ind]

reconstruct_0 = reconstruct_sinogram(sparse_sinogram, theta_0, remove_ring_artifact=False, algorithm=algorithm)
reconstruct_1 = reconstruct_sinogram(sparse_sinogram, theta_0, remove_ring_artifact=True, algorithm=algorithm)

# Remove z dimension
reconstruct_0 = np.squeeze(reconstruct_0, axis=0)
reconstruct_1 = np.squeeze(reconstruct_1, axis=0)

# Crop images back to their original size
ground_truth = crop(ground_truth, original_size, original_size)
final_phantom = crop(final_phantom, original_size, original_size)
reconstruct_0 = crop(reconstruct_0, original_size, original_size)
reconstruct_1 = crop(reconstruct_1, original_size, original_size)

# Get MSE, PSNR, SSIM for the final validation results compared to the ground truth and tomopy reconstructions
print('Tomopy Reconstruction')
mse_recon, ssim_recon, psnr_recon, err_string_0 = compare(ground_truth, reconstruct_0, verbose=True)

print('Tomopy Reconstruction with Ring Artifact Removal')
mse_recon, ssim_recon, psnr_recon, err_string_1 = compare(ground_truth, reconstruct_1, verbose=True)

print('CT_NVAE Reconstruction')
mse_recon, ssim_recon, psnr_recon, err_string_NVAE = compare(ground_truth, final_phantom, verbose=True)

# Create a multi-panel figure with the following: ground truth, tomopy reconstructions, CT_NVAE reconstruction
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(ground_truth, cmap='gray')
plt.title('Ground Truth')
plt.subplot(2, 2, 2)
plt.imshow(reconstruct_0, cmap='gray')
plt.title('Tomopy Reconstruction')
plt.subplot(2, 2, 3)
plt.imshow(reconstruct_1, cmap='gray')
plt.title('Tomopy Reconstruction with Ring Artifact Removal')
plt.subplot(2, 2, 4)
plt.imshow(final_phantom, cmap='gray')
plt.title('CT_NVAE Reconstruction')
plt.savefig(results_path + '/final_visualization_' + dataset_type + '_img_' + str(img_ind) + '.png')