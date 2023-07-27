from computed_tomography.utils import reconstruct_sinogram
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio



def compare(recon0, recon1, verbose=False):

    mse_recon = mean_squared_error(recon0, recon1)
    # np.mean((recon0-recon1)**2)
    
    small_side = np.min(recon0.shape)
    if small_side<7:
        if small_side%2: # if odd
            win_size=small_side
        else:
            win_size=small_side-1
    else:
        win_size=None

    ssim_recon = ssim(recon0, recon1,
                      data_range=recon0.max() - recon0.min(), win_size=win_size)
    
    
    psnr_recon = peak_signal_noise_ratio(recon0, recon1,
                      data_range=recon0.max() - recon0.min())
    
    err_string = 'MSE: {:.8f}, SSIM: {:.3f}, PSNR: {:.3f}'
    if verbose:
        print(err_string.format(mse_recon, ssim_recon, psnr_recon))
    err_vec = [mse_recon, ssim_recon, psnr_recon]
    return(err_vec, err_string)

def crop(img_2d, final_x, final_y):
    """Center crop 2D image"""
    x,y = img_2d.shape
    remain_x = final_x % 2
    remain_y = final_y % 2

    return(img_2d[x//2 - final_x//2:x//2+final_x//2+remain_x, y//2-final_y//2:y//2+final_y//2+remain_y])

def analyze_single_slice(sparse_sinogram, theta_0, ground_truth, final_phantom, original_size, algorithm, verbose=True):
    reconstruct = reconstruct_sinogram(sparse_sinogram, theta_0, remove_ring_artifact=False, algorithm=algorithm)
    reconstruct_ring = reconstruct_sinogram(sparse_sinogram, theta_0, remove_ring_artifact=True, algorithm=algorithm)

    # Remove z dimension
    reconstruct = np.squeeze(reconstruct, axis=0)
    reconstruct_ring = np.squeeze(reconstruct_ring, axis=0)

    # Crop images back to their original size
    ground_truth = crop(ground_truth, original_size, original_size)
    final_phantom = crop(final_phantom, original_size, original_size)
    reconstruct = crop(reconstruct, original_size, original_size)
    reconstruct_ring = crop(reconstruct_ring, original_size, original_size)

    # Get MSE, PSNR, SSIM for the final validation results compared to the ground truth and tomopy reconstructions
    if verbose:
        print('Tomopy Reconstruction')
    err_vec, err_string = compare(ground_truth, reconstruct, verbose=verbose)

    if verbose:
        print('Tomopy Reconstruction with Ring Artifact Removal')
    err_vec_ring, err_string_ring = compare(ground_truth, reconstruct_ring, verbose=verbose)

    if verbose:
        print('CT_NVAE Reconstruction')
    err_vec_NVAE, err_string_NVAE = compare(ground_truth, final_phantom, verbose=verbose)

    return(ground_truth, reconstruct, reconstruct_ring, final_phantom, err_vec, err_vec_ring, err_vec_NVAE)


def visualize_single_slice(ground_truth, reconstruct_0, reconstruct_1, final_phantom, 
                           results_path, dataset_type, rank, img_ind):
    # Create a multi-panel figure with the following: ground truth, tomopy reconstructions, CT_NVAE reconstruction
    plt.figure(figsize=(12, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Ground Truth')
    vmin = np.min(ground_truth)
    vmax = np.max(ground_truth)

    plt.subplot(2, 2, 2)
    plt.imshow(reconstruct_0, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Tomopy Reconstruction')
    
    plt.subplot(2, 2, 3)
    plt.imshow(reconstruct_1, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Tomopy Reconstruction with Ring Artifact Removal')
    
    plt.subplot(2, 2, 4)
    plt.imshow(final_phantom, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('CT_NVAE Reconstruction')
    plt.savefig(results_path + '/final_visualization_' + dataset_type + '_rank_' + str(rank) + '_img_' + str(img_ind) + '.png')