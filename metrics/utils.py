import numpy as np

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
    return(mse_recon, ssim_recon, psnr_recon, err_string)

def crop(img_2d, final_x, final_y):
    """Center crop 2D image"""
    x,y = img_2d.shape
    remain_x = final_x % 2
    remain_y = final_y % 2

    return(img_2d[x//2 - final_x//2:x//2+final_x//2+remain_x, y//2-final_y//2:y//2+final_y//2+remain_y])