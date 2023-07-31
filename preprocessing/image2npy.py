import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

def image_to_3D_npy(origin_dir, dest_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    files = os.listdir(origin_dir)
    for f in tqdm(files, desc="Converting images to npy"):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(origin_dir, f)).convert('L')
            img = np.array(img)

            # Normalize the pixel values
            img = img / 255.0

            # Reshape the image to have a single channel
            img = img.reshape((1,) + img.shape)

            name = os.path.splitext(f)[0] + '.npy'
            np.save(os.path.join(dest_dir, name), img)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python preprocessing/image2npy.py $PNG_2D_PATH $NPY_3D_PATH')
        sys.exit(1)
    
    origin_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    image_to_3D_npy(origin_dir, dest_dir)
