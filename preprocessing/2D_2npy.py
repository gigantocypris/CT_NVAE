# mkdir the destination first

import os
import sys
import numpy as np
from PIL import Image

def image2npy(origin_dir, dest_dir):
    for f in os.listdir(origin_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(origin_dir, f)).convert('L')
            img = np.array(img)
            name = os.path.splitext(f)[0] + '.npy'
            np.save(os.path.join(dest_dir, name), img)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python image2npy.py origin_directory npy_directory')
        sys.exit(1)
    
    origin_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    image2npy(origin_dir, dest_dir)
