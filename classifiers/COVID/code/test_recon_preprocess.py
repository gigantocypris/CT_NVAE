# Usage: python $CT_NVAE_PATH/classifiers/COVID/code/test_recon_preprocess.py $RECON_PATH $RECON_PNG_PATH

import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def process_files(input_folder_path, output_folder_path):
    # Get a list of all file names in the directory
    files = os.listdir(input_folder_path)

    # Split the files into two lists: reconstructions and labels
    label_files = [f for f in files if '_label.npy' in f]

    # Check if output_folder_path exists, create if it doesn't
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Create a txt file to store the image paths and their labels
    with open(os.path.join(output_folder_path, 'recon_label.txt'), 'w') as f:
        # Read label files, convert their corresponding recon files to png, 
        # and write their paths and labels to the txt file
        for label_file in tqdm(label_files):
            # Load the .npy file
            label = np.load(os.path.join(input_folder_path, label_file))

            # Get the image number from the file name
            img_number = label_file.replace('_label.npy', '')

            # Load the corresponding recon file
            recon_file = img_number + '_recon.npy'
            recon = np.load(os.path.join(input_folder_path, recon_file))
            recon = recon.reshape(224,224)

            # Convert the numpy array to a PIL Image object and save it as a png file
            recon = (recon * 255).astype(np.uint8)
            img = Image.fromarray(recon)
            img = Image.fromarray(recon).convert('RGB')  # Convert grayscale to RGB
            img_path = os.path.join(output_folder_path, f'{img_number}.png')
            img.save(img_path)

            # Write the image path and its label to the txt file
            f.write(f'{img_path} {label.item()}\n')

if __name__ == '__main__':
    # Create a parser for the command-line arguments
    parser = argparse.ArgumentParser(description='Process .npy files.')
    parser.add_argument('input_folder_path', type=str, help='Path to the input folder.')
    parser.add_argument('output_folder_path', type=str, help='Path to the output folder.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the path to your directory
    process_files(args.input_folder_path, args.output_folder_path)