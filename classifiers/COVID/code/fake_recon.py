# Usage: python $CT_NVAE_PATH/classifiers/COVID/code/fake_recon.py $PREPROCESSED_PATH $RECON_PATH
import os
import shutil
import argparse

def process_files(input_folder_path, output_folder_path):
    print('input_folder_path', input_folder_path)
    # Get a list of all file names in the directory
    files = os.listdir(input_folder_path)

    # Make sure output directory exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Loop over all files in the directory
    for file in files:
        # Construct the full path of the file
        file_path = os.path.join(input_folder_path, file)

        # Check if the file is "theta.npy"
        if file == "theta.npy":
            continue

        # Check if the file is a non-label and non-sinogram .npy file
        elif file.endswith('.npy') and not ('_label.npy' in file or '_sinogram.npy' in file):
            # Rename the file to [original name]_recon.npy
            new_file_name = file.replace('.npy', '_recon.npy')
        else:
            # Keep the original file name
            new_file_name = file

        # Construct the full path of the destination file
        dest_path = os.path.join(output_folder_path, new_file_name)

        # Copy the file to the output directory
        shutil.copyfile(file_path, dest_path)
    print('output_folder_path', output_folder_path)


if __name__ == '__main__':
    # Create a parser for the command-line arguments
    parser = argparse.ArgumentParser(description='Copy and rename .npy files.')
    parser.add_argument('input_folder_path', type=str, help='Path to the input folder.')
    parser.add_argument('output_folder_path', type=str, help='Path to the output folder.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the path to your directory
    process_files(args.input_folder_path, args.output_folder_path)