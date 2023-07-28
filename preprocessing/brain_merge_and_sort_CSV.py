# Purpose: Merge and sort CSV files.
# Usage: python preprocessing/brain_merge_and_sort_CSV.py <csv_directory> <final_csv_path> <thickness_output_path>
# Example: python $CT_NVAE_PATH/preprocessing/brain_merge_and_sort_CSV.py $TEMP_CSV_DIR $FINAL_CSV_PATH $THICKNESS


import os
import glob
import pandas as pd
import argparse
# Within each 'StudyInstanceUID', the rows will be sorted by 'LastPosition'.
# The groups of 'StudyInstanceUID' will be ordered by their 'thickness

def merge_and_sort_files(csv_directory, final_csv_path, thickness_output_path):
    # Get all CSV files in the directory
    csv_files = sorted(glob.glob(os.path.join(csv_directory, '*.csv')), key=lambda x: int(os.path.basename(x).split('_')[0]))

    # Read and concatenate all CSV files
    df = pd.concat((pd.read_csv(f) for f in csv_files))

    # Convert ImagePositionPatient from string to list
    df['ImagePositionPatient'] = df['ImagePositionPatient'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

    # Create a new column for the last element of ImagePositionPatient
    df['LastPosition'] = df['ImagePositionPatient'].apply(lambda x: x[-1])

    # Calculate 'thickness' for each StudyInstanceUID
    thickness = df.groupby('StudyInstanceUID').size()
    df['thickness'] = df['StudyInstanceUID'].map(thickness)

    # Create a DataFrame for thickness and remove duplicates
    thickness_df = pd.DataFrame(thickness).reset_index()
    thickness_df.columns = ['StudyInstanceUID', 'thickness']
    thickness_df = thickness_df.drop_duplicates()

    # Write the StudyInstanceUID and its corresponding thickness to a new CSV file
    thickness_df.to_csv(thickness_output_path, index=False)

    # Sort dataframe by 'StudyInstanceUID', 'LastPosition', and 'thickness'
    df = df.sort_values(['StudyInstanceUID', 'LastPosition', 'thickness'])

    # Write the sorted dataframe to a new CSV file
    df.to_csv(final_csv_path, index=False)

def main():
    parser = argparse.ArgumentParser(description='Merge and sort CSV files.')
    parser.add_argument('csv_directory', type=str, help='Path to the directory containing CSV files.')
    parser.add_argument('final_csv_path', type=str, help='Path to the output CSV file.')
    parser.add_argument('thickness_output_path', type=str, help='Path to the output CSV file for thickness.')

    args = parser.parse_args()

    merge_and_sort_files(args.csv_directory, args.final_csv_path, args.thickness_output_path)

if __name__ == "__main__":
    main()
