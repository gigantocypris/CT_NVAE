from tqdm import tqdm
import os

dataset_path = '/global/cfs/cdirs/m3562/users/lchien/Contrastive-COVIDNet'
testfile = os.path.join(dataset_path,'data/SARS-Cov-2', 'test_split.txt')
testimages = open(testfile, 'r').readlines()
newlines = []

for input in tqdm(testimages):
    splits = input.split(' ')
    path = splits[1][2:]
    newpath = dataset_path + path
    newlines.append(f"{splits[0]} {newpath} {splits[2]} {splits[3]}")

with open(testfile, 'w') as f:
    f.writelines(newlines)
