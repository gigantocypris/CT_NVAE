"""
Creates synthetic dataset of foam objects
"""

import argparse
import numpy as np
import xdesign as xd 

def create_dataset(num_train, 
                   SIZE_UPPER, SIZE_LOWER, N_PIXEL, GAP,
                   save_name='foam_train.npy'):
    x_train = []
    for i in range(num_train):
        phantom = xd.Foam(size_range=[SIZE_UPPER, SIZE_LOWER], gap=GAP, porosity=np.random.rand())
        discrete = xd.discrete_phantom(phantom, N_PIXEL)
        x_train.append(discrete)
        print(i)
    x_train = np.stack(x_train, axis=0) # shape is num_train x N_PIXEL x N_PIXEL
    np.save(save_name, x_train)
    del x_train

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('-t', dest = 'num_train', type=int, help='number of training examples', default=64)
    parser.add_argument('-v', dest = 'num_valid', type=int, help='number of validation examples', default=16)
    args = parser.parse_args()
    
    ### INPUTS ###

    N_PIXEL = 128 # size of each phantom is N_PIXEL x N_PIXEL

    # parameters to generate the foam phantoms
    SIZE_LOWER = 0.01
    SIZE_UPPER = 0.2
    GAP = 0

    num_train = args.num_train # number of phantoms created for training
    num_valid = args.num_valid # number of phantoms created for validation

    ### END OF INPUTS ###

    np.random.seed(0)

    create_dataset(num_train, 
                    SIZE_UPPER, SIZE_LOWER, N_PIXEL, GAP,
                    save_name='foam_train.npy')

    create_dataset(num_valid, 
                    SIZE_UPPER, SIZE_LOWER, N_PIXEL, GAP,
                    save_name='foam_valid.npy')

    