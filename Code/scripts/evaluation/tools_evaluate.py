import sys
sys.path.append("../Code")
import numpy as np



def downsampling(array, fs_sub):
    downsampling_factor = int(500 / fs_sub)
    array_sub = array[::downsampling_factor]
    return array_sub

def normalize_array(array, high=1, low=-1, axis_n=0):
    mins = np.min(array, axis=axis_n)
    maxs = np.max(array, axis=axis_n)
    rng = maxs - mins
    if axis_n == 1:
        array = array.T
    norm_array = high - (((high - low) * (maxs - array)) / rng)
    if axis_n == 1:
        norm_array = norm_array.T
    return norm_array

