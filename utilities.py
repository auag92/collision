"""
@author: ashanker9
"""

import numba
import pickle
import numpy as np
from toolz.curried import curry
import matplotlib.pyplot as plt


@curry
def save_file(fname, obj):
    """
    save python object as a pickle
    """
    with open(fname, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def load_file(fname):
    """
    load python object from pickle file
    """
    with open(fname, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def rgb2gray(rgb):
    """
    convert a 3 chanel rgb image to grey scale
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


@numba.njit()
def get_image(arr, cx, cy, pts):
    for ix in range(len(pts)):
        arr[cx[ix], cy[ix]] += pts[ix]
    return arr


def draw_im(im, title=None):
    """
    plot a 2d image
    """
    
    plt.imshow(im, cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()

    
@curry
def return_slice(x_data, cutoff=5, s=None):
    """
    returns region of interest around the center voxel 
    upto the cutoff length
    """
    
    if not s:
        s = np.asarray(x_data.shape).astype(int) // 2
    
    if x_data.ndim == 2:
        return x_data[(s[0] - cutoff):(s[0] + cutoff+1),
                      (s[1] - cutoff):(s[1] + cutoff+1)]
    elif x_data.ndim ==3:
        return x_data[(s[0] - cutoff):(s[0] + cutoff+1),
                      (s[1] - cutoff):(s[1] + cutoff+1),
                      (s[2] - cutoff):(s[2] + cutoff+1)]