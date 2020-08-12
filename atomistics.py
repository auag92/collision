import ase
import numpy as np
import scipy.io as sio
import ase.io as aio
from scipy.ndimage import measurements,morphology
import numba
from numba import njit, jit
from scipy.spatial import cKDTree

try:
    import pyfftw
    np.fftpack = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
except ImportError:
    print("you can install PyFFTW for speed-up as - ")
    print("conda install -c conda-forge pyfftw")
    pass

import scipy.ndimage.morphology as morph
from toolz.curried import curry
from toolz.curried import pipe

fft = curry(np.fft.fft)  # pylint: disable=invalid-name

ifft = curry(np.fft.ifft)  # pylint: disable=invalid-name

fftn = curry(np.fft.fftn)  # pylint: disable=invalid-name

ifftn = curry(np.fft.ifftn)  # pylint: disable=invalid-name

fftshift = curry(np.fft.fftshift)  # pylint: disable=invalid-name

ifftshift = curry(np.fft.ifftshift)

conj = curry(np.conj)

func = curry(lambda x, y: conj(x) * fftn(y))

# fabs = curry(lambda x: abs(x))

fabs = curry(lambda x: np.absolute(x))

@curry
def get_scaled_positions(coords, cell, pbc, wrap=True):
    """Get positions relative to unit cell.
    If wrap is True, atoms outside the unit cell will be wrapped into
    the cell in those directions with periodic boundary conditions
    so that the scaled coordinates are between zero and one."""

    fractional = np.linalg.solve(cell.T, coords.T).T

    if wrap:
        for i, periodic in enumerate(pbc):
            if periodic:
                fractional = np.mod(fractional, 1.0)
    return fractional

@curry
def get_real_positions(coords, cell):
    """Return real space coordinates
    given fractional coordiantes and
    cell parameters"""
    return np.dot(cell.T, coords.T).T


def sphere(r=10):
    """
    args: radius of the sphere

    returns: A 3D cubic matric of dim (2*r+1)^1
    """
    return pipe(2*r+1,
                lambda x: np.mgrid[:x,:x,:x],
                lambda xx: (xx[0]-r)**2 + (xx[1]-r)**2+(xx[2]-r)**2,
                lambda x: (x<r*r)*1)


@curry
def imfilter(x_data, f_data):
    """
    to convolve f_data over x_data
    """

    return pipe(f_data,
                ifftshift,
                fftn,
                lambda x: conj(x)*fftn(x_data),
                ifftn,
                fabs)

@curry
def AtomCenters(coords, box, len_pixel):
    """
    Args: Coordinates of all atoms [ndArray], dimension of box, pixel size

    returns: Atom Center location voxels
    """

    atom_centers = np.zeros(coords.shape)
    for dim in range(3):
        atom_centers[:,dim] = pipe(coords[:,dim],
                                   lambda x: np.around(x*len_pixel))
    return atom_centers


@curry
def padder(inp, shape, const_val=0):
    """
    args :  input matrix, new shape

    returns : matrix reshaped to given shape
    """
    ls = np.floor((shape - inp.shape) / 2).astype(int)
    hs = np.ceil((shape - inp.shape) / 2).astype(int)
    return np.pad(inp, ((ls[0], hs[0]), (ls[1], hs[1]), (ls[2], hs[2])), 'constant', constant_values=const_val)


@curry
def return_slice(x_data, cutoff):

    s  = (np.asarray(x_data.shape) // 2 + 1).astype(int)
    cutoff = (np.asarray(cutoff) // 2 + 1).astype(int)

    if x_data.ndim == 2:
        return x_data[(s[0] - cutoff[0]):(s[0] + cutoff[0]),
                      (s[1] - cutoff[1]):(s[1] + cutoff[1])]
    elif x_data.ndim ==3:
        return x_data[(s[0] - cutoff[0]):(s[0] + cutoff[0]),
                      (s[1] - cutoff[1]):(s[1] + cutoff[1]),
                      (s[2] - cutoff[2]):(s[2] + cutoff[2])]
    else:
        print('Incorrect Number of Dimensions!')

@curry
def grid_maker_fft(atom, len_pixel, atomSi, atomOx, full=True):

    dgnls = atom.cell.diagonal()
    coords = pipe(atom,
                  lambda x: x.get_positions(),
                  lambda x: np.mod(x, dgnls),
                  lambda x: x - x.min(axis=0))

    box_dim  = np.ceil((coords.max(axis=0)) * len_pixel).astype(int) + 1

    atom_ids = np.array(atom.get_chemical_symbols())
    idx_Ox = np.where(atom_ids == 'O')[0]
    idx_Si = np.where(atom_ids == 'Si')[0]

    atom_centers = AtomCenters(coords, box_dim, len_pixel)
    x , y, z = [atom_centers[:,dim].astype(int) for dim in range(3)]

    S_Ox, S_Si = np.zeros(box_dim), np.zeros(box_dim)

    S_Ox[x[idx_Ox], y[idx_Ox], z[idx_Ox]] = 1
    S_Si[x[idx_Si], y[idx_Si], z[idx_Si]] = 1

    if full:
        scaler = [len_pixel * (2*r_Si+1)] * 3
    else:
        scaler = 0.0
    scaled_box_dim = (box_dim + scaler)

    atomSi = padder(atomSi, scaled_box_dim)
    atomOx = padder(atomOx, scaled_box_dim)

    S_Ox = padder(S_Ox, scaled_box_dim)
    S_Si = padder(S_Si, scaled_box_dim)

    S_Ox = (imfilter(S_Ox, atomOx) > 0.0001) * 1
    S_Si = (imfilter(S_Si, atomSi) > 0.0001) * 1

    S    = ((S_Ox + S_Si) < 0.1) * 1

    return S, S_Ox, S_Si, box_dim

@curry
def grid_maker_edt(atom, len_pixel, r_Si, r_Ox, full=True):
    dgnls = atom.cell.diagonal()
    coords = pipe(atom,
                  lambda x: x.get_positions(),
                  lambda x: np.mod(x, dgnls),
                  lambda x: x - x.min(axis=0))
    box_dim  = np.ceil((coords.max(axis=0)) * len_pixel).astype(int) + 1

    atom_ids = np.array(atom.get_chemical_symbols())
    idx_Ox = np.where(atom_ids == 'O')[0]
    idx_Si = np.where(atom_ids == 'Si')[0]

    atom_centers = AtomCenters(coords, box_dim, len_pixel)
    x , y, z = [atom_centers[:,dim].astype(int) for dim in range(3)]

    S_Ox, S_Si = np.ones(box_dim), np.ones(box_dim)

    S_Ox[x[idx_Ox], y[idx_Ox], z[idx_Ox]] = 0
    S_Si[x[idx_Si], y[idx_Si], z[idx_Si]] = 0

    if full:
        scaler = [len_pixel * (2*r_Si+1)] * 3
        scaled_box_dim = (box_dim + scaler)
        S_Ox = padder(S_Ox, scaled_box_dim, 1)
        S_Si = padder(S_Si, scaled_box_dim, 1)
    else:
        scaler = 0.0

    S_Ox = morph.distance_transform_edt(S_Ox) / len_pixel
    S_Si = morph.distance_transform_edt(S_Si) / len_pixel

    S_Ox = (S_Ox < r_Ox) * 1
    S_Si = (S_Si < r_Si) * 1

    S    = ((S_Ox + S_Si) < 0.1) * 1
    return S, S_Ox, S_Si, np.array(box_dim)

@curry
def accessibleRegion(S, atomH, r_h, overlap=0.05):
    vol_h = 4/3 * np.pi * r_h**3
    S_mod = imfilter(x_data=(S<0.0001),
                     f_data = padder(atomH,
                                     np.array(S.shape))) / vol_h
    S_mod = S_mod < overlap
    return S_mod

@curry
def return_labelled(x_data):
    S_l, n_count = measurements.label(x_data)
    top = list(np.unique(S_l[:,:,0]))[1:]
    bot = list(np.unique(S_l[:,:,-1]))[1:]
    m = list(set(top).intersection(bot))
    return S_l, n_count, m

@curry
def get_pld_old(s, len_pixel=10):
    s1 = morph.distance_transform_edt(s) / len_pixel
    rds = np.arange(0.5, 9.51, 0.5)
    for r in rds:
        S_mod = np.zeros(s1.shape)
        S_mod[s1 > r] = 1
        S_l, n_count, m = return_labelled(S_mod)
        if len(m) is 0:
            for r1 in np.arange(r-0.5, r, 0.01):
                S_mod = np.zeros(s1.shape)
                S_mod[s1 > r1] = 1
                S_l, n_count, m = return_labelled(S_mod)
                if len(m) is not 0:
                    continue
                else:
                    pld = 2 * r1
                    break
            break
        else:
            continue
    return pld


def is_connected(S):
        S_l, n_count, m = return_labelled(S)
        if len(m) is 0:
            return False
        else:
            return True
@curry
def get_pld(s, len_pixel=10):
    s1 = morph.distance_transform_edt(s) / len_pixel
    rds = np.arange(0.5, 9.51, 0.5)
    pld = 0.0
    for r0 in rds:
        S_mod = np.zeros(s1.shape)
        S_mod[s1 >= r0] = 1
        if not is_connected(S_mod):
            for r1 in np.arange(r0-0.50, r0+0.01, 0.1):
                S_mod = np.zeros(s1.shape)
                S_mod[s1 >= r1] = 1
                if not is_connected(S_mod):
                    for r2 in np.arange(r1-0.10, r1+0.01, 0.01):
                        S_mod = np.zeros(s1.shape)
                        S_mod[s1 >= r2] = 1
                        if is_connected(S_mod):
                            pld = 2 * r2
                            continue
                        else:
                            break
                else:
                    continue
            break
        else:
            continue
    return pld


@curry
def get_lcd(s, len_pixel=10):
    s1 = morph.distance_transform_edt(s) / len_pixel
    lcd = 2 * s1.max()
    return lcd
