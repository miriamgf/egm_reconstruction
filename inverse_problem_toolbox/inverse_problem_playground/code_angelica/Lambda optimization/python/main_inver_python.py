#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:04:38 2025

Inverse problem on data from Angelica/Joao

@author: obarquero
"""

#%% functions definitions move to a file

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def mesh_laplacian(vertex, face):
    """
    Calculate the Laplacian matrix for a mesh with vertices and faces.
    
    Args:
        vertex (ndarray): Array of vertex coordinates (nvertex x 3).
        face (ndarray): Array of face indices (nface x 3).
    
    Returns:
        lap (csr_matrix): Sparse Laplacian matrix (nvertex x nvertex).
        edge (csr_matrix): Sparse edge connectivity matrix (nvertex x nvertex).
    """
    nvertex = vertex.shape[0]
    nface = face.shape[0]
    
    print(f"MESH_LAPLACIAN: Calculating Laplacian matrix for {nvertex} vertices...")
    
    # Initialize the edge matrix
    edge = lil_matrix((nvertex, nvertex), dtype=np.float64)
    
    # Calculate edge lengths and populate edge matrix
    for i in range(nface):
        # Compute the length of all triangle edges
        Diff = vertex[face[i, [0, 1, 2]], :] - vertex[face[i, [1, 2, 0]], :]
        Norm = np.sqrt(np.sum(Diff ** 2, axis=1))
        
        edge[face[i, 0], face[i, 1]] = Norm[0]
        edge[face[i, 1], face[i, 2]] = Norm[1]
        edge[face[i, 2], face[i, 0]] = Norm[2]
        
        # Make edges symmetric
        edge[face[i, 1], face[i, 0]] = Norm[0]
        edge[face[i, 2], face[i, 1]] = Norm[1]
        edge[face[i, 0], face[i, 2]] = Norm[2]
    
    # Initialize the Laplacian matrix
    lap = lil_matrix((nvertex, nvertex), dtype=np.float64)
    
    # Calculate Laplacian matrix
    for i in range(nvertex):
        # Indices of neighbors
        k = edge[i, :].nonzero()[1]
        ni = len(k)  # Number of neighbors
        
        if ni > 0:
            hi = np.mean(edge[i, k].toarray())  # Average distance to neighbors
            invhi = np.mean(1.0 / edge[i, k].toarray())  # Average inverse distance
            
            # Laplacian of the vertex itself
            lap[i, i] = -(4 / hi) * invhi
            
            # Laplacian of direct neighbors
            lap[i, k] = (4 / (hi * ni)) * (1.0 / edge[i, k].toarray())
    
    # Convert to sparse matrix format for efficiency
    edge = csr_matrix(edge)
    lap = csr_matrix(lap)
    
    print("MESH_LAPLACIAN: Calculation complete.")
    return lap, edge

import numpy as np
import scipy.sparse as sp

def mesh_laplacian_current(lap, index):
    """
    Computes the zero Laplacian interpolation matrix.

    Parameters:
        lap (ndarray or sparse matrix): The Laplacian matrix for the full mesh.
        index (ndarray): Row vector of indices into a subset of vertices (known potentials).

    Returns:
        int (ndarray): Interpolation matrix.
        keepindex (ndarray): Indices of unique known points.
        repindex (ndarray): Indices of duplicate entries in the input index.
    """
    # Check if the Laplacian matrix is square
    if lap.shape[0] != lap.shape[1]:
        raise ValueError("Laplacian matrix must be square.")
    
    # Convert sparse matrix to dense if necessary
    if sp.issparse(lap):
        lap = lap.toarray()
    
    # Ensure index is a row vector
    index = np.asarray(index).flatten()

    # Remove duplicates from index
    KnownIndex, ia, ib = np.unique(index, return_index=True, return_inverse=True)
    if len(KnownIndex) != len(index):
        print("\nWarning: Trimming duplicate values from index. Use keepindex for unique indices.\n")
    
    keepindex = index
    repindex = np.setdiff1d(np.arange(len(index)), np.sort(ib))

    # Sort KnownIndex
    KnownIndex = np.sort(index)

    k = len(KnownIndex)
    n = lap.shape[0]
    
    print(k)
    print(n)

    print(f"Calculating interpolation matrix for {k} known vertices to {n} total vertices...")

    # Find 'unknown' indices
    UnknownIndex = np.setdiff1d(np.arange(n), KnownIndex)

    # Reshuffle rows and columns of the Laplacian matrix
    lapi = np.concatenate([KnownIndex, UnknownIndex])
    lap = lap[lapi, :][:, lapi]

       # Segregate known/unknown portions of lap
    # L11: known/known part (k x k)
    L11 = lap[KnownIndex, :][:, KnownIndex]
    
    # L12: known/unknown part (k x (n-k))
    L12 = lap[KnownIndex, :][:, k:n]
    
    # L21: unknown/known part ((n-k) x k)
    L21 = lap[k:n, :][:, KnownIndex]
    
    # L22: unknown/unknown part ((n-k) x (n-k))
    L22 = lap[k:n, :][:, k:n]
    
    # Debug print to check the shapes
    print(f"L11 shape: {L11.shape}, L12 shape: {L12.shape}, L21 shape: {L21.shape}, L22 shape: {L22.shape}")
    
    # Ensure L12 and L22 have compatible row dimensions, and L11 and L21 have compatible column dimensions
    assert L12.shape[0] == L22.shape[0], f"Row dimensions of L12 and L22 do not match: {L12.shape[0]} != {L22.shape[0]}"
    assert L11.shape[1] == L21.shape[1], f"Column dimensions of L11 and L21 do not match: {L11.shape[1]} != {L21.shape[1]}"
    # Convert to sparse for quicker computation (note: L12 and L22 are sparse)
    A = sp.vstack([L12, L22])  # Stacking rows in Python (equivalent to [L12; L22] in MATLAB)
    B = sp.hstack([L11, L21])  # Stacking columns in Python (equivalent to [L11, L21] in MATLAB)
    
    # Check dimensions after stacking
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    
    

    # Create sparse matrices for computation
    #A = sp.csr_matrix(np.vstack([L12, L22]))
    #B = sp.csr_matrix(np.vstack([L11, L21]))
    
    # Solve for the interpolation matrix
    int_part = -sp.linalg.spsolve(A, B).toarray()

    # Append identity matrix for known potentials
    int_matrix = np.vstack([np.eye(k), int_part])

    # Reshuffle the columns of the interpolation matrix
    order = np.argsort(KnownIndex)
    int_matrix = int_matrix[:, order]

    # Reshuffle the rows of the interpolation matrix
    total_order = np.argsort(np.concatenate([KnownIndex, UnknownIndex]))
    int_matrix = int_matrix[total_order, :]

    # Convert to sparse for efficiency
    int_matrix = sp.csr_matrix(int_matrix)

    print("Interpolation matrix computation completed.")

    return int_matrix, keepindex, repindex


"""
Examples

# Example vertices and faces
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
faces = np.array([
    [0, 1, 2],
    [0, 1, 3]
])

lap, edge = mesh_laplacian(vertices, faces)

print("Laplacian matrix:\n", lap.toarray())
print("Edge matrix:\n", edge.toarray()) convert this to matlab


-----------------
"""
"""
import numpy as np
from scipy.sparse import csr_matrix

# Example Laplacian matrix and index
lap = csr_matrix([
    [3, -1, -1, -1],
    [-1, 2, 0, -1],
    [-1, 0, 2, -1],
    [-1, -1, -1, 3]
])
index = np.array([0, 2, 2])

# Compute interpolation matrix
int_matrix, keepindex, repindex = mesh_laplacian_current(lap, index)

print("Interpolation Matrix:\n", int_matrix.toarray())
print("Keep Index:", keepindex)
print("Rep Index:", repindex)

"""



#%% Read data

#files names
signal_file = "../01 - data/electric_data_Exx_Fxx_Rxx_filtered.mat"
electrodes_idx_file = '../01 - data/eletrodos_LR.mat'
heart_geo_file = "../01 - data/../../01 - data/heart_geometry_20000_exp14.mat"
tank_geo_file = "../01 - data/tank_geometry.mat"
mtransfer_file = "../01 - data/MTransfer_exp14_LR_20000.mat"

#read data

import h5py
import numpy as np
import scipy.io
# Reading Files

# Extracting tank signals
with h5py.File(signal_file, 'r') as f:
    # Assuming the first key contains the needed dataset
    #first_key = list(f.keys())[0]
    first_key = 'D_EL'
    signal_data = f[first_key]
    # Accessing the 'Data' field and selecting rows
    raw_signal= signal_data['Data']
    # Keeping only 60 electrodes (adjust for zero-based indexing)
    r_signal = raw_signal[:,:].T
    signal = raw_signal[:,np.r_[128:174, 176:190]].T  # Append additional rows
   


tank_geo = scipy.io.loadmat(tank_geo_file)['tank_geo']
faces_tank = tank_geo[0][0][0]
vertices_tank = tank_geo[0][0][1]

electrodes = scipy.io.loadmat(electrodes_idx_file)['ans']

mtransfer = scipy.io.loadmat(mtransfer_file)['MTransfer']
# Clear unnecessary variables
# In Python, unused variables will be garbage collected, but you can explicitly delete them:
# del heart_geo_file, tank_geo_file, mtransfer_file, signal_file, electrodes_idx_file


# Clear unnecessary variables
del heart_geo_file, tank_geo_file, mtransfer_file, signal_file, electrodes_idx_file
# Optional: del heart_data, tank_data, mtransfer_data, electrodes_data if loaded variables are no longer needed


#%% Interpolate BSPs

#why do you need to do that
"""
y = signal.copy()
idx = electrodes.T

lap, edge = mesh_laplacian(vertices_tank, faces_tank-1)
"""

#load interp signal
interp_signal = scipy.io.loadmat('interp_signal.mat')['interp_signal']

A = mtransfer.copy()



#%% tikhonov

from precompute_matrix import precompute_matrix

AA, L , LL = precompute_matrix(A, None, order = 0)




#%%
#some plottings
import matplotlib.pyplot as plt
from scipy.signal import welch
plt.close('all')
fs = 4e3

plt.figure()
plt.subplot(321)
plt.plot(signal[0,:])
plt.title('bsp')
plt.subplot(322)
plt.plot(r_signal[0,:])
plt.title('r_atria')
plt.subplot(323)
plt.plot(interp_signal[0,:])
plt.title('bsp_interp')
plt.subplot(324)
plt.plot(r_signal[18,:])
plt.title('l_atria')
plt.subplot(325)
plt.plot(signal[40,:])
plt.title('bsp')
plt.subplot(326)
plt.plot(r_signal[66,:])
plt.title('ventricle')

#psd
f, psd = welch(interp_signal[0,:], fs=fs, nperseg=len(interp_signal[0,:]/8))

idx = f<60

plt.figure()
plt.subplot(121)
plt.plot(r_signal[0,:])
plt.subplot(122)
plt.plot(f[idx],psd[idx])



#%%

#perform some filtering
import filtering

y_,_ = filtering.detrendSpline(interp_signal,fs,l_w = 0.2)
#y,_ = filtering.detrendSpline(y,fs,l_w = 0.25)
y_filtered = filtering.ECG_filtering_real(y_, fs,filt_order = 8)
#y_filtered = filtering.ECG_filtering_real(y, fs)

#%%
plt.close('all')
# Loop through each row and plot
for i in range(signal.shape[0]):    
    plt.figure(figsize=(20, 10))
    
    f, psd = welch(interp_signal[i,:], fs=fs, nperseg=len(interp_signal[i,:]/8))
    f, psd_f = welch(y_filtered[i,:], fs=fs, nperseg=len(interp_signal[i,:]/8))

    idx = f<60
    
    plt.subplot(121)
    plt.plot(interp_signal[i, :], label=f'Signal {i}',linewidth = 0.3, alpha = 0.8)
    #plt.plot(x_[i,:],label='detrended')
    plt.plot(y_filtered[i,:],label='filtered')
    plt.title(f'Row {i + 1}')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(f[idx],psd[idx], label = f'PSD {i}')
    plt.plot(f[idx],psd_f[idx], label = f'PSD_filtered {i}')
    plt.xlabel('f[Hz]')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(True)
    
    plt.show(block=False)
    
    
    
    print(f"Displaying row {i + 1}. Close the plot and press any key to continue.")
    plt.waitforbuttonpress()  # Wait for a key press
    plt.close() 


#%% Several options to perform the tikhonov

#%%1. Downsampling

from scipy.signal import resample_poly

# Downsampling factor
factor = 16

# Downsample using resample_poly
y_filt_down = resample_poly(y_filtered.T, up=1, down=factor).T

fs_d = fs/factor

print(f"Original signal length: {y_filtered.shape}")
print(f"Downsampled signal length: {y_filt_down.shape}")
print(f"fs {fs_d}")


#Tikhonov regularization
import forward_inverse_problem as fip

#time estimations
est_on = 2 
est_off = 3.5

samples_on = int(est_on*fs)
samples_off = int(est_off*fs)

#x_hat,lambda_opt, lambdas_,errors_,magnitude_term_,lambda_opt_= fip.classical_tikhonov(A, AA, L, LL, y_filtered[:,samples_on:samples_off],n_iterations = 3)

#x_hat,lambda_opt,magnitude_term,error_term,maxcurve_index = fip.classical_tikhonov_noiter(A,AA,L,LL,y_filtered[:,samples_on:samples_off],size_chunk=400)

#add noise to measurementes:
scale = 0.005 * np.max(np.abs(y_filtered))
# Generate Gaussian noise
noise = np.random.normal(0, scale, y_filtered.shape)
# Add noise to the signal
y_filtered_n = y_filtered + noise


import os
print("primer método")
x_hat_1,lambda_opt_1,magnitude_term_1,error_term_1,maxcurve_index_1 = fip.classical_tikhonov_noiter_global(A,AA,L,LL,y_filtered[:,samples_on:samples_off],positive_curvature_only = True)
saved_data = {
    "x_hat": x_hat_1,
    "lambda_opt": lambda_opt_1,
    "magnitude_term": magnitude_term_1,
    "error_term": error_term_1,
    "maxcurve_index": maxcurve_index_1,
}

np.save("tikh_1",saved_data)

os.system('clear')
print("segundo método")
#x_hat_2,lambda_opt_2,magnitude_term_2,error_term_2,maxcurve_index_2 = fip.classical_tikhonov_noiter_global(A,AA,L,LL,y_filtered_n[:,samples_on:samples_off],positive_curvature_only = True)

os.system('clear')
print("segundo método")
x_hat_3,lambda_opt_3,magnitude_term_3,error_term_3,maxcurve_index_3 = fip.classical_tikhonov_noiter_global(A,AA,L,LL,y_filtered[:,samples_on:samples_off])
saved_data_2 = {
    "x_hat": x_hat_3,
    "lambda_opt": lambda_opt_3,
    "magnitude_term": magnitude_term_3,
    "error_term": error_term_3,
    "maxcurve_index": maxcurve_index_3,
}

np.save("tikh_2",saved_data_2)



# %%




# %%
