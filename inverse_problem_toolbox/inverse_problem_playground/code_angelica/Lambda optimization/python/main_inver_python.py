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
factor = 18

# Downsample using resample_poly
y_filt_down = resample_poly(y_filtered.T, up=1, down=factor).T

fs_d = fs/factor

print(f"Original signal length: {y_filtered.shape}")
print(f"Downsampled signal length: {y_filt_down.shape}")
print(f"fs {fs_d}")


#%%
#Tikhonov regularization
import forward_inverse_problem as fip

#time estimations
est_on = 2 
est_off = 3

samples_on = int(est_on*fs_d)
samples_off = int(est_off*fs_d)

#x_hat,lambda_opt, lambdas_,errors_,magnitude_term_,lambda_opt_= fip.classical_tikhonov(A, AA, L, LL, y_filtered[:,samples_on:samples_off],n_iterations = 3)

#x_hat,lambda_opt,magnitude_term,error_term,maxcurve_index = fip.classical_tikhonov_noiter(A,AA,L,LL,y_filtered[:,samples_on:samples_off],size_chunk=400)

#add noise to measurementes:
scale = 0.005 * np.max(np.abs(y_filtered))
# Generate Gaussian noise
noise = np.random.normal(0, scale, y_filtered.shape)
# Add noise to the signal
y_filtered_n = y_filtered + noise


#%%
import os
import pickle 
run_tikh = False

if run_tikh:
    print("primer método")
    lambda_test = np.logspace(-0.5,-12,10)
    x_hat,lambda_opt,magnitude_term,error_term,maxcurve_index = fip.classical_tikhonov_noiter_global(A,AA,L,LL,y_filt_down[:,samples_on:samples_off],positive_curvature_only = True,lambda_test = lambda_test)
    
    lambda_test = np.logspace(-0.5,-12,10)
    y_filtered_norm = y_filt_down/ np.max(np.abs(y_filt_down),axis = 1, keepdims=True)
    x_hat_norm,lambda_opt_norm,magnitude_term_norm,error_term_norm,maxcurve_index_norm = fip.classical_tikhonov_noiter_global(A,AA,L,LL,y_filtered_norm[:,samples_on:samples_off],positive_curvature_only = False,lambda_test = lambda_test)
    
    import pickle
    
    # Save atria-related outputs (positive curvature)
    with open('output.pkl', 'wb') as atria_file:
        pickle.dump((x_hat, lambda_opt, magnitude_term, error_term, maxcurve_index), atria_file)
    
    # Save ventricles-related outputs (normalized data, positive curvature off)
    with open('output_norm.pkl', 'wb') as ventricles_file:
        pickle.dump((x_hat_norm, lambda_opt_norm, magnitude_term_norm, error_term_norm, maxcurve_index_norm), ventricles_file)
    
    print("Files saved as 'output.pkl' and 'output_norm.pkl'")
    
else:
    # Load atria-related outputs (positive curvature)
    with open('output.pkl', 'rb') as atria_file:
        atria_outputs = pickle.load(atria_file)
    
    # Load ventricles-related outputs (normalized data, positive curvature off)
    with open('output_norm.pkl', 'rb') as ventricles_file:
        ventricles_outputs = pickle.load(ventricles_file)
    
    # Access individual outputs for atria
    x_hat, lambda_opt, magnitude_term, error_term, maxcurve_index = atria_outputs
    
    # Access individual outputs for ventricles
    x_hat_norm, lambda_opt_norm, magnitude_term_norm, error_term_norm, maxcurve_index_norm = ventricles_outputs


"""saved_data = {
    "x_hat": x_hat_1,
    "lambda_opt": lambda_opt_1,
    "magnitude_term": magnitude_term_1,
    "error_term": error_term_1,
    "maxcurve_index": maxcurve_index_1,
}


np.save("tikh_2",saved_data_2)
"""

plt.figure()
plt.plot(np.log(error_term),np.log(magnitude_term),'.-',label = "L_curve pca atria")
plt.plot(np.log(error_term)[maxcurve_index],np.log(magnitude_term)[maxcurve_index],'rX')

plt.figure()
plt.plot(np.log(error_term_norm),np.log(magnitude_term_norm),'.-',label = "L_curve pca atria")
plt.plot(np.log(error_term_norm)[maxcurve_index_norm],np.log(magnitude_term_norm)[maxcurve_index_norm],'rX')



#%%

from sklearn.decomposition import PCA
#PCA in y

row_max = np.max(np.abs(y_filt_down), axis=1, keepdims=True)
#row_max[row_max == 0] = 1  # Avoid division by zero
y_normalized = y_filt_down / row_max

# Perform PCA
pca = PCA(n_components=y_filt_down.shape[0])
transformed = pca.fit_transform(y_normalized)  # Scores (projection onto principal components)
components = pca.components_  # Principal components (time patterns)

# Visualize explained variance
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance (%)')
plt.grid()
plt.show()

# Visualize principal components
fig, axes = plt.subplots(25, 1, figsize=(12, 8), sharex=True)

t = np.arange(len(r_signal[0,:]))/4e3
tt = np.arange(len(components[0]))/fs_d
for i, ax in enumerate(axes):
    if i==0:
        ax.plot(t,r_signal[0,:],color='blue')
        ax.set_title('Atrial signal')
        ax.axis('off')
    elif i==1:
        ax.plot(t,r_signal[26,:],color='blue')
        ax.set_title('Ventricle signal')
        ax.axis('off')
        
    else: 
        ax.plot(tt,components[i], color='blue')
        ax.set_title(f'Component {i -2}', fontsize=10)
        ax.axis('off')  # Remove axes for better visualization

plt.tight_layout()
plt.show()


#%% reconstruction
# Allow user to select components interactively or specify indices
print(f"Total Components: {components.shape[0]}")
selected_components = input("Enter the indices of components to retain (e.g., 0,1,2): ")
selected_indices = list(map(int, selected_components.split(',')))

all_indices = np.arange(components.shape[0])  # All component indices
not_selected_indices = np.setdiff1d(all_indices, selected_indices)

# Reconstruction with selected components
selected_scores = transformed[:, selected_indices]
selected_basis = components[selected_indices, :]
reconstructed_selected = np.dot(selected_scores, selected_basis)

# Reconstruction with non-selected components
not_selected_scores = transformed[:, not_selected_indices]
not_selected_basis = components[not_selected_indices, :]
reconstructed_not_selected = np.dot(not_selected_scores, not_selected_basis)

# Scale back both reconstructions to the original scale
reconstructed_selected *= row_max
reconstructed_not_selected *= row_max

#%%
#correlate components with atria and ventricle signals

components_norm = components / np.max(np.abs(components), axis=1, keepdims=True)
atria_norm = r_signal[0,:]/ np.max(np.abs(r_signal[0,:]))
ventricle_norm = r_signal[26,:]/ np.max(np.abs(r_signal[28,:]))

atria_norm = resample_poly(atria_norm, up=1, down=factor).T
ventricle_norm = resample_poly(ventricle_norm, up=1, down=factor).T

n_components = components_norm.shape[0]
corr_atria = np.zeros(n_components)
corr_ventricle = np.zeros(n_components)
for i in range(n_components):
    corr_atria[i] = np.corrcoef(components_norm[i], atria_norm)[0, 1]
    corr_ventricle[i] = np.corrcoef(components_norm[i], ventricle_norm)[0, 1]
    

top_atria_comp = np.argsort(corr_atria)
top_ventri_comp = np.argsort(corr_ventricle)


#atria top componente
atri_com = components_norm[top_atria_comp[-3:]]
ventricle_com = components_norm[top_ventri_comp[-3:]]

plt.figure()
plt.plot(atria_norm,label='atria signal')
plt.plot(atri_com[0,:],label='componentes correlated with atria')

plt.figure()

plt.plot(ventricle_norm,label='atria signal')
plt.plot(ventricle_com[0,:],label='componentes correlated with atria')


#%%
#inverse problem with pca

lambda_test = np.logspace(-0.5,-12,10)

run_tikh_pca_selected_atria = False
#Selected atria

if run_tikh_pca_selected_atria:
    #y_filtered_norm = y_filt_down/ np.max(np.abs(y_filt_down),axis = 1, keepdims=True)
    x_hat_pca_atria,lambda_opt_pca_atria,magnitude_term_pca_atria,error_term_pca_atria,maxcurve_index_pca_atria = fip.classical_tikhonov_noiter_global(A,AA,L,LL,reconstructed_selected[:,samples_on:samples_off],positive_curvature_only = False,lambda_test = lambda_test)

    #non selected ventricles
    x_hat_pca_v,lambda_opt_pca_v,magnitude_term_pca_v,error_term_pca_v,maxcurve_index_pca_v = fip.classical_tikhonov_noiter_global(A,AA,L,LL,reconstructed_not_selected[:,samples_on:samples_off],positive_curvature_only = False,lambda_test = lambda_test)
    import pickle

    # Save atria-related outputs
    with open('output_atria.pkl', 'wb') as atria_file:
        pickle.dump((x_hat_pca_atria, lambda_opt_pca_atria, magnitude_term_pca_atria, error_term_pca_atria, maxcurve_index_pca_atria), atria_file)
    
    # Save ventricles-related outputs
    with open('output_ventricles.pkl', 'wb') as ventricles_file:
        pickle.dump((x_hat_pca_v, lambda_opt_pca_v, magnitude_term_pca_v, error_term_pca_v, maxcurve_index_pca_v), ventricles_file)

    print("Files saved as 'output_atria.pkl' and 'output_ventricles.pkl'")

else:
    with open('output_atria.pkl', 'rb') as atria_file:
        atria_outputs = pickle.load(atria_file)

    with open('output_ventricles.pkl', 'rb') as ventricles_file:
        ventricles_outputs = pickle.load(ventricles_file)

    # Access individual outputs
    x_hat_pca_atria, lambda_opt_pca_atria, magnitude_term_pca_atria, error_term_pca_atria, maxcurve_index_pca_atria = atria_outputs
    x_hat_pca_v, lambda_opt_pca_v, magnitude_term_pca_v, error_term_pca_v, maxcurve_index_pca_v = ventricles_outputs

#%%
lambda_test = np.logspace(-0.5,-12,10)

run_tikh_pca_selected_v= False
#Selected atria

if run_tikh_pca_selected_v:
    #y_filtered_norm = y_filt_down/ np.max(np.abs(y_filt_down),axis = 1, keepdims=True)
    x_hat_pca_2_atria,lambda_opt_pca_2_atria,magnitude_term_pca_2_atria,error_term_pca_2_atria,maxcurve_index_pca_2_atria = fip.classical_tikhonov_noiter_global(A,AA,L,LL,reconstructed_not_selected[:,samples_on:samples_off],positive_curvature_only = False,lambda_test = lambda_test)

    #selected ventricles
    x_hat_pca_2_v,lambda_opt_pca_2_v,magnitude_term_pca_2_v,error_term_pca_2_v,maxcurve_index_pca_2_v = fip.classical_tikhonov_noiter_global(A,AA,L,LL,reconstructed_selected[:,samples_on:samples_off],positive_curvature_only = False,lambda_test = lambda_test)
    import pickle

    # Save atria-related outputs
    with open('output_atria_2.pkl', 'wb') as atria_file:
        pickle.dump((x_hat_pca_2_atria, lambda_opt_pca_2_atria, magnitude_term_pca_2_atria, error_term_pca_2_atria, maxcurve_index_pca_2_atria), atria_file)
    
    # Save ventricles-related outputs
    with open('output_ventricles_2.pkl', 'wb') as ventricles_file:
        pickle.dump((x_hat_pca_2_v, lambda_opt_pca_2_v, magnitude_term_pca_2_v, error_term_pca_2_v, maxcurve_index_pca_2_v), ventricles_file)

    print("Files saved as 'output_atria.pkl' and 'output_ventricles.pkl'")

else:
    with open('output_atria_2.pkl', 'rb') as atria_file:
        atria_2_outputs = pickle.load(atria_file)

    with open('output_ventricles_2.pkl', 'rb') as ventricles_file:
        ventricles_2_outputs = pickle.load(ventricles_file)

    # Access individual outputs
    x_hat_pca_2_atria, lambda_opt_pca_2_atria, magnitude_term_pca_2_atria, error_term_pca_2_atria, maxcurve_index_pca_2_atria = atria_outputs
    x_hat_pca_2_v, lambda_opt_pca_2_v, magnitude_term_pca_2_v, error_term_pca_2_v, maxcurve_index_pca_2_v = ventricles_outputs


#%%
plt.figure()
plt.plot(np.log(error_term_pca_atria),np.log(magnitude_term_pca_atria),'.-',label = "L_curve pca atria")
plt.plot(np.log(error_term_pca_atria)[maxcurve_index_pca_atria],np.log(magnitude_term_pca_atria)[maxcurve_index_pca_atria],'rX')


plt.plot(np.log(error_term_pca_v),np.log(magnitude_term_pca_v),'.-',label = "L_curve pca vnetricl")
plt.plot(np.log(error_term_pca_v)[maxcurve_index_pca_v],np.log(magnitude_term_pca_v)[maxcurve_index_pca_v],'rX')


plt.plot(np.log(error_term_pca_2_v),np.log(magnitude_term_pca_2_v),'.-',label = "L_curve pca ventricle_recons2")
plt.plot(np.log(error_term_pca_2_v)[maxcurve_index_pca_2_v],np.log(magnitude_term_pca_2_v)[maxcurve_index_pca_2_v],'rX')


#%%
plt.close('all')
for i in range(0,x_hat_pca_atria.shape[0],50):    
    #plt.figure(figsize=(20, 10))
    plt.figure()
    print(i)
    plt.plot(x_hat_pca_atria[i,:]/np.max(np.abs(x_hat_pca_atria[i,:])),label = 'reconstructed atria')
    plt.plot(x_hat_pca_v[i,:]/np.max(np.abs(x_hat_pca_v[i,:])),label = 'reconstructed v')
    plt.plot(x_hat_pca_2_v[i,:]/np.max(np.abs(x_hat_pca_2_v[i,:])),label = 'reconstructed v 2')
    plt.legend()

    plt.grid(True)
    
    plt.show(block=False)
    
    
    
    print(f"Displaying row {i + 1}. Close the plot and press any key to continue.")
    plt.waitforbuttonpress()  # Wait for a key press
    plt.close() 


#%% 

plt.close('all')
#compare with the vnetricle
t = np.linspace(0,1,4000)
tt = np.arange(x_hat_pca_atria.shape[1])/fs_d
for i in range(0,x_hat_pca_atria.shape[0],50):    
    plt.figure(figsize=(20, 10))
    #plt.figure()
    print(i)
    plt.plot(t,r_signal[20,4000*2:4000*3]/np.max(np.abs(r_signal[20,4000*2:4000*3])),label = 'original')
    plt.plot(tt,x_hat[i,:]/np.max(np.abs(x_hat[i,:])),label = 'reconstructed all')
    plt.plot(tt,x_hat_pca_v[i,:]/np.max(np.abs(x_hat_pca_v[i,:])),label = 'reconstructed v')
    plt.plot(tt,x_hat_pca_2_v[i,:]/np.max(np.abs(x_hat_pca_2_v[i,:])),label = 'reconstructed v 2')
    plt.legend()

    plt.grid(True)
    
    plt.show(block=False)
    
    
    
    print(f"Displaying row {i + 1}. Close the plot and press any key to continue.")
    plt.waitforbuttonpress()  # Wait for a key press
    plt.close() 

#%%

plt.close('all')
#compare with the vnetricle
t = np.linspace(0,1,4000)
tt = np.arange(x_hat_pca_atria.shape[1])/fs_d
for i in range(0,x_hat_pca_atria.shape[0],50):    
    plt.figure(figsize=(20, 10))
    #plt.figure()
    print(i)
    plt.plot(t,r_signal[0,4000*2:4000*3]/np.max(np.abs(r_signal[0,4000*2:4000*3])),label = 'original')
    plt.plot(tt,x_hat[i,:]/np.max(np.abs(x_hat[i,:])),label = 'reconstructed all')
    plt.plot(tt,x_hat_pca_atria[i,:]/np.max(np.abs(x_hat_pca_atria[i,:])),label = 'reconstructed a')
    plt.plot(tt,x_hat_pca_2_atria[i,:]/np.max(np.abs(x_hat_pca_2_atria[i,:])),label = 'reconstructed a 2')
    plt.legend()

    plt.grid(True)
    
    plt.show(block=False)
    
    
    
    print(f"Displaying row {i + 1}. Close the plot and press any key to continue.")
    plt.waitforbuttonpress()  # Wait for a key press
    plt.close() 