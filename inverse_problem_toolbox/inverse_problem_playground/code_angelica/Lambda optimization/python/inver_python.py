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

y = signal.copy()
idx = electrodes.T

lap, edge = mesh_laplacian(vertices_tank, faces_tank-1)
