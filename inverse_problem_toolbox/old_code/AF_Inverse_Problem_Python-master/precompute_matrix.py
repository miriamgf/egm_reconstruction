# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:17:08 2018

@author: Miguel Ángel
"""

import numpy as np
from scipy import signal as sigproc
import data_load as dl

import os
import scipy.io
import h5py

data_path = os.getcwd() + '/data/';

def precompute_matrix(A,atrial_model,order=0):
    """
    Function which applies narrow-band filter to x and calculates variables for 
    minimizing computing time in calculation of inverse matrix A.
    
    Parameters:
        A (matrix): transfer matrix
        atrial_model (dict): atrial model
        order (int): Tikhonov order (0, 1 or 2).
    Returns:
        AA (matrix): A'A
        L (matrix): matrix which takes part in regularization term of Tikhonov approach
        LL (matrix): L'L
    """
    # In zero-order Tikhonov approach: L matrix is the identity matrix.
    if order == 0:
        L=np.eye(A.shape[1],A.shape[1])
        
    else:
        # Load file with g1 and g2 L matrix.
        if order == 1:
            try:
                L = np.transpose(np.array((h5py.File(data_path + '/L_matrix.mat','r')).get('L_g1')))
            except:
                L = scipy.io.loadmat(data_path + '/L_matrix.mat').get('L_g1')
        elif order == 2:
            try:
                L = np.transpose(np.array((h5py.File(data_path + '/L_matrix.mat','r')).get('L_g2')))
            except:
                L = scipy.io.loadmat(data_path + '/L_matrix.mat').get('L_g2')
    
    AA=np.matmul(np.transpose(A),A)
    LL=np.matmul(np.transpose(L),L)
    
    return AA,L,LL

def D_nodes_constrained(n_nodes='basket'):
    """
    Returns diagonal matrix with selected available nodes for Constrained Tikhonov
 
    Parameters:
        n_nodes: number of nodes to select (default: clinical-based basket)
    Returns:
        D (matrix): diagonal matrix with selected available nodes.
        known_nodes (matrix): selected nodes.
    """
    dD=np.zeros(2048)

    if isinstance(n_nodes,str) and n_nodes=='basket':
        known_nodes=dl.load_constrained_nodes()-1
    else: 
        if np.size(n_nodes)==1 and n_nodes!=0:
            Delta_x_epi=np.floor(2048/n_nodes);
            known_nodes=np.uint16(np.arange(0,np.size(dD)-1,Delta_x_epi))
        elif np.size(n_nodes)>1:
            known_nodes=n_nodes;
            
    dD[known_nodes]=1
    D = np.diag(dD);
            
    return D, known_nodes