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

data_path = os.getcwd() + "/data/"
data_path_real = os.getcwd() + "/data/Real/"


def precompute_matrix(A, atrial_model, order=0):
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
        L = np.eye(A.shape[1], A.shape[1])

    else:
        # Load file with g1 and g2 L matrix.
        if order == 1:
            try:
                L = np.transpose(
                    np.array((h5py.File(data_path + "/L_matrix.mat", "r")).get("L_g1"))
                )
            except:
                L = scipy.io.loadmat(data_path + "/L_matrix.mat").get("L_g1")
        elif order == 2:
            try:
                L = np.transpose(
                    np.array((h5py.File(data_path + "/L_matrix.mat", "r")).get("L_g2"))
                )
            except:
                L = scipy.io.loadmat(data_path + "/L_matrix.mat").get("L_g2")

    AA = np.matmul(np.transpose(A), A)
    LL = np.matmul(np.transpose(L), L)

    return AA, L, LL


def precompute_matrix_real(A, atrial_model, model, order=0):
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
        L = np.eye(A.shape[1], A.shape[1])
    elif order == 1:
        # Load file with g1 and g2 L matrix.
        try:
            L = np.transpose(
                np.array(
                    (h5py.File(data_path_real + model + "/L_matrix.mat", "r")).get(
                        "L_consg1"
                    )
                )
            )
        except:
            L = scipy.io.loadmat(data_path_real + model + "/L_matrix.mat").get(
                "L_consg1"
            )
    elif order == 2:
        try:
            L = np.transpose(
                np.array(
                    (h5py.File(data_path_real + model + "/L_matrix.mat", "r")).get(
                        "L_consg2"
                    )
                )
            )
        except:
            L = scipy.io.loadmat(data_path_real + model + "/L_matrix.mat").get(
                "L_consg2"
            )

    AA = np.matmul(np.transpose(A), A)
    LL = np.matmul(np.transpose(L), L)

    return AA, L, LL


def D_nodes_constrained(n_nodes="basket"):
    """
    Returns diagonal matrix with selected available nodes for Constrained Tikhonov

    Parameters:
        n_nodes: number of nodes to select (default: clinical-based basket)
    Returns:
        D (matrix): diagonal matrix with selected available nodes.
        known_nodes (matrix): selected nodes.
    """
    dD = np.zeros(2048)

    if isinstance(n_nodes, str) and n_nodes == "basket":
        known_nodes = dl.load_constrained_nodes() - 1
    else:
        if np.size(n_nodes) == 1 and n_nodes != 0:
            Delta_x_epi = np.floor(2048 / n_nodes)
            known_nodes = np.uint16(np.arange(0, np.size(dD) - 1, Delta_x_epi))
        elif np.size(n_nodes) > 1:
            known_nodes = n_nodes

    dD[known_nodes] = 1
    D = np.diag(dD)

    return D, known_nodes


def D_nodes_constrained_real(model, x, geo_size):
    """
    Returns diagonal matrix with selected available nodes for Constrained Tikhonov

    Parameters:
        n_nodes: number of nodes to select (default: clinical-based basket)
    Returns:
        D (matrix): diagonal matrix with selected available nodes.
        known_nodes (matrix): selected nodes.
    """

    dD = np.zeros(geo_size)
    known_nodes = dl.load_constrained_nodes_real(model) - 1
    dD[known_nodes] = 1
    D = np.diag(dD)

    x_ref_cons = np.zeros((geo_size, x.shape[1]))
    counter = 0
    for i in known_nodes:
        x_ref_cons[i, :] = x[counter, :]
        counter += 1

    # Sort nodes
    known_nodes = np.sort(known_nodes)
    return D, known_nodes, x_ref_cons


def D_nodes_constrained_real_dropout(model, x, geo_size, prob=0.3):
    """
    Returns diagonal matrix with selected available nodes for Constrained Tikhonov

    Parameters:
        n_nodes: number of nodes to select (default: clinical-based basket)
    Returns:
        D (matrix): diagonal matrix with selected available nodes.
        known_nodes (matrix): selected nodes.
    """

    dD = np.zeros(geo_size)
    known_nodes_init = dl.load_constrained_nodes_real(model) - 1

    # Modify number of nodes depending on the probability
    nodes_to_drop = int(known_nodes_init.size * prob)
    index_to_drop = np.random.randint(0, known_nodes_init.size, nodes_to_drop)
    known_nodes = np.delete(known_nodes_init, index_to_drop)

    # Array with dropped nodes
    dropped_nodes = np.setdiff1d(known_nodes_init, known_nodes)

    dD[known_nodes] = 1
    D = np.diag(dD)

    x_ref_cons = np.zeros((geo_size, x.shape[1]))
    counter = 0
    for i in known_nodes:
        x_ref_cons[i, :] = x[counter, :]
        counter += 1

    # Sort nodes
    known_nodes = np.sort(known_nodes)
    dropped_nodes = np.sort(dropped_nodes)

    return D, known_nodes, x_ref_cons, dropped_nodes
