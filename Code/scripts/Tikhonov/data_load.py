# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io
import h5py

data_path = "/home/mgutierrez/Desktop/Autoencoders/Data4/"
data_path_real = "/home/mgutierrez/Desktop/Autoencoders/Data4/"


def load_egms(model="SR"):
    """
    Load electrograms and select 2500 time instants

    Parameters:
        model (str): Model to load

    Returns:
        x: Electrograms for the selected model
    """

    try:
        EG = np.transpose(
            np.array((h5py.File(data_path + model + "/EGMs.mat", "r")).get("EG"))
        )
    except:
        EG = scipy.io.loadmat(data_path + model + "/EGMs.mat").get("EG")

    Nsamples = 2499
    if model == "SR":
        samples = np.arange(1, Nsamples)
    else:
        samples = np.arange((EG.shape)[1] - Nsamples - 1502 + 2, (EG.shape)[1] - 1500)

    x = EG[:, samples]

    return x


def load_egms_real(model):
    """
    Load electrograms and select 2500 time instants

    Parameters:
        model (str): Model to load

    Returns:
        x: Electrograms for the selected model
    """

    try:
        EGM = np.transpose(
            np.array(
                (h5py.File(data_path_real + model + "/signals.mat", "r")).get("EGM")
            )
        )
        ECG = np.transpose(
            np.array(
                (h5py.File(data_path_real + model + "/signals.mat", "r")).get("ECG")
            )
        )
    except:
        EGM = scipy.io.loadmat(data_path_real + model + "/signals.mat").get("EGM")
        ECG = scipy.io.loadmat(data_path_real + model + "/signals.mat").get("ECG")

    x = EGM
    y = ECG

    return x, y


def load_transfer():
    """
    Load the transfer matrix for atria and torso models

    Returns:
        MTransfer: Transfer matrix for atria and torso models
    """

    try:
        MTransfer = np.transpose(
            np.array((h5py.File(data_path + "transfer.mat", "r")).get("MTransfer"))
        )
    except:
        MTransfer = scipy.io.loadmat(data_path + "transfer.mat").get("MTransfer")

    # Transform transfer matrix to account for WCT correction. A matrix is the result of
    # referencing MTransfer to a promediated MTransfer. THe objective is to obtain an ECG
    # referenced to the WCT, following the next expression:
    # ECG_CTW = MTransfer * EGM - M_ctw * MTransfer * EGM =
    # = (MTransfer - M_ctw * MTransfer) * EGM = MTransfer_ctw * EGM
    M_wct = (1 / (MTransfer.shape)[0]) * np.ones(
        ((MTransfer.shape)[0], (MTransfer.shape)[0])
    )
    A = MTransfer - np.matmul(M_wct, MTransfer)

    return A


def load_transfer_real(model):
    """
    Load the transfer matrix for atria and torso models

    Returns:
        MTransfer: Transfer matrix for atria and torso models
    """

    try:
        A = np.transpose(
            np.array(
                (h5py.File(data_path_real + model + "/transfer.mat", "r")).get("A")
            )
        )
    except:
        A = scipy.io.loadmat(data_path_real + model + "/transfer.mat").get("A")

    return A


def load_geometry_real(model):
    """
    Load geometry data of the atria and the torso

    Returns:
        atrial_model: Dictionary of atrial model geometry data
        torso_model: Dictionary of torso model geometry data
    """

    try:
        atrial_model_temp = np.array(
            (h5py.File(data_path_real + model + "/geometry.mat", "r")).get(
                "atrial_model"
            )
        )
        torso_model_temp = np.array(
            (h5py.File(data_path_real + model + "/geometry.mat", "r")).get(
                "torso_model"
            )
        )
    except:
        atrial_model_temp = scipy.io.loadmat(
            data_path_real + model + "/geometry.mat"
        ).get("atrial_model")
        torso_model_temp = scipy.io.loadmat(
            data_path_real + model + "/geometry.mat"
        ).get("torso_model")

    atrial_model = dict()
    atrial_model["vertices"] = ((atrial_model_temp["vertices"])[0, :])[0]
    atrial_model["faces"] = ((atrial_model_temp["faces"])[0, :])[0]
    atrial_model["field"] = ((atrial_model_temp["field"])[0, :])[0]

    torso_model = dict()
    torso_model["vertices"] = ((torso_model_temp["vertices"])[0, :])[0]
    torso_model["faces"] = ((torso_model_temp["faces"])[0, :])[0]
    torso_model["field"] = ((torso_model_temp["field"])[0, :])[0]

    return atrial_model, torso_model


def load_geometry():
    """
    Load geometry data of the atria and the torso

    Returns:
        atrial_model: Dictionary of atrial model geometry data
        torso_model: Dictionary of torso model geometry data
    """

    try:
        atrial_model_temp = np.array(
            (h5py.File(data_path + "geometry.mat", "r")).get("atrial_model")
        )
        torso_model_temp = np.array(
            (h5py.File(data_path + "geometry.mat", "r")).get("torso_model")
        )
    except:
        atrial_model_temp = scipy.io.loadmat(data_path + "geometry.mat").get(
            "atrial_model"
        )
        torso_model_temp = scipy.io.loadmat(data_path + "geometry.mat").get(
            "torso_model"
        )

    atrial_model = dict()
    atrial_model["vertices"] = ((atrial_model_temp["vertices"])[0, :])[0]
    atrial_model["faces"] = ((atrial_model_temp["faces"])[0, :])[0]
    atrial_model["normales"] = ((atrial_model_temp["normales"])[0, :])[0]
    atrial_model["areas"] = ((atrial_model_temp["areas"])[0, :])[0]
    atrial_model["distances"] = ((atrial_model_temp["distances"])[0, :])[0]

    torso_model = dict()
    torso_model["vertices"] = ((torso_model_temp["vertices"])[0, :])[0]
    torso_model["faces"] = ((torso_model_temp["faces"])[0, :])[0]

    return atrial_model, torso_model


def load_constrained_nodes():
    """
    Load clinical-based known nodes distribution for Constrained Tikhonov approach

    Returns:
        Cons_nodes (array): clinical-based known nodes distribution
    """

    try:
        Cons_nodes = np.transpose(
            np.array(
                (h5py.File(data_path + "Constellation_nodes.mat", "r")).get(
                    "constellation_nodes"
                )
            )
        )
    except:
        Cons_nodes = scipy.io.loadmat(data_path + "Constellation_nodes.mat").get(
            "constellation_nodes"
        )

    return Cons_nodes


def load_constrained_nodes_real(model):
    """
    Load clinical-based known nodes distribution for Constrained Tikhonov approach

    Returns:
        Cons_nodes (array): clinical-based known nodes distribution
    """

    try:
        Cons_nodes = np.transpose(
            np.array(
                (h5py.File(data_path_real + model + "/EGM_mesh_index.mat", "r")).get(
                    "EGM_mesh_index"
                )
            )
        )
    except:
        Cons_nodes = scipy.io.loadmat(
            data_path_real + model + "/EGM_mesh_index.mat"
        ).get("EGM_mesh_index")

    Cons_nodes = Cons_nodes[:, 0]

    return Cons_nodes
