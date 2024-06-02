# -*- coding: utf-8 -*-
# """
# Created on Thu Jan 28 11:26:06 2021

# @author: Miguel Ángel
# """


import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scipy.io import loadmat
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from itertools import product
from numpy.random import default_rng
import scipy.io
import h5py
from scipy import signal as sigproc
from scipy.interpolate import interp1d
from math import floor
from scipy import signal
from add_white_noise import *
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import tensorflow as tf
import random
import math
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
import time
import keras
from keras import models, layers
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras import datasets, layers, models, losses, Model
from generators import *
from numpy import reshape
import matplotlib.image
from scipy.io import savemat
from plots import *
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from datetime import time


# %% Path Models
# %% Path Models
current = os.path.dirname(os.path.realpath(__file__))
torsos_dir = "../../../Labeled_torsos/"
directory = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"

fs = 500

# %%


# %% Add noise to signals
def add_noise(X, SNR=20, fs=50):

    X_noisy, _ = addwhitenoise(X, SNR=SNR, fs=fs)

    # Normalizar
    # mm = np.mean(X_noisy, axis=0)
    # ss = np.std(X_noisy, axis=0)
    # X_noisy_normalized = (X_noisy - mm[np.newaxis,:]) / ss[np.newaxis,:]
    # X_noisy_normalized=  X_noisy
    return X_noisy


def ECG_filtering(signal, fs, f_low=3, f_high=30):
    """
    Frequency filtering of ECG-EGM.
    SR model: low-pass filtering, 4th-order Butterworth filter.
    FA models: bandpass filtering, 4th-order Butterworth filter.

    Parameters:
        signal (array): signal to process
        fs (int): sampling rate
        f_low (int-float): low cut-off frecuency (default=3Hz)
        f_high (int-float): high cut-off frecuency (default=30Hz)
        model (string): FA model to assess (default: SR)
    Returns:
        proc_ECG_EGM (array): filtered ECG-EGM
    """

    # Remove DC component
    sig_temp = remove_mean(signal)

    # Bandpass filtering
    b, a = sigproc.butter(
        4, [f_low / round((fs / 2)), f_high / round((fs / 2))], btype="bandpass"
    )
    proc_ECG_EGM = np.zeros(sig_temp.shape)

    for index in range(0, sig_temp.shape[0]):
        proc_ECG_EGM[index, :] = sigproc.filtfilt(b, a, sig_temp[index, :])

    return proc_ECG_EGM


def load_egms_df(data_dir):
    # % Check models in directory
    all_model_names = []

    for subdir, dirs, files in os.walk(data_dir):
        if subdir != data_dir:
            model_name = subdir.split("/")[-1]

            all_model_names.append(model_name)
    print("data", all_model_names)

    df = pd.DataFrame(columns=["id", "AF_signal"])

    for model_name in all_model_names:

        # %% 1)  Compute EGM of the model
        egms = load_egms(model_name)
        df.loc[len(df)] = [model_name, egms]

    return df


def load_data(
    data_type,
    n_classes=2,
    SR=True,
    subsampling=True,
    fs_sub=50,
    SNR=20,
    norm=False,
    classification=False,
    sinusoid=False,
    n_batch=50,
):
    """
    Returns y values depends of the classification model

    Parameters:
        data_type: '3channelTensor' -> tx6x4x3 tensor (used for CNN-based classification)
                   '1channelTensor' -> tx6x16 tensor (used for CNN-based classification)

                   'Flat' ->  tx64 matrix (used for MLP-based classification) or tx10
        n_classes: for classification task 1-->
                    '1' -> Rotor/no rotor
                    '2' ->  RA/LA/No rotor (2 classes)
                    '3' ->  7 regions (3 classes) + no rotor (8 classes)
        SR: If Sinus Ryhthm Model is included (True/False)
        subsampling: if downsampling from fs= 500Hz to a lower fs_sub is done (True/False)
        fs_sub: Freq sampl to downsample the signal from its original fs= 500Hz
        norm: if normalizing is performed to models when loading them (True/False)
        classification: if classification task is performed
        sinusoid: if a database of sinusoids is generated (True/False)
        n_batch: batch size (number of samples/batch)


    Returns:
        X -> input data.
        Y -> labels.
        Y_model -> model associated for each time instant.
        egm_tensor -> original EGM values associated to each BSP
        length_list -> length (samples) of each EGM model
        all_model_names -> Name of all AF models loaded
        transfer_matrices -> All transfer matrices used for Forward Problem (*for using them for Tikhonov later)
    """
    fs = 500

    # % Check models in directory
    all_model_names = []

    for subdir, dirs, files in os.walk(directory):
        if subdir != directory:
            model_name = subdir.split("/")[-1]
            if "Sinusal" in model_name and SR == False:
                continue
            else:
                all_model_names.append(model_name)

    if sinusoid:
        n = 80  # NUmber of sinusoid models generated
        all_model_names = ["Model {}".format(m) for m in range(n + 1)]
    print(len(all_model_names), "Models")

    # % Load models
    X = []
    Y = []
    Y_model = []
    n_model = 1
    egm_tensor = []
    length_list = []
    AF_models = []

    # Load corrected transfer matrices
    transfer_matrices = load_transfer(ten_leads=False, bsps_set=False)

    AF_model_i = 0

    for model_name in all_model_names:

        # %% 1)  Compute EGM of the model
        egms = load_egms(model_name, sinusoid)

        # 1.1) Discard models <1500
        # if len(egms[1])<1500:
        # continue

        # 1.2)  EGMs filtering.
        x = ECG_filtering(egms, 500)

        # 1.3 Normalize models
        if norm == True:

            high = 1
            low = -1

            mins = np.min(x, axis=0)
            maxs = np.max(x, axis=0)
            rng = maxs - mins

            bsps_64_n = high - (((high - low) * (maxs - x)) / rng)

        # 2) Compute the Forward problem with each of the transfer matrices
        for matrix in transfer_matrices:

            # Forward problem
            y = forward_problem(x, matrix[0])
            bsps_64 = y[matrix[1].ravel(), :]
            bsps_64_or = bsps_64
            # 3) Add NOISE and Filter
            if SNR != None:
                bsps_64_noise = add_noise(np.array(bsps_64), SNR=SNR, fs=500)

            # 5) Filter AFTER adding noise
            bsps_64_filt = ECG_filtering(bsps_64_noise, 500)

            # RESAMPLING signal to fs= fs_sub
            if subsampling:
                bsps_64 = signal.resample_poly(bsps_64_filt, fs_sub, 500, axis=1)
                x_sub = signal.resample_poly(x, fs_sub, 500, axis=1)

            else:

                bsps_64 = bsps_64_filt
                x_sub = x

            if classification:

                y_labels = get_labels(n_classes, model_name)
                y_labels_list = y_labels.tolist()

                # RESAMPLING labels to fs= fs_sub
                if subsampling:
                    y_labels = sample(y_labels, len(bsps_64[1]))

                y_model = np.full(len(y_labels), n_model)
                Y_model.extend(y_model)
                Y.extend(y_labels)
                Y_model.extend(y_model)

            else:

                Y.extend(np.full(len(x_sub), 0))

            if data_type == "3channelTensor":

                tensors_model = get_tensor_model(bsps_64, tensor_type="3channel")
                X.extend(tensors_model)

            elif data_type == "1channelTensor":

                tensors_model = get_tensor_model(bsps_64, tensor_type="1channel")
                # Interpolate
                reshape_tensor = np.reshape(
                    tensors_model,
                    (
                        len(tensors_model),
                        tensors_model.shape[1],
                        tensors_model.shape[2],
                        1,
                    ),
                )
                tensors_model = tf.keras.layers.UpSampling2D(
                    size=(2, 2), interpolation="bilinear"
                )(reshape_tensor)
                tensors_model = np.reshape(
                    tensors_model,
                    (
                        len(tensors_model),
                        tensors_model.shape[1],
                        tensors_model.shape[2],
                    ),
                )

                # Truncate for being divisible by the batch size
                batch_size = n_batch
                if tensors_model.shape[0] % batch_size != 0:
                    trunc_val = np.floor_divide(tensors_model.shape[0], batch_size)
                    tensors_model = tensors_model[0 : batch_size * trunc_val, :, :]
                    x_sub = x_sub[:, 0 : batch_size * trunc_val]
                length_list.append(tensors_model.shape[0])

                if x_sub.shape[1] != tensors_model.shape[0]:
                    print(x_sub.shape, tensors_model.shape, model_name)

                X.extend(tensors_model)
                egm_tensor.extend(x_sub.T)

            else:
                X.extend(bsps_64.T)
                egm_tensor.extend(x_sub.T)

            if not classification:
                y_model = np.full(len(tensors_model), n_model)
                Y_model.extend(y_model)

                # Count AF Model
                AF_model_i_array = np.full(len(tensors_model), AF_model_i)
                AF_models.extend(AF_model_i_array)

            n_model += 1

        AF_model_i += 1

    return (
        np.array(X),
        np.array(Y),
        np.array(Y_model),
        np.array(egm_tensor),
        length_list,
        AF_models,
        all_model_names,
        transfer_matrices,
    )


def sample(input, count):
    ss = float(len(input)) / count
    return [input[int(floor(i * ss))] for i in range(count)]


def plot_load_data(egms, x, bsps_64_noise, bsps_64_filt, bsps_64_or, n_model, bsps_64):

    # PLOT
    if n_model == 160:
        print("Ploting...")
        y_50 = np.linspace(0, 4, 200, endpoint=False)
        y_500 = np.linspace(0, 4, 2000, endpoint=False)
        plt.figure(layout="tight")
        plt.subplot(4, 1, 1)
        plt.plot(y_500, egms[0, 0:2000], label="original")
        plt.plot(y_500, x[0, 0:2000], label="filtered")
        plt.xlabel("Seconds")
        plt.legend()
        plt.title("Original vs filtered EGM")
        plt.subplot(4, 1, 2)
        plt.plot(
            y_500,
            bsps_64_noise[0, 0:2000],
            alpha=0.75,
            label="2. BSPM noise added (20 db)",
        )
        plt.plot(y_500, bsps_64_filt[0, 0:2000], label="3. BSPM filtered")
        plt.plot(y_50, bsps_64[0, 0:200], label="4. BSPM subsampled to 50 Hz")
        plt.plot(y_500, bsps_64_or[0, 0:2000], alpha=0.75, label="1. BSPM original")

        plt.title("BSPM in node 0")
        plt.xlabel("Seconds")
        plt.legend()
        plt.subplot(4, 1, 3)
        plt.plot(
            y_500,
            bsps_64_noise[20, 0:2000],
            alpha=0.75,
            label="2. BSPM noise added (20 db)",
        )
        plt.plot(y_500, bsps_64_filt[20, 0:2000], label="3. BSPM filtered")
        plt.plot(y_50, bsps_64[20, 0:200], label="4. BSPM subsampled to 50 Hz")
        plt.plot(y_500, bsps_64_or[20, 0:2000], alpha=0.75, label="1. BSPM original")

        plt.title("BSPM in node 20")
        plt.xlabel("Seconds")
        plt.legend()
        plt.subplot(4, 1, 4)
        plt.plot(
            y_500,
            bsps_64_noise[50, 0:2000],
            alpha=0.75,
            label="2. BSPM noise added (20 db)",
        )
        plt.plot(y_500, bsps_64_filt[50, 0:2000], label="3. BSPM filtered")
        plt.plot(y_50, bsps_64[50, 0:200], label="4. BSPM subsampled to 50 Hz")
        plt.plot(y_500, bsps_64_or[50, 0:2000], alpha=0.75, label="1. BSPM original")

        plt.title("BSPM in node 50")
        plt.xlabel("Seconds")
        plt.legend()
        plt.show()
        print("close...")


def sinusoids_generator(n, m, fs=100):
    """Generates n sinusoids (n nodes)of length m samples with f0 between 5 Hz and 10 Hz and a freq sample of fs"""

    freq = np.random.uniform(5, 10, size=n)
    phases = np.random.uniform(0, 2 * np.pi, size=n)
    t = np.arange(m) / fs
    sinusoids = np.sin(
        2 * np.pi * np.outer(t, freq) + phases
    )  # y = A*sin (2pi * f0 *t + phase)
    return sinusoids


def load_egms(model_name, sinusoid=False):
    """
    Load electrograms and select 2500 time instants and 2048 nodes

    Parameters:
        model (str): Model to load

    Returns:
        x: Electrograms for the selected model
    """
    if sinusoid:

        EG = sinusoids_generator(2048, 2500, fs=500).T
    else:

        try:
            EG = np.transpose(
                np.array(
                    (h5py.File(directory + model_name + "/EGMs.mat", "r")).get("x")
                )
            )
        except:
            EG = scipy.io.loadmat(directory + model_name + "/EGMs.mat").get("x")

    return EG


def remove_mean(signal):
    """
    Remove mean from signal

    Parameters:
        signal (array): signal to process

    Returns:
        signotmean: signal with its mean removed
    """
    signotmean = np.zeros(signal.shape)
    for index in range(0, signal.shape[0]):
        signotmean[index, :] = sigproc.detrend(signal[index, :], type="constant")
    return signotmean


def forward_problem(EGMs, MTransfer):
    """
    Calculate ECGI forward problem from atrial EGMs

    Parameters:
        EGMs (array): atrial electrograms.
        MTransfer (matrix): transfer matrix.
    Returns:
        ECG (array): ECG reconstruction (torso).
    """
    ECG = np.matmul(MTransfer, EGMs)
    return ECG


def load_transfer(ten_leads=False, bsps_set=False):
    """
    Load the transfer matrix for atria and torso models

    Returns:
        MTransfer: Transfer matrix for atria and torso models
    """

    all_torsos_names = []
    for subdir, dirs, files in os.walk(torsos_dir):
        for file in files:
            if file.endswith(".mat"):
                all_torsos_names.append(file)

    transfer_matrices = []
    for torso in all_torsos_names:
        MTransfer = scipy.io.loadmat(torsos_dir + torso).get("TransferMatrix")

        if ten_leads == True:
            BSP_pos = scipy.io.loadmat(torsos_dir + torso).get("torso")["leads"][0, 0]
        elif bsps_set == True:
            torso_electrodes = loadmat(directory + "torso_electrodes.mat")
            BSP_pos = torso_electrodes["torso_electrodes"][0]
            # BSP_pos = get_bsps_192 (torso, False)
        else:
            BSP_pos = scipy.io.loadmat(torsos_dir + torso).get("torso")["bspmcoord"][
                0, 0
            ]

        # Transform transfer matrix to account for WCT correction. A matrix is the result of
        # referencing MTransfer to a promediated MTransfer. THe objective is to obtain an ECG
        # referenced to the WCT, following the next expression:
        # ECG_CTW = MTransfer * EGM - M_ctw * MTransfer * EGM =
        # = (MTransfer - M_ctw * MTransfer) * EGM = MTransfer_ctw * EGM

        M_wct = (1 / (MTransfer.shape)[0]) * np.ones(
            ((MTransfer.shape)[0], (MTransfer.shape)[0])
        )
        A = MTransfer - np.matmul(M_wct, MTransfer)

        transfer_matrices.append((A, BSP_pos - 1))

    return transfer_matrices


def get_bsps_192(model_name, ten_leads=False):
    """
    Reduce the 659 BSPs to 192, 3 BSPSs for each node.

    Parameters:
      model_name: name of the model

    Return:
      bsps_192: array with 192 BSPs

    """
    torso_electrodes = loadmat(directory + "/torso_electrodes.mat")
    torso_electrodes = torso_electrodes["torso_electrodes"][0]

    bsps = loadmat(directory + model_name + "/BSPs.mat")
    bsps = bsps["y"]

    bsps_192 = []
    for i in torso_electrodes:
        bsps_192.append(bsps[i, :])

    return np.array(bsps_192)


def get_tensor_model(bsps_64, tensor_type="3channel"):
    """
    Get X (tensor) from one model

    Parameters:
      bsps_64: 64 x n_time matrix bsps for 1 model
      y_model: array of y labels from 1 model

    Return:
      all_tensors: array of all tnesor from 1 model
    """
    all_tensors = np.array([])

    for t in range(0, bsps_64.shape[1]):
        if tensor_type == "3channel":
            tensor_model = get_subtensor_2(bsps_64[:, t], tensor_type)
        else:
            tensor_model = get_subtensor_2(bsps_64[:, t], tensor_type)
        if all_tensors.size == 0:
            all_tensors = tensor_model
        else:
            all_tensors = np.concatenate((all_tensors, tensor_model), axis=0)

    return all_tensors


def get_bsps_64(bsps_192, seed="Y"):
    """
    Reduce 192 BSPs to 64, selecting 1 random BSPs of the 3 posibilities for each node

    Parameters:
    bsps_192: 192 x n_times matrix for 1 model

      Returns:
        bsps_64: 64 x n_time matrix for 1 model
    """

    if seed == "Y":
        rng = default_rng(0)
    else:
        rng = default_rng()

    bsps_64 = []
    pos = 0
    num_row = int(bsps_192.shape[0] / 3)

    for i in range(0, num_row):
        rand = rng.integers(low=0, high=2, endpoint=True)
        bsps = bsps_192[pos : pos + 3][rand]
        bsps_64.append(bsps)
        pos += 3

    return np.array(bsps_64)


def replace_null_labels(labels):
    closest_i = labels[
        (labels != 0).argmax(axis=0)
    ]  # in position 0, as there is no left position it takes the first right label instead
    for index in range(0, len(labels)):
        if labels[index] > 0:
            closest_i = labels[index]
        else:
            labels[index] = closest_i
    return labels


def get_labels(class_type, model_name):
    """
    Get y labels for each classification type: 2, 3 or 7.

    Parameters:
      class_type:  2 -> binary classification, exist(1) or does not exist(0) driver.
                   3 -> multi-classification: does not exist(0), LA(1) and RA(2)
                   7 -> multi-classification: does not exist(0), 1-7 for each region
                   Any other value of class_type raise a ValueError.

      model_name: name of the model.

      Return:
        labels: y labels fro the model
    """

    regions = loadmat(directory + "/regions.mat")
    regions = regions["regions"][0]

    path_driver_position = directory + model_name + "/driver_position.mat"
    driver_position = loadmat(path_driver_position)
    driver_position = driver_position["driver_position"]

    nodes_aux = driver_position[:, 1]
    driver_position = driver_position[:, 0]

    # Obtener el nodo correspondiente a una etiqueta
    nodes = []
    for i in nodes_aux:
        if i.size == 0:
            nodes.append(9999)
        else:
            nodes.append(i)
    nodes = np.array(nodes, dtype=int)

    # Obtener solo las etiquetas.
    labels = np.array(driver_position, dtype=int)

    # Hay o no hay rotor
    if class_type == 2:
        return labels

    # No hay, Izq o dcha
    if class_type == 3:
        for subdir, dirs, files in os.walk(directory):
            for subdir2, dirs2, files2 in os.walk(subdir):
                if subdir != directory:
                    # Si es RA y hay driver, y=2
                    if model_name.startswith("RA"):
                        labels[labels == 1] = 2
                    if model_name == "TwoRotors_181219":
                        labels[labels == 1] = 2
        return labels

    # No hay o 7 regiones
    if class_type == 6:
        for index, item in enumerate(labels):
            if item != 0:

                labels[index] = regions[nodes[index] - 1]
        # replace 0 with previous label
        labels = replace_null_labels(labels)

        return labels


def get_subtensor_2(bsps_64_t, tensor_type="3_channel"):
    """
    Get (6 x 4 x 3) tensor for 1 instance of time.

    Parameters:
    bsps_64_t: 1 instance of time of bsps_64

    Return:
    subtensor: 6 x 4 x 3 matrix
    """

    patches = get_patches_name(bsps_64_t)

    if tensor_type == "3channel":
        torso = np.array(
            [
                [patches["A6"], patches["A12"], patches["B12"], patches["B6"]],
                [patches["A5"], patches["A11"], patches["B11"], patches["B5"]],
                [patches["A4"], patches["A10"], patches["B10"], patches["B4"]],
                [patches["A3"], patches["A9"], patches["B9"], patches["B3"]],
                [patches["A2"], patches["A8"], patches["B8"], patches["B2"]],
                [patches["A1"], patches["A7"], patches["B7"], patches["B1"]],
            ]
        )

        back = np.array(
            [
                [patches["D12"], patches["D6"], patches["C12"], patches["C6"]],
                [patches["D11"], patches["D5"], patches["C11"], patches["C5"]],
                [patches["D10"], patches["D4"], patches["C10"], patches["C4"]],
                [patches["D9"], patches["D3"], patches["C9"], patches["C3"]],
                [patches["D8"], patches["D2"], patches["C8"], patches["C2"]],
                [patches["D7"], patches["D1"], patches["C7"], patches["C1"]],
            ]
        )

        side = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [patches["R8"], patches["R4"], patches["L4"], patches["L8"]],
                [patches["R7"], patches["R3"], patches["L3"], patches["L7"]],
                [patches["R6"], patches["R2"], patches["L2"], patches["L6"]],
                [patches["R5"], patches["R1"], patches["L1"], patches["L5"]],
            ]
        )

        subtensor = np.stack((torso, side, back), axis=-1)
        subtensor = subtensor.reshape(1, 6, 4, 3)

    else:
        interp_lat_R4 = np.mean((patches["A6"], patches["A5"], patches["R4"]))
        interp_lat_R8 = np.mean((patches["D12"], patches["D11"], patches["R8"]))
        interp_lat_L8 = np.mean((patches["C6"], patches["C5"], patches["L8"]))
        interp_lat_L4 = np.mean((patches["B6"], patches["B5"], patches["L4"]))

        interp_lat_R1 = np.mean((patches["A1"], patches["A2"], patches["R1"]))
        interp_lat_R5 = np.mean((patches["D7"], patches["D8"], patches["R5"]))
        interp_lat_L5 = np.mean((patches["C1"], patches["C2"], patches["L5"]))
        interp_lat_L1 = np.mean((patches["B1"], patches["B2"], patches["L1"]))

        subtensor = np.array(
            [
                [
                    [
                        patches["B6"],
                        patches["B12"],
                        patches["A12"],
                        patches["A6"],
                        interp_lat_R4,
                        interp_lat_R8,
                        patches["D12"],
                        patches["D6"],
                        patches["C12"],
                        patches["C6"],
                        interp_lat_L8,
                        interp_lat_L4,
                        patches["B6"],
                        patches["B12"],
                        patches["A12"],
                        patches["A6"],
                    ],
                    [
                        patches["B5"],
                        patches["B11"],
                        patches["A11"],
                        patches["A5"],
                        patches["R4"],
                        patches["R8"],
                        patches["D11"],
                        patches["D5"],
                        patches["C11"],
                        patches["C5"],
                        patches["L8"],
                        patches["L4"],
                        patches["B5"],
                        patches["B11"],
                        patches["A11"],
                        patches["A5"],
                    ],
                    [
                        patches["B4"],
                        patches["B10"],
                        patches["A10"],
                        patches["A4"],
                        patches["R3"],
                        patches["R7"],
                        patches["D10"],
                        patches["D4"],
                        patches["C10"],
                        patches["C4"],
                        patches["L7"],
                        patches["L3"],
                        patches["B4"],
                        patches["B10"],
                        patches["A10"],
                        patches["A4"],
                    ],
                    [
                        patches["B3"],
                        patches["B9"],
                        patches["A9"],
                        patches["A3"],
                        patches["R2"],
                        patches["R6"],
                        patches["D9"],
                        patches["D3"],
                        patches["C9"],
                        patches["C3"],
                        patches["L6"],
                        patches["L2"],
                        patches["B3"],
                        patches["B9"],
                        patches["A9"],
                        patches["A3"],
                    ],
                    [
                        patches["B2"],
                        patches["B8"],
                        patches["A8"],
                        patches["A2"],
                        patches["R1"],
                        patches["R5"],
                        patches["D8"],
                        patches["D2"],
                        patches["C8"],
                        patches["C2"],
                        patches["L5"],
                        patches["L1"],
                        patches["B2"],
                        patches["B8"],
                        patches["A8"],
                        patches["A2"],
                    ],
                    [
                        patches["B1"],
                        patches["B7"],
                        patches["A7"],
                        patches["A1"],
                        interp_lat_R1,
                        interp_lat_R5,
                        patches["D7"],
                        patches["D1"],
                        patches["C7"],
                        patches["C1"],
                        interp_lat_L5,
                        interp_lat_L1,
                        patches["B1"],
                        patches["B7"],
                        patches["A7"],
                        patches["A1"],
                    ],
                ]
            ]
        )

        subtensor = subtensor.reshape(1, 6, 16)

    return subtensor


def get_patches_name(bsps_64):
    """
    Get names of patches in bsps_64

    Parameters:
        bsps_64:
    Return:
        patches: dictionary whit patche name as key and bsps as value.
    """
    patches = {}

    index = 1
    for i in range(0, 12):
        patches["A{0}".format(index)] = bsps_64[i]
        index += 1

    index = 1
    for i in range(12, 24):
        patches["B{0}".format(index)] = bsps_64[i]
        index += 1

    index = 1
    for i in range(24, 36):
        patches["C{0}".format(index)] = bsps_64[i]
        index += 1

    index = 1
    for i in range(36, 48):
        patches["D{0}".format(index)] = bsps_64[i]
        index += 1

    index = 1
    for i in range(48, 56):
        patches["L{0}".format(index)] = bsps_64[i]
        index += 1

    index = 1
    for i in range(56, 64):
        patches["R{0}".format(index)] = bsps_64[i]
        index += 1

    return patches


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.0
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def normalize_array(array, high, low, axis_n=0):
    """
    This functions normalized a 2D 'array' along axis 'axis_n' and between the values 'high' and 'low'

    To normalize a full signal, indicate the index dimension

    """
    mins = np.min(array, axis=axis_n)
    maxs = np.max(array, axis=axis_n)
    rng = maxs - mins
    # if axis_n==1:
    # array=array.T
    norm_array = high - (((high - low) * (maxs - array)) / rng)
    # if axis_n==1:
    # norm_array=norm_array.T
    return norm_array


def train_test_val_split_Autoencoder(
    X_1channel,
    AF_models,
    BSPM_Models,
    all_model_names,
    random_split,
    train_percentage,
    test_percentage,
):
    """
    This function splits the input tensor into train, tets and validation
    Parameters:
        X_1channel-> input BSPs tensor
        AF_models -> list corresponding to the original AF model that corresponds to each BSP
        BSPM_Models -> list of BSP model values for each sample
        all_model_names -> Name of all AF models loaded
        random_split -> AF models are randomly shaffled and assigned to each subset (train, test, val)
        train_percentage -> train percentage of the input dataset dedicated to training the models
        test_percentage -> test percentage of the input dataset dedicated to testing the models.
        *Validation is computed as 100-train_percentage-test_percentage


    Return:
        x_train -> training tensor
        x_test -> testing tensor
        x_val -> validation tensor
        train_models, test_models, val_models  -> BSP Models in train, test and val (Only id of AF Model)
        AF_models_train, AF_models_test, AF_models_val  -> Tensor AF Models in train, test and val
        BSPM_train, BSPM_test, BSPM_val  -> Tensor BSP Models in train, test and val

    """
    caution_split = True  # Split in train and test taking into account the high corr between selected models in 'set_models'

    if caution_split:
        # Select indices of highly correlated signals

        set_models = {
            "190619",
            "190717",
            "191001",
            "200316_001",
            "200428",
            "200316",
            "200212",
        }  #
        indx = []
        for i in range(0, len(all_model_names)):
            for s in set_models:
                if s in all_model_names[i]:
                    indx.append(i)

    AF_models_unique = np.unique(AF_models)

    # Random
    if random_split:

        if caution_split:
            train_models = random.sample(
                list(AF_models_unique),
                int(np.floor(AF_models[-1] * train_percentage - len(indx))),
            )
            train_models = train_models + indx
        else:
            train_models = random.sample(
                list(AF_models_unique), int(np.floor(AF_models[-1] * train_percentage))
            )

        aux_models = [x for x in AF_models_unique if x not in train_models]
        test_models = random.sample(
            list(aux_models), int(np.floor(AF_models[-1] * test_percentage))
        )
        val_models = [x for x in aux_models if x not in test_models]
        x_train = X_1channel[np.in1d(AF_models, train_models)]
        x_test = X_1channel[np.in1d(AF_models, test_models)]
        x_val = X_1channel[np.in1d(AF_models, val_models)]

        print("TRAIN MODELS:", train_models)
        print("TEST MODELS:", test_models)
        print("VAL MODELS:", val_models)

        BSPM_train = BSPM_Models[np.in1d(AF_models, train_models)]
        BSPM_test = BSPM_Models[np.in1d(AF_models, test_models)]
        BSPM_val = BSPM_Models[np.in1d(AF_models, val_models)]

        AF_models_arr = np.array(AF_models)
        AF_models_train = AF_models_arr[np.in1d(AF_models, train_models)]
        AF_models_test = AF_models_arr[np.in1d(AF_models, test_models)]
        AF_models_val = AF_models_arr[np.in1d(AF_models, val_models)]

    else:

        x_train = X_1channel[np.where((Y_model >= 1) & (Y_model <= 200))]
        x_test = X_1channel[np.where((Y_model > 180) & (Y_model <= 244))]
        x_val = X_1channel[np.where((Y_model > 244) & (Y_model <= 286))]

    # Save the model names in train, test and val
    test_model_name = [all_model_names[index] for index in AF_models_test]
    val_model_name = [all_model_names[index] for index in AF_models_val]
    train_model_name = [all_model_names[index] for index in AF_models_train]

    return (
        x_train,
        x_test,
        x_val,
        train_models,
        test_models,
        val_models,
        AF_models_train,
        AF_models_test,
        AF_models_val,
        BSPM_train,
        BSPM_test,
        BSPM_val,
    )


def video_generate(x_test, latent_vector, video=False):
    samples, height, width = x_test.shape
    size = (width, height)
    print(size)
    FPS = 5
    if video:

        # Save frames as pnf into folder
        for i in range(samples):
            im = Image.fromarray(x_test[i, :, :])

            matplotlib.image.imsave("./Frames_test/frame{}.png".format(i), im)

        # Save frames as pnf into folder
        for i in range(len(decoded_imgs)):
            im = Image.fromarray(decoded_imgs[i, :, :])

            matplotlib.image.imsave("./Frames_reconstruction/frame{}.png".format(i), im)

        # Save frames as pnf into folder
        for i in range(len(latent_vector)):
            im = Image.fromarray(latent_vector[i, :, :, 1])

            matplotlib.image.imsave("./Frames_LS/frame{}.png".format(i), im)

        # Load png and convert to video
        frameSize = (width, height)
        out = cv2.VideoWriter(
            "./Videos/output_video_test_{}_fps.avi".format(FPS),
            cv2.VideoWriter_fourcc(*"DIVX"),
            FPS,
            frameSize,
        )

        for filename in glob.glob("./Frames_test/*.png"):
            img = cv2.imread(filename)
            out.write(img)

        out.release()

        samples, height, width, filters = latent_vector.shape
        frameSize = (width, height)
        out = cv2.VideoWriter(
            "./Videos/output_LS_{}_fps.avi".format(FPS),
            cv2.VideoWriter_fourcc(*"DIVX"),
            FPS,
            frameSize,
        )

        for filename in glob.glob("./Frames_LS/*.png"):
            img = cv2.imread(filename)
            out.write(img)

        out.release()


def corr_pearson_cols(array1, array2):
    """
    Calcula la correlación de Pearson entre las columnas de dos arrays.

    Args:
        array1: un array de numpy de dimensión (n,m)
        array2: otro array de numpy de dimensión (n,m)

    Returns:
        Un array de numpy de dimensión (m,) que contiene la correlación de Pearson
        de las columnas de array1 y array2.
    """

    # Verificar si ambos arrays tienen las mismas dimensiones
    assert (
        array1.shape == array2.shape
    ), "Los arrays deben tener las mismas dimensiones."

    # Calcular la correlación de Pearson de las columnas de ambos arrays
    n_cols = array1.shape[1]
    corr = np.zeros(n_cols)
    for i in range(n_cols):
        corr[i], _ = pearsonr(array1[:, i], array2[:, i])

    return corr


def corr_spearman_cols(array1, array2):
    """
    Calcula la correlación de Spearman entre las columnas de dos arrays.

    Args:
        array1: un array de numpy de dimensión (n,m)
        array2: otro array de numpy de dimensión (n,m)

    Returns:
        Un array de numpy de dimensión (m,) que contiene la correlación de Spearman
        de las columnas de array1 y array2.
    """

    # Verificar si ambos arrays tienen las mismas dimensiones
    assert (
        array1.shape == array2.shape
    ), "Los arrays deben tener las mismas dimensiones."

    # Calcular la correlación de Spearman de las columnas de ambos arrays
    n_cols = array1.shape[1]
    corr = np.zeros(n_cols)
    for i in range(n_cols):
        corr[i], _ = spearmanr(array1[:, i], array2[:, i])

    return corr


def correlation_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample):
    """This function computes the Spearman correlation between the reconstruction and the
    real signal, but separately for each AF Model"""

    correlation_list = []
    test_models_corr = []

    for model in np.unique(AF_models_test):
        # 1. Normalize Reconstruction
        estimation_array = estimate_egms_n[
            np.where((AF_models_test == model))
        ]  # select window of signal belonging to model i
        y_array = y_test_subsample[
            np.where((AF_models_test == model))
        ]  # select window of signal belonging to model i
        correlation_pearson_nodes = corr_spearman_cols(estimation_array, y_array)
        correlation_list.extend([correlation_pearson_nodes])
        test_models_corr.extend([AF_models_test[model]])

    correlation_array = np.array(correlation_list)

    return correlation_array, test_models_corr


def DTW_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample):
    """This function computes DTW between the reconstruction and the real signal, but separately for each AF Model.
    It computes Fast DTW for each node in each AF Model
    """
    dtw_list = []
    dtw_list_random = []

    # Create random signals

    random_signal1 = sampl = np.random.uniform(low=0.5, high=13.3, size=(2000, 512))

    for model in np.unique(AF_models_test):

        estimation_array = estimate_egms_n[
            np.where((AF_models_test == model))
        ]  # select window of signal belonging to model i
        y_array = y_test_subsample[
            np.where((AF_models_test == model))
        ]  # select window of signal belonging to model i
        random_signal = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(estimation_array.shape[0], estimation_array.shape[1]),
        )

        dtw_list_node = []
        dtw_list_node_random = []

        for node in range(0, 512):  # Compute DTW in each node
            # DTW
            dtw_test, path = fastdtw(estimation_array[:, node], y_array[:, node])
            dtw_random, path = fastdtw(random_signal[:, node], y_array[:, node])

            dtw_list_node.append(dtw_test)
            dtw_list_node_random.append(dtw_random)

        dtw_list.extend([dtw_list_node])
        dtw_list_random.extend([dtw_list_node_random])

    dtw_array = np.array(dtw_list)
    dtw_array_random = np.array(dtw_list_random)

    return dtw_array, dtw_array_random


def RMSE_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample):
    """This function computes RMSE between the reconstruction and the real signal, but separately for each AF Model.
    It computes RMSE for each node in each AF Model
    """
    # DTW by nodes and models
    rmse_list = []

    for model in np.unique(AF_models_test):

        estimation_array = estimate_egms_n[
            np.where((AF_models_test == model))
        ]  # select window of signal belonging to model i
        y_array = y_test_subsample[
            np.where((AF_models_test == model))
        ]  # select window of signal belonging to model i
        rmse_list_node = []

        for node in range(0, 512):

            # RMSE
            MSE = mean_squared_error(estimation_array[:, node], y_array[:, node])
            RMSE = math.sqrt(MSE)
            rmse_list_node.append(RMSE)

        rmse_list.extend([rmse_list_node])

    rmse_array = np.array(rmse_list)

    return rmse_array


def normalize_by_models(data, Y_model):
    """
    This function normalizes the input tensor between -1 and 1 in each model separately

    """
    # Normalize by models of BSPM

    data_orig = data
    if data.ndim == 3:
        data = reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))

    elif data.ndim == 4:
        data = reshape(
            data, (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3])
        )
    elif data.ndim != 2:
        raise (
            ValueError(
                "Input shape for norm not correct. Nnim should be 2, but is", data.ndim
            )
        )

    data_n = []

    for model in np.unique(Y_model):
        arr_to_norm = data[
            np.where((Y_model == model))
        ]  # select window of signal belonging to model i
        norm_model = normalize_array(arr_to_norm, 1, -1)
        data_n.extend(norm_model)  # Add to new norm array

    data_n = np.array(data_n)

    if data_orig.ndim == 3:
        data_n = data_n.reshape(
            data_orig.shape[0], data_orig.shape[1], data_orig.shape[2]
        )

    elif data_orig.ndim == 4:
        data_n = data_n.reshape(
            data_orig.shape[0],
            data_orig.shape[1],
            data_orig.shape[2],
            data_orig.shape[3],
        )

    return data_n


def interpolate_fun(array, n_models, final_nodes, sig=False):
    """
    This function applies interpolation to a final number of 'final_nodes' points.
    It operates in each model indepently

    """

    if sig:
        if array.ndim > 2:
            array = reshape(array, (array.shape[0], array.shape[1], 1, 1))
        else:
            interpol = tf.keras.layers.UpSampling2D(
                size=(4, 1), interpolation="bilinear"
            )(array)
            array_interpol = reshape(interpol, (interpol.shape[0], interpol.shape[1]))
            print(array_interpol.shape)

    else:
        array_interpol = []
        for model in range(0, n_models):
            s = array[model, :]
            f = interp1d(np.linspace(0, 1, len(s)), s, kind="linear")
            # Creamos un nuevo array de tamaño (1, 2048)
            interp = f(np.linspace(0, 1, final_nodes))
            array_interpol.append(interp)

        array_interpol = np.array(array_interpol)

    return array_interpol


def preprocess_latent_space(
    latent_vector_train,
    latent_vector_test,
    latent_vector_val,
    train_models,
    test_models,
    val_models,
    Y_model,
    egm_tensor,
    dimension,
    norm=False,
):
    """
    This function preprocess the latent space, following the scheme:
    1) Center data at 0
    2) Reshape for normalization
    3) Normalization between -1 and 1
    """
    # Center latent space
    center_function = lambda x: x - x.mean(axis=0)

    latent_vector_train = center_function(latent_vector_train)
    latent_vector_test = center_function(latent_vector_test)
    latent_vector_val = center_function(latent_vector_val)

    if dimension == 5:

        # Reshape latent space --> Flatten 'nº batch' x 'batch size' to normalize
        latent_vector_train = reshape(
            latent_vector_train,
            (
                latent_vector_train.shape[0] * latent_vector_train.shape[1],
                latent_vector_train.shape[2],
                latent_vector_train.shape[3],
                latent_vector_train.shape[4],
            ),
        )
        latent_vector_test = reshape(
            latent_vector_test,
            (
                latent_vector_test.shape[0] * latent_vector_test.shape[1],
                latent_vector_test.shape[2],
                latent_vector_test.shape[3],
                latent_vector_test.shape[4],
            ),
        )
        latent_vector_val = reshape(
            latent_vector_val,
            (
                latent_vector_val.shape[0] * latent_vector_val.shape[1],
                latent_vector_val.shape[2],
                latent_vector_val.shape[3],
                latent_vector_val.shape[4],
            ),
        )

    # first we merge Latent Space train/test/val
    con = np.concatenate((latent_vector_train, latent_vector_test))
    latent_space = np.concatenate((con, latent_vector_val))

    # Normalize
    if norm:

        latent_space_n = normalize_by_models(latent_space, Y_model)
        egm_tensor_n = normalize_by_models(egm_tensor, Y_model)

    else:

        latent_space_n = latent_space
        egm_tensor_n = egm_tensor

    return latent_space_n, egm_tensor_n


def get_run_logdir():
    # Tensorboard logs name generator

    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def reshape_tensor(tensor, n_dim_input, n_dim_output):
    """
    Reshapes the tensors used during pipeline, considering that the first two dimensions are (#n batches, batch size).
    In the case of n_dim_input = 5, the last dimension is the number of channels.

    Parameters
    ----------
    tensor: tensor to reshape
    n_dim_input: input shape
    n_dim_output: desired output shape

    Returns
    -------

    """

    try:

        # case of autoencoder output: first two dimensions
        if n_dim_input == 5 and n_dim_output == 2:
            reshaped_tensor = reshape(
                tensor,
                (
                    tensor.shape[0] * tensor.shape[1],
                    tensor.shape[2] * tensor.shape[3] * tensor.shape[4],
                ),
            )
            return reshaped_tensor

        # case of regression output
        elif n_dim_input == 3 and n_dim_output == 2:
            reshaped_tensor = reshape(
                tensor, (tensor.shape[0] * tensor.shape[1], tensor.shape[2])
            )
            return reshaped_tensor

        elif n_dim_input == 5 and n_dim_output == 4:
            reshaped_tensor = reshape(
                tensor,
                (
                    tensor.shape[0] * tensor.shape[1],
                    tensor.shape[2],
                    tensor.shape[3],
                    tensor.shape[4],
                ),
            )
            return reshaped_tensor
        elif n_dim_input == 4 and n_dim_output == 5:
            reshaped_tensor = reshape(
                tensor,
                (tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3], 1),
            )

            return reshaped_tensor

        elif n_dim_input == 4 and n_dim_output == 2:
            reshaped_tensor = reshape(
                tensor,
                (tensor.shape[0], tensor.shape[1] * tensor.shape[2] * tensor.shape[3]),
            )

            return (reshaped_tensor,)

    except:
        raise (
            ValueError(
                "Input shape - Output shape combination is not implemented: check reshape_tensor documentation"
            )
        )


def interpolate_reconstruction(estimate_egms_reshaped):
    """
    This function applies interpolation to reconstructed signals which have been undersampled in the node
    dimension, considering the original number of nodes 2048.

    """
    no_nodes = estimate_egms_reshaped.shape[1]
    original_no = 2048
    ratio = original_no / no_nodes
    interpol = tf.keras.layers.UpSampling2D(size=(ratio, 1), interpolation="bilinear")(
        estimate_egms_reshaped
    )

    return interpol


def array_to_dic_by_models(dic, model_list, AF_models, all_model_names):
    """
    This function rearranges sub-realizations corresponding to originally different AF Models
    into separate fields with its corresponding name in a dictionary

    """
    new_dic = {}
    reconstruction = dic["reconstruction"]
    label = dic["label"]
    for model in model_list:
        model_name = all_model_names[model]
        key = f"model{model_name}"
        rec = reconstruction[np.in1d(AF_models, model)]
        lab = label[np.in1d(AF_models, model)]
        new_dic[key] = {
            "reconstruction": rec.tolist(),  # Convert rec to a list if necessary
            "label": lab.tolist(),  # Convert lab to a list if necessary
        }

    return new_dic
