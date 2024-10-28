# -*- coding: utf-8 -*-
# """
# Created on Thu Jan 28 11:26:06 2021

# @author: Miguel Ángel
# """

# This script contains general tools used transversally through all pipeline 


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
from tools_.noise_simulation import NoiseSimulation
from scripts.config import DataConfig
from tools_.oclusion import Oclussion
import time
#from noise_simulation import *

# %% Path Models
# %% Path Models
current = os.path.dirname(os.path.realpath(__file__))
torsos_dir = "../../../Labeled_torsos/"
#directory = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
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

def truncate_length_bsps(n_batch, tensors_model, length_list, x_sub):
    batch_size = n_batch
    if tensors_model.shape[0] % batch_size != 0:
        trunc_val = np.floor_divide(tensors_model.shape[0], batch_size)
        tensors_model = tensors_model[0 : batch_size * trunc_val, :, :]
        x_sub = x_sub[:, 0 : batch_size * trunc_val]

    length_list.append(tensors_model.shape[0])

    return tensors_model, length_list, x_sub


def interpolate_2D_array(tensors_model):
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
    return tensors_model


def sample(input, count):
    ss = float(len(input)) / count
    return [input[int(floor(i * ss))] for i in range(count)]


def plot_load_data(egms, x, bsps_64_noise, bsps_64_filt, bsps_64_or, n_model, bsps_64):

    # PLOT
    if n_model == 160:
        print("Ploting...")
        y_50 = np.linspace(0, 4, 200, endpoint=False)
        y_500 = np.linspace(0, 4, 2000, endpoint=False)
        #plt.figure(layout="tight")
        #plt.subplot(4, 1, 1)
        #plt.plot(y_500, egms[0, 0:2000], label="original")
        #plt.plot(y_500, x[0, 0:2000], label="filtered")
        #plt.xlabel("Seconds")
        #plt.legend()
        #plt.title("Original vs filtered EGM")
        #plt.subplot(4, 1, 2)
        #plt.plot(
            #y_500,
            #bsps_64_noise[0, 0:2000],
            #alpha=0.75,
            #label="2. BSPM noise added (20 db)",
        #)
        #plt.plot(y_500, bsps_64_filt[0, 0:2000], label="3. BSPM filtered")
        #plt.plot(y_50, bsps_64[0, 0:200], label="4. BSPM subsampled to 50 Hz")
        #plt.plot(y_500, bsps_64_or[0, 0:2000], alpha=0.75, label="1. BSPM original")

        #plt.title("BSPM in node 0")
        #plt.xlabel("Seconds")
        #plt.legend()
        #plt.subplot(4, 1, 3)
        #plt.plot(
        #    y_500,
        #    bsps_64_noise[20, 0:2000],
        #    alpha=0.75,
        #    label="2. BSPM noise added (20 db)",
        #)
        ##plt.plot(y_500, bsps_64_filt[20, 0:2000], label="3. BSPM filtered")
        #plt.plot(y_50, bsps_64[20, 0:200], label="4. BSPM subsampled to 50 Hz")
        #plt.plot(y_500, bsps_64_or[20, 0:2000], alpha=0.75, label="1. BSPM original")

        #plt.title("BSPM in node 20")
        #plt.xlabel("Seconds")
        #plt.legend()
        #plt.subplot(4, 1, 4)
        #plt.plot(
            #y_500,
            #bsps_64_noise[50, 0:2000],
            #alpha=0.75,
            #label="2. BSPM noise added (20 db)",
        #)
        #plt.plot(y_500, bsps_64_filt[50, 0:2000], label="3. BSPM filtered")
        #plt.plot(y_50, bsps_64[50, 0:200], label="4. BSPM subsampled to 50 Hz")
        #plt.plot(y_500, bsps_64_or[50, 0:2000], alpha=0.75, label="1. BSPM original")

        #plt.title("BSPM in node 50")
        #plt.xlabel("Seconds")
        #plt.legend()
        #plt.show()
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

def data_generator(data, batch_size=1):
    num_batches = data.shape[0]
    while True:
        for i in range(num_batches):
            yield data[i:i + batch_size]


def get_run_logdir():
    # Tensorboard logs name generator

    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
