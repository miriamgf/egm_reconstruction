# This script was developed by Miguel Ángel Cámara Vázquez and Miriam Gutiérrez Fernández


import os
from tools_.noise_simulation import *
import sys, os
from tools_.tools_1 import *

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
from scipy.stats import pearsonr, spearmanr
from datetime import time
from tools_.noise_simulation import NoiseSimulation
from scripts.config import DataConfig
from tools_.oclusion import Oclussion
import time


# %% Path Models
# %% Path Models
current = os.path.dirname(os.path.realpath(__file__))
torsos_dir = "../../../Labeled_torsos/"
# directory = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"


class LoadDataset:
    """
    The LoadDataset class loads signals from original format (.mat), performs basic preprocessing in EGMs
    and BSPs and creates variables that store metadata for tracking during the training and validation process.
    Also basic preprocessing applicable and fixed for all data is performed, like EGM filtering or interpolation
    """

    def __init__(
        self,
        params,
        data_type,
        n_classes=2,
        SR=True,
        downsampling=True,
        fs=500,
        SNR_bsps=True,
        SNR_em_noise=20,
        SNR_white_noise=20,
        norm=False,
        classification=False,
        sinusoid=False,
        n_batch=50,
        patches_oclussion=None,
        directory=directory,
        unfold_code=1,
        inference=True,
    ):
        self.params = params
        self.data_type = data_type
        self.n_classes = n_classes
        self.SR = SR
        self.downsampling = downsampling
        self.fs = fs
        self.SNR_bsps = SNR_bsps
        self.SNR_em_noise = SNR_em_noise
        self.SNR_white_noise = SNR_white_noise
        self.norm = norm
        self.classification = classification
        self.sinusoid = sinusoid
        self.n_batch = n_batch
        self.patches_oclussion = patches_oclussion
        self.directory = directory
        self.unfold_code = unfold_code
        self.inference = inference

    def load_data(
        self,
        data_type,
        n_classes=2,
        SR=True,
        downsampling=True,
        fs_sub=50,
        SNR_bsps=True,
        SNR_em_noise=20,
        SNR_white_noise=20,
        norm=False,
        classification=False,
        sinusoid=False,
        n_batch=50,
        patches_oclussion=None,
        directory=directory,
        unfold_code=1,
        inference=True,
    ):
        """
        Returns y values depends of the classification model

        Parameters:
            self.data_type: '3channelTensor' -> tx6x4x3 tensor (used for CNN-based classification)
                    '1channelTensor' -> tx6x16 tensor (used for CNN-based classification)

                    'Flat' ->  tx64 matrix (used for MLP-based classification) or tx10
            self.n_classes: for classification task 1-->
                        '1' -> Rotor/no rotor
                        '2' ->  RA/LA/No rotor (2 classes)
                        '3' ->  7 regions (3 classes) + no rotor (8 classes)
            SR: If Sinus Ryhthm Model is included (True/False)
            
            norm: if normalizing is performed to models when loading them (True/False)
            classification: if classification task is performed
            sinusoid: if a database of sinusoids is generated (True/False)


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

        for subdir, dirs, files in os.walk(self.directory):
            if subdir != self.directory:
                model_name = subdir.split("/")[-1]
                if "Sinusal" in model_name and self.SR == False:
                    continue
                else:
                    all_model_names.append(model_name)

        if self.sinusoid:
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
        transfer_matrices = self.load_transfer(ten_leads=False, bsps_set=False)

        AF_model_i = 0

        len_target_signal = 2000  # configure_noise_database: hacer una estimación de la longitud del array de BSPMs
        Noise_Simulation = NoiseSimulation(
            params = self.params,
            SNR_em_noise=self.SNR_em_noise,
            SNR_white_noise=self.SNR_white_noise,
            oclusion=None,
            fs=self.fs,
        )  # instance of class

        test_models_deterministic = [
            "LA_PLAW_140711_arm",
            "LA_RSPV_CAF_150115",
            "Simulation_01_200212_001_  5",
            "Simulation_01_200212_001_ 10",
            "Simulation_01_200316_001_  3",
            "Simulation_01_200316_001_  4",
            "Simulation_01_200316_001_  8",
            "Simulation_01_200428_001_004",
            "Simulation_01_200428_001_008",
            "Simulation_01_200428_001_010",
            "Simulation_01_210119_001_001",
            "Simulation_01_210208_001_002",
        ]
        if self.inference:
            all_model_names = test_models_deterministic

        noise_database = Noise_Simulation.configure_noise_database(
            len_target_signal,
            all_model_names,
            em=True,
            ma=False,
            gn=True,
        )

        for model_name in all_model_names:

            if self.inference:
                if model_name not in model_name:
                    break
            print("Loading model", model_name, "......")

            # %% 1)  Compute EGM of the model
            egms = self.load_egms(model_name, self.sinusoid)

            # 1.1) Discard models <1500
            # if len(egms[1])<1500:
            # continue

            # 1.2)  EGMs filtering.
            x = self.ECG_filtering(egms, fs=self.fs)

            # 1.3 Normalize EGMS
            if self.norm:

                high = 1
                low = -1

                mins = np.min(x, axis=0)
                maxs = np.max(x, axis=0)
                rng = maxs - mins

                bsps_64_n = high - (((high - low) * (maxs - x)) / rng)

            # 2) Compute the Forward problem with each of the transfer matrices
            for matrix in transfer_matrices:

                # Forward problem
                y = self.forward_problem(x, matrix[0])
                bsps_64 = y[matrix[1].ravel(), :]
                bsps_64_or = bsps_64
                bsps_64_filt = bsps_64_or

                plt.figure(figsize=(20, 7))
                plt.subplot(2, 1, 1)
                plt.plot(x[0, 0:2000])
                plt.subplot(2, 1, 2)
                plt.plot(y[0, 0:2000])

                plt.title(model_name)
                os.makedirs("output/figures/Noise_module/", exist_ok=True)
                plt.savefig("output/figures/input_output/forward_problem.png")

                # RESAMPLING signal to fs= fs_sub
                if self.downsampling:
                    bsps_64 = signal.resample_poly(
                        bsps_64_filt, self.fs_sub, 500, axis=1
                    )
                    x_sub = signal.resample_poly(x, self.fs_sub, 500, axis=1)

                    plt.figure(figsize=(20, 7))
                    plt.plot(x[0, 0:2000])
                    plt.title(model_name)
                    plt.savefig("output/figures/input_output/subsample.png")

                else:

                    bsps_64 = bsps_64_filt
                    x_sub = x

                if self.classification:

                    y_labels = self.get_labels(self.n_classes, model_name)
                    y_labels_list = y_labels.tolist()

                    # RESAMPLING labels to fs= fs_sub
                    if self.downsampling:
                        y_labels = sample(y_labels, len(bsps_64[1]))

                    y_model = np.full(len(y_labels), n_model)
                    Y_model.extend(y_model)
                    Y.extend(y_labels)
                    Y_model.extend(y_model)

                else:

                    Y.extend(np.full(len(x_sub), 0))

                # RESHAPE TO TENSOR

                if self.data_type == "3channelTensor":

                    tensor_model = self.get_tensor_model(
                        bsps_64, tensor_type="3channel"
                    )
                    X.extend(tensor_model)

                elif self.data_type == "1channelTensor":

                    tensor_model = self.get_tensor_model(
                        bsps_64, tensor_type="1channel", unfold_code=self.unfold_code
                    )

                    # Noise
                    start_time = time.time()

                    if self.SNR_bsps != None:
                        # New noise module
                        # num_patches must be maximum 16 (for 64 electrodes)
                        tensor_model_noisy, map_distribution_noise = (
                            Noise_Simulation.add_noise(
                                tensor_model,
                                AF_model_i,
                                noise_database,
                                num_patches=10,
                                distribution_noise_mode=2,
                                n_noise_chunks_per_signal=3,
                            )
                        )

                    # 5) Filter AFTER adding noise

                    tensor_model_filt = self.ECG_filtering(
                        tensor_model_noisy, order=3, fs=500, f_low=3, f_high=30
                    )
                    tensor_model = tensor_model_filt

                    plt.figure(figsize=(20, 5))
                    plt.plot(tensor_model_noisy[0:1000, 0, 0], label="Noisy")
                    plt.plot(tensor_model[0:1000, 0, 0], label="Original")
                    plt.plot(tensor_model_filt[0:1000, 0, 0], label="Cleaned")
                    plt.legend()
                    plt.savefig("output/figures/Noise_module/filtered_vs_original.png")

                    # Turn off electrodes
                    if self.patches_oclussion != "PT":
                        oclussion = Oclussion(
                            tensor_model, patches_oclussion=self.patches_oclussion
                        )
                        tensor_model = oclussion.turn_off_patches()

                    # Interpolate
                    tensor_model = interpolate_2D_array(tensor_model)

                    plt.figure(figsize=(20, 7))
                    plt.plot(x_sub[0, 0:2000])
                    plt.plot(tensor_model[0:2000, 0, 0])
                    plt.title(model_name)
                    plt.savefig("output/figures/input_output/before_truncate.png")

                    # Truncate length to be divisible by the batch size
                    # tensor_model, length_list, x_sub = truncate_length_bsps(self.n_batch, tensor_model, length_list, x_sub)

                    X.extend(tensor_model)
                    egm_tensor.extend(x_sub.T)

                    # plt.figure(figsize=(20, 7))
                    # plt.plot(x_sub[0, 0:2000])
                    # plt.plot(tensors_model[0:2000, 0, 0])
                    # plt.title(model_name)
                    # plt.savefig(model_name)

                    # plt.savefig('output/figures/input_output/saving_truncate.png')

                else:
                    X.extend(bsps_64.T)
                    egm_tensor.extend(x_sub.T)

                if not self.classification:
                    y_model = np.full(len(tensor_model), n_model)
                    Y_model.extend(y_model)

                    # Count AF Model
                    AF_model_i_array = np.full(len(tensor_model), AF_model_i)
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

    def load_transfer(self, ten_leads=False, bsps_set=False):
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
                BSP_pos = scipy.io.loadmat(torsos_dir + torso).get("torso")["leads"][
                    0, 0
                ]
            elif bsps_set == True:
                torso_electrodes = loadmat(self.directory + "torso_electrodes.mat")
                BSP_pos = torso_electrodes["torso_electrodes"][0]
                # BSP_pos = get_bsps_192 (torso, False)
            else:
                BSP_pos = scipy.io.loadmat(torsos_dir + torso).get("torso")[
                    "bspmcoord"
                ][0, 0]

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

    def load_egms(self, model_name, directory, sinusoid=False):
        """
        Load electrograms and select 2500 time instants and 2048 nodes

        Parameters:
            model (str): Model to load

        Returns:
            x: Electrograms for the selected model
        """
        if sinusoid:

            EG = sinusoids_generator(2048, 2500, fs=self.fs).T
        else:

            try:
                EG = np.transpose(
                    np.array(
                        (h5py.File(self.directory + model_name + "/EGMs.mat", "r")).get(
                            "x"
                        )
                    )
                )
            except:
                EG = scipy.io.loadmat(self.directory + model_name + "/EGMs.mat").get(
                    "x"
                )

        return EG

    def forward_problem(self, EGMs, MTransfer):
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

    def get_bsps_192(self, model_name, ten_leads=False):
        """
        Reduce the 659 BSPs to 192, 3 BSPSs for each node.

        Parameters:
        model_name: name of the model

        Return:
        bsps_192: array with 192 BSPs

        """
        torso_electrodes = loadmat(self.directory + "/torso_electrodes.mat")
        torso_electrodes = torso_electrodes["torso_electrodes"][0]

        bsps = loadmat(self.directory + model_name + "/BSPs.mat")
        bsps = bsps["y"]

        bsps_192 = []
        for i in torso_electrodes:
            bsps_192.append(bsps[i, :])

        return np.array(bsps_192)

    def get_tensor_model(self, bsps_64, tensor_type="3channel", unfold_code=1):
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
                tensor_model = self.get_subtensor_2(bsps_64[:, t], tensor_type)
            else:
                tensor_model = self.get_subtensor_2(
                    bsps_64[:, t], tensor_type, unfold_code=1
                )
            if all_tensors.size == 0:
                all_tensors = tensor_model
            else:
                all_tensors = np.concatenate((all_tensors, tensor_model), axis=0)

        return all_tensors

    def get_bsps_64(self, bsps_192, seed="Y"):
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

    def get_labels(self, class_type, model_name):
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

        regions = loadmat(self.directory + "/regions.mat")
        regions = regions["regions"][0]

        path_driver_position = self.directory + model_name + "/driver_position.mat"
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
            for subdir, dirs, files in os.walk(self.directory):
                for subdir2, dirs2, files2 in os.walk(subdir):
                    if subdir != self.directory:
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

    def get_subtensor_2(self, bsps_64_t, tensor_type="3_channel", unfold_code=1):
        """
        Get (6 x 4 x 3) tensor for 1 instance of time.

        Parameters:
        bsps_64_t: 1 instance of time of bsps_64

        Return:
        subtensor: 6 x 4 x 3 matrix
        """

        patches = self.get_patches_name(bsps_64_t)

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

            if unfold_code == 1:
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

            if unfold_code == 2:

                unfold_order = ["L_side", "front", "L_side", "back"]
                new_subtensor = np.zeros(subtensor.shape)

                patch_indices = {
                    "front": (slice(0, 6), slice(0, 4)),
                    "R_side": (slice(0, 6), slice(4, 6)),
                    "back": (slice(0, 6), slice(6, 10)),
                    "L_side": (slice(0, 6), slice(10, 12)),
                }
                for segment in unfold_order:
                    new_subtensor[patch_indices[segment]] = subtensor[
                        patch_indices[segment]
                    ]

                new_subtensor = subtensor

            subtensor = subtensor.reshape(1, 6, 16)

        return subtensor

    def get_patches_name(self, bsps_64):
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

    def ECG_filtering(self, signal, fs, order=2, f_low=3, f_high=30):
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
        # sig_temp = signal

        # Bandpass filtering
        b, a = sigproc.butter(
            order, [f_low / round((self.fs / 2)), f_high / round((self.fs / 2))], btype="bandpass"
        )

        proc_ECG_EGM = np.zeros(sig_temp.shape)
        if sig_temp.ndim == 3:
            for i in range(sig_temp.shape[1]):
                for j in range(sig_temp.shape[2]):
                    # for index in range(sig_temp.shape[0]):
                    proc_ECG_EGM[:, i, j] = sigproc.filtfilt(b, a, sig_temp[:, i, j])
        else:
            for index in range(0, sig_temp.shape[0]):
                proc_ECG_EGM[index, :] = sigproc.filtfilt(b, a, sig_temp[index, :])

        return proc_ECG_EGM

    def __call__(self, verbose=False, all=False):
        """Calls the Load class."""
        return self.load_data(
            data_type=self.data_type,
            n_classes=self.n_classes,
            SR=self.SR,
            downsampling=self.downsampling,
            SNR_bsps=self.SNR_bsps,
            SNR_em_noise=self.SNR_em_noise,
            SNR_white_noise=self.SNR_white_noise,
            norm=self.norm,
            classification=self.classification,
            sinusoid=self.sinusoid,
            n_batch=self.n_batch,
            patches_oclussion=self.patches_oclussion,
            directory=self.directory,
            unfold_code=self.unfold_code,
            inference=self.inference,
        )
