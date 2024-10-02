# This script was developed Miriam Gutiérrez Fernández
# """
import os
from tools_.noise_simulation import *
import sys, os
from tools_.tools_1 import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.config import TrainConfig_1
import os
from tools_.noise_simulation import *
import sys, os
from tools_.tools_1 import *
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


class Preprocess_Dataset:
    '''
    The Preprocess_Dataset class preprocess dataset loaded previously. Among the tasks that it performs:
        - Temporal Downsampling
        - Split into train and test
        - Batch generation
        - Normalization
        -Reshape to fit into input layers of the network

    '''

    def __init__(self, X_1channel, egm_tensor, AF_models, Y_model, dic_vars, Y, all_model_names, transfer_matrices ):

        self.X_1channel = X_1channel
        self.egm_tensor = egm_tensor
        self.AF_models = AF_models
        self.Y_model = Y_model
        self.dic_vars = dic_vars
        self.Y = Y
        self.all_model_names= all_model_names
        self.transfer_matrices= transfer_matrices

    def preprocess_main(self, X_1channel, egm_tensor, AF_models, Y_model):
        '''
        This function defines the main steps to perform preprocessing over target signals
        1) Apply downsampling and truncate length according to batch size
        2) Normalization between -1 and 1
        3) Clean Nans generated during Noise Generation (zero-Patches)
        4) Train/Test/Val split
        5) Batch generation 


        '''
        # Downsampling and truncate
        X_1channel, egm_tensor, AF_models, Y_model = self.preprocess_compression(X_1channel, egm_tensor, AF_models,Y_model, 
                                            fs_sub = DataConfig.fs_sub, batch_size= TrainConfig_1.batch_size_1, 
                                            downsampling = True)

        # Normalize BSPS and EGM
        X_1channel = normalize_by_models(X_1channel, Y_model)
        egm_tensor = normalize_by_models(egm_tensor, Y_model)
        X_1channel = np.nan_to_num(X_1channel, nan=0.0) #Nans generated during noise addition

        plt.figure()
        plt.plot(X_1channel[0:200, 0, 0], label='bsps')
        plt.plot(egm_tensor[0:200, 0], label='egm')
        plt.legend()
        os.makedirs('output/figures/input_output/', exist_ok=True)
        plt.savefig('output/figures/input_output/norm.png')


        new_items = {'Original_X_1channel': X_1channel, 'Y': self.Y, 'Y_model': Y_model, 'egm_tensor': egm_tensor,
                    'AF_models': AF_models, 'all_model_names': self.all_model_names, 'transfer_matrices': self.transfer_matrices,
                    'X_1channel_norm': X_1channel}
        self.dic_vars.update(new_items)

        # Train/Test/Val Split
        random_split = True
        print('Splitting...')
        x_train, x_test, x_val, train_models, test_models, val_models, AF_models_train, AF_models_test, AF_models_val, BSPM_train, BSPM_test, BSPM_val = self.train_test_val_split_Autoencoder(
            X_1channel,AF_models, Y_model, self.all_model_names, random_split=True, train_percentage=0.90, test_percentage=0.2, deterministic = False)

        print('TRAIN SHAPE:', x_train.shape, 'models:', train_models)
        print('TEST SHAPE:', x_test.shape, 'models:', test_models)
        print('VAL SHAPE:', x_val.shape, 'models:', val_models)

        new_items = {'x_train_raw': x_train, 'x_test_raw': x_test, 'x_val_raw': x_val, 'train_models': train_models,
                    'BSPM_train': BSPM_train, 'BSPM_test': BSPM_test, 'BSPM_val': BSPM_val,
                    'test_models': test_models, 'AF_models_train': AF_models_train, 'AF_models_test': AF_models_test,
                    'AF_models_val': AF_models_val}
        self.dic_vars.update(new_items)


        x_train, x_test, x_val = self.preprocessing_autoencoder_input(x_train, x_test, x_val, TrainConfig_1.batch_size_1)

        new_items = {'x_train': x_train, 'x_test': x_test, 'x_val': x_val}
        self.dic_vars.update(new_items)

        y_train, y_test, y_val = self.preprocessing_y(egm_tensor,Y_model, AF_models, train_models,test_models, val_models, TrainConfig_1.batch_size_1, norm =False)

        plt.figure()
        plt.plot(x_train[0, :, 0, 0, 0], label = 'bsps')
        plt.plot(y_train[0, :,0], label = 'egm')
        plt.legend()
        os.makedirs('output/figures/input_output/', exist_ok=True)
        plt.savefig('output/figures/input_output/preprocessing.png')

        return x_train, x_test, x_val, y_train, y_test, y_val, self.dic_vars, BSPM_train, BSPM_test, BSPM_val, AF_models_train, AF_models_test, AF_models_val, train_models, test_models, val_models

    
    def preprocess_compression(self, X_1channel, egm_tensor, AF_models, Y_model,  fs_sub, batch_size, downsampling = True):

        '''
        This function process loaded data (BSPS, EGMs and metadata) to perform downsampling and 
        truncate the length of the arrays according to the specified batch size

        This operation must be accomplished by AF model to ensure that the truncation alineates when splitting
        into train - test - val

        '''

        AF_models = np.array(AF_models)

        new_X_1channel = []
        new_egm_tensor = []
        new_AF_models = []
        new_Y_model = []

        for AF_model_i in np.unique(AF_models):

            X_1channel_from_model_i = X_1channel[np.where(AF_models == AF_model_i)]
            egm_tensor_from_model_i = egm_tensor[np.where(AF_models == AF_model_i)]
            AF_models_from_model_i = AF_models[np.where(AF_models == AF_model_i)]
            Y_model_from_model_i = Y_model[np.where(AF_models == AF_model_i)]



            if downsampling:
                #X_1channel_sub = signal.resample_poly(X_1channel, fs_sub, 500, axis=0)
                #egm_tensor_sub = signal.resample_poly(egm_tensor, fs_sub, 500, axis=0)
                #AF_models_sub = signal.resample_poly(AF_models, fs_sub, 500, axis=0)
                #Y_model_sub = signal.resample_poly(Y_model, fs_sub, 500, axis=0)
                downsampling_factor = int(500 / fs_sub)     
                X_1channel_sub = X_1channel_from_model_i[::downsampling_factor]
                egm_tensor_sub = egm_tensor_from_model_i[::downsampling_factor]       
                AF_models_sub = AF_models_from_model_i[::downsampling_factor]
                Y_model_sub = Y_model_from_model_i[::downsampling_factor]

            X_1channel_sub = self.truncate_length_by_batch_size(batch_size, X_1channel_sub)
            egm_tensor_sub = self.truncate_length_by_batch_size(batch_size, egm_tensor_sub)
            AF_models_sub = self.truncate_length_by_batch_size(batch_size, AF_models_sub)
            Y_model_sub = self.truncate_length_by_batch_size(batch_size, Y_model_sub)

            new_X_1channel.extend(X_1channel_sub)
            new_egm_tensor.extend(egm_tensor_sub)
            new_AF_models.extend(AF_models_sub)
            new_Y_model.extend(Y_model_sub)

        return np.array(new_X_1channel), np.array(new_egm_tensor),  list(new_AF_models), np.array(new_Y_model)


    def truncate_length_by_batch_size(self, batch_size, signal_data):

        if signal_data.shape[0] % batch_size != 0:
            trunc_val = np.floor_divide(signal_data.shape[0], batch_size)
            signal_data = signal_data[0 : batch_size * trunc_val, ...]
        return signal_data
    
    def preprocessing_autoencoder_input(self, x_train, x_test, x_val, n_batch):
        """
        Function to preprocess input to fit autoencoder shapes

        Autoencoder Input shape = [# batches, batch_size, 12, 32, 1]
        Autoencoder Output shape = [# batches, batch_size, 12, 32, 1]
        Autoencoder Latent space shape = [# batches, batch_size, 3, 4, 12]

        Parameters
        ----------
        x_train: numpy array containing training data (shape:
        x_test

        Returns
        x_train
        -------

        """
        try:
            # Reshape and batch_generation to fit Conv (Add 1 dimension)

            x_train_reshaped = reshape(
                x_train,
                (
                    int(len(x_train) / n_batch),
                    n_batch,
                    x_train.shape[1],
                    x_train.shape[2],
                    1,
                ),
            )
            x_test_reshaped = reshape(
                x_test,
                (int(len(x_test) / n_batch), n_batch, x_test.shape[1], x_test.shape[2], 1),
            )
            x_val_reshaped = reshape(
                x_val,
                (int(len(x_val) / n_batch), n_batch, x_val.shape[1], x_val.shape[2], 1),
            )

        except:

            raise Exception(
                "Input shape for autoencoder 3D is [# batches, batch_size, 12, 32, 1]. Current input shape is: ",
                x_train.shape,
            )

        return x_train_reshaped, x_test_reshaped, x_val_reshaped


    def preprocessing_regression_input(
        self, 
        latent_vector_train,
        latent_vector_test,
        latent_vector_val,
        train_models,
        test_models,
        val_models,
        Y_model,
        egm_tensor,
        AF_models,
        n_batch,
        random_split=True,
        norm=False,
    ):
        """
        Regression Input shape: [# batches, batch_size, 3, 4, 12]
        Regression Output shape: [# batches, batch_size, #nodes]

        Parameters
        ----------
        latent_vector_train: [# batches, batch_size, 3, 4, 12]
        latent_vector_test: [# batches, batch_size, 3, 4, 12]
        latent_vector_val: [# batches, batch_size, 3, 4, 12]
        Y_model
        egm_tensor

        Returns
        y_train, y_test, y_val, x_train_ls, x_test_ls, x_val_ls: x and x
        n_nodes: number of nodes are predicted (original geometry: 2048 nodes in heart geom)

        """

        try:

            latent_space_n, egm_tensor_n = self.preprocess_latent_space(
                latent_vector_train,
                latent_vector_test,
                latent_vector_val,
                train_models,
                test_models,
                val_models,
                Y_model,
                egm_tensor,
                dimension=5,
                norm=True,
            )
        except:

            raise Exception(
                "Input shape for Regression network is [# batches, batch_size, 3, 4, 12]. Current input shape is: ",
                latent_vector_train.shape,
            )

        # Split egm_tensor
        if random_split:
            x_train = latent_space_n[np.in1d(AF_models, train_models)]
            x_test = latent_space_n[np.in1d(AF_models, test_models)]
            x_val = latent_space_n[np.in1d(AF_models, val_models)]
        else:
            x_train = latent_space_n[np.where((Y_model >= 1) & (Y_model <= 200))]
            x_test = latent_space_n[np.where((Y_model > 180) & (Y_model <= 244))]
            x_val = latent_space_n[np.where((Y_model > 244) & (Y_model <= 286))]

        # Split EGM (Label)
        if random_split:
            y_train = egm_tensor_n[np.in1d(AF_models, train_models)]
            y_test = egm_tensor_n[np.in1d(AF_models, test_models)]
            y_val = egm_tensor_n[np.in1d(AF_models, val_models)]

        else:

            y_train = egm_tensor_n[np.where((Y_model >= 1) & (Y_model <= 200))]
            y_test = egm_tensor_n[np.where((Y_model > 180) & (Y_model <= 244))]
            y_val = egm_tensor_n[np.where((Y_model > 244) & (Y_model <= 286))]

        # %% Subsample EGM nodes

        y_train_subsample = y_train[:, 0:2048:3]
        y_test_subsample = y_test[:, 0:2048:3]
        y_val_subsample = y_val[:, 0:2048:3]

        n_nodes = y_train_subsample.shape[1]

        # Batch generation
        x_train_ls = reshape(
            x_train,
            (
                int(len(x_train) / n_batch),
                n_batch,
                x_train.shape[1],
                x_train.shape[2],
                x_train.shape[3],
            ),
        )
        x_test_ls = reshape(
            x_test,
            (
                int(len(x_test) / n_batch),
                n_batch,
                x_test.shape[1],
                x_test.shape[2],
                x_test.shape[3],
            ),
        )
        x_val_ls = reshape(
            x_val,
            (
                int(len(x_val) / n_batch),
                n_batch,
                x_val.shape[1],
                x_val.shape[2],
                x_val.shape[3],
            ),
        )

        y_train = reshape(
            y_train_subsample,
            (int(len(y_train_subsample) / n_batch), n_batch, y_train_subsample.shape[1]),
        )
        y_test = reshape(
            y_test_subsample,
            (int(len(y_test_subsample) / n_batch), n_batch, y_test_subsample.shape[1]),
        )
        y_val = reshape(
            y_val_subsample,
            (int(len(y_val_subsample) / n_batch), n_batch, y_val_subsample.shape[1]),
        )

        return y_train, y_test, y_val, x_train_ls, x_test_ls, x_val_ls, n_nodes


    def preprocessing_y(
        self, 
        egm_tensor,
        Y_model,
        AF_models,
        train_models,
        test_models,
        val_models,     
        n_batch,
        norm=False,
        random_split=True
    ):
        # Normalize
        if norm:
            egm_tensor_n = []
            for model in np.unique(Y_model):

                # 2. Normalize egm (output)
                arr_to_norm_egm = egm_tensor[
                    np.where((Y_model == model))
                ]  # select window of signal belonging to model i
                egm_tensor_norm = normalize_array(arr_to_norm_egm, 1, -1)
                egm_tensor_n.extend(egm_tensor_norm)  # Add to new norm array

            egm_tensor_n = np.array(egm_tensor_n)

        else:

            egm_tensor_n = egm_tensor

        # Split EGM (Label)
        if random_split:
            y_train = egm_tensor_n[np.in1d(AF_models, train_models)]
            y_test = egm_tensor_n[np.in1d(AF_models, test_models)]
            y_val = egm_tensor_n[np.in1d(AF_models, val_models)]

        else:

            y_train = egm_tensor_n[np.where((Y_model >= 1) & (Y_model <= 200))]
            y_test = egm_tensor_n[np.where((Y_model > 180) & (Y_model <= 244))]
            y_val = egm_tensor_n[np.where((Y_model > 244) & (Y_model <= 286))]

        # %% Subsample EGM nodes

        if DataConfig.n_nodes_regression == 2048:
            N = 1
        elif DataConfig.n_nodes_regression == 1024:
            N = 2
        elif DataConfig.n_nodes_regression == 682:
            N = 3
        elif DataConfig.n_nodes_regression == 512:
            N = 4



        y_train_subsample = y_train[:, 0:2048:N]  #:, 0:2048:2] --> 1024
        y_test_subsample = y_test[:, 0:2048:N]
        y_val_subsample = y_val[:, 0:2048:N]

        n_nodes = y_train_subsample.shape[1]

        y_train = reshape(
            y_train_subsample,
            (int(len(y_train_subsample) / n_batch), n_batch, y_train_subsample.shape[1]),
        )
        y_test = reshape(
            y_test_subsample,
            (int(len(y_test_subsample) / n_batch), n_batch, y_test_subsample.shape[1]),
        )
        y_val = reshape(
            y_val_subsample,
            (int(len(y_val_subsample) / n_batch), n_batch, y_val_subsample.shape[1]),
        )

        return y_train, y_test, y_val
    
    def train_test_val_split_Autoencoder(self,
        X_1channel,
        AF_models,
        BSPM_Models,
        all_model_names,
        random_split,
        train_percentage,
        test_percentage,
        deterministic = True
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

            if deterministic:
                # Deterministic assignation
                train_models_deterministic= ['RA_RAA_141230', 'Simulation_01_190502_001_003', 'Simulation_01_190502_001_004', 
                'Simulation_01_190502_001_006', 'Simulation_01_190619_001_001', 'Simulation_01_190619_001_002', 
                'Simulation_01_190619_001_003', 'Simulation_01_190619_001_004', 'Simulation_01_190717_001_001', 
                'Simulation_01_190717_001_002', 'Simulation_01_190717_001_003', 'Simulation_01_190717_001_004', 
                'Simulation_01_191001_001_001', 'Simulation_01_191001_001_002', 'Simulation_01_191001_001_005', 
                'Simulation_01_191001_001_007', 'Simulation_01_200212_001_  1', 'Simulation_01_200212_001_  2', 
                'Simulation_01_200212_001_  4', 'Simulation_01_200212_001_  6', 'Simulation_01_200212_001_  7', 
                'Simulation_01_200212_001_  9', 'Simulation_01_200316_001_  1', 'Simulation_01_200316_001_  5', 
                'Simulation_01_200316_001_  7', 'Simulation_01_200428_001_001', 'Simulation_01_200428_001_002', 
                'Simulation_01_200428_001_003', 'Simulation_01_200428_001_005', 'Simulation_01_200428_001_006', 
                'Simulation_01_200428_001_007', 'Simulation_01_200428_001_009', 'Simulation_01_201223_001_002', 
                'Simulation_01_210209_001_003', 'Simulation_01_210210_001_001', 'TwoRotors_181219']

                val_models_deterministic= ['LA_RIPV_150121', 'RA_RAFW_140807', 'Simulation_01_190502_001_005',
                'Simulation_01_200212_001_  8', 'Sinusal_150629']

                test_models_deterministic = ['LA_PLAW_140711_arm', 'LA_RSPV_CAF_150115', 'Simulation_01_200212_001_  5',
                'Simulation_01_200212_001_ 10', 'Simulation_01_200316_001_  3',
                'Simulation_01_200316_001_  4', 'Simulation_01_200316_001_  8',
                'Simulation_01_200428_001_004', 'Simulation_01_200428_001_008',
                'Simulation_01_200428_001_010', 'Simulation_01_210119_001_001',
                'Simulation_01_210208_001_002']

                train_models, test_models, val_models = [],[],[]
                for elemento in train_models_deterministic:
                    if elemento in all_model_names:
                        train_models.append(all_model_names.index(elemento))
                
                for elemento in test_models_deterministic:
                    if elemento in all_model_names:
                        test_models.append(all_model_names.index(elemento))

                for elemento in val_models_deterministic:
                    if elemento in all_model_names:
                        val_models.append(all_model_names.index(elemento))
            else:

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
    
    def preprocess_latent_space(self,
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

    def __call__(self, verbose=False, all=False):
        """Calls the Preprocess class."""
        return self.preprocess_main(self.X_1channel,self.egm_tensor,self.AF_models,self.Y_model,
        )


    
