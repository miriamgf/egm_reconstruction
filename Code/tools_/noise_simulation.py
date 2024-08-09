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
import wfdb



# %% Path Models
# %% Path Models
current = os.path.dirname(os.path.realpath(__file__))
torsos_dir = "../../../Labeled_torsos/"
directory = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"

class NoiseSimulation:

    def __init__(self, signal, SNR_em_noise = 20, SNR_white_noise=20, oclusion =None, fs=500, random_order_chuncks=True):

        self.signal = signal
        self.SNR_electrode_noise = True
        self.SNR_white_noise = True
        self.fs = fs
        self.SNR_em_noise = SNR_em_noise
        self.SNR_white_noise = SNR_white_noise
        self.random_order_chunks = True

    def configure_noise_database(self, target_signal, all_model_names, em = True, ma = False, gn = True, noise_augmentation = 2):
        '''
        This function organizes the loaded flat noise signal into
        #N_models equal groups to encapsulate the chunks by single AF models.
        This reserves a given samples of the database for each AF model and each of its leads
        
        '''
        n_models = len(all_model_names)
        dic_noise = {}

        if em:
            em_signal_flat = self.load_physionet_signals(type_noise='em')
            em_split = np.split(em_signal_flat, n_models)
            em_split_split = self.split_into_irregular_chunks(em_split, target_signal_length = len(target_signal), noise_augmentation=3)
            

            dic_noise['em'] = em_split_split
            #TODO: Data augmentation: Concatenar dentro de un mismo modelo de AF
        elif ma:
            ma_signal_flat = self.load_physionet_signals(len(target_signal), type_noise='ma')
            ma_split = np.split(ma_signal_flat, n_models)
            dic_noise['ma'] = ma_split
        
        return dic_noise

    def add_noise(self, X, AF_model_i, noise_dic, distribution_noise_mode = 2):

        # 1) En cada electrodo se asigna solo una realización de ruido
        if distribution_noise_mode ==1:
            pass

        # 2) En un mismo electrodo se agregan varias realizaciones en diferentes instantes
        if distribution_noise_mode ==2:
            if self.SNR_em_noise != None:
                
                #Flatten electrodes
                if X.ndim == 3:
                    X=X.reshape(X.shape[0], X.shape[1]*X.shape[2])

                
                em_noise = noise_dic['em'][AF_model_i]
                num_noise_chunks = len(em_noise)
                num_electrodes = X.shape[0]
                N = round(len(em_noise)//3) #Al menos en cada nodo haya 3 episodios

                #distribute un clusters of electrodes de available noise chunks
                binary_list = self.generate_list_probability_based(N)

                for electrode in num_electrodes:
                    lead_i = X[:, electrode]



                

        
    
    import random

    def generate_list_probability_based(self, N, prob_inicial=0.5, prob_incremento=0.2):
        """
        Genera una lista de 0s y 1s de longitud N, donde la probabilidad de que un elemento
        sea 1 aumenta si el elemento anterior también es 1.

        :param N: Longitud de la lista a generar.
        :param prob_inicial: Probabilidad inicial de obtener un 1 para el primer elemento.
        :param prob_incremento: Incremento de probabilidad si el elemento anterior es 1.
        :return: Lista generada de 0s y 1s.
        """
        lista = []
        prob_actual = prob_inicial

        for i in range(N):
            # Genera un 1 o un 0 basado en la probabilidad actual
            if random.random() < prob_actual:
                lista.append(1)
                # Si se generó un 1, aumenta la probabilidad para el siguiente elemento
                prob_actual = min(1.0, prob_actual + prob_incremento)
            else:
                lista.append(0)
                # Si se generó un 0, restablece la probabilidad inicial
                prob_actual = prob_inicial

        return lista

    
    def split_into_irregular_chunks(self, split_noise, target_signal_length=2500, noise_augmentation = 3 ):
        '''
        Splits a list of arrays into irregularly sized chunks based on the specified signal length.

        Parameters:
        split_noise (list of np.ndarray): A list of NumPy arrays to be split into chunks. Each array will be processed individually.
        target_signal_length (int): The target length of the signal used to determine the minimum and maximum chunk sizes. Default is 2500.

        Returns:
        list of list of np.ndarray: A list where each element is a list of chunks (NumPy arrays) derived from the corresponding array in `split_noise`.

        Description:
        This function divides each array in `split_noise` into chunks of varying sizes, ensuring that each chunk's size falls between a minimum and maximum value. 
        The minimum and maximum chunk sizes are dynamically determined based on the `target_signal_length`. The function aims to produce at least 10 chunks per array, 
        though the exact number of chunks may vary depending on the array size and the calculated chunk sizes.

        - `min_size`: Minimum chunk size, calculated as 5% of `target_signal_length`.
        - `max_size`: Maximum chunk size, calculated as 10% of `target_signal_length`.
        - `min_chunks`: Minimum number of chunks to be produced, set to 10.

        Chunks are created such that their sizes are randomly determined within the specified range, but the function ensures that the total size of chunks does not exceed 
        the length of the original array. If fewer chunks are left to create, the last chunk will absorb any remaining elements to fit the exact size of the array.

      
        '''

        # Automatically configures sizes of chunks based on signal length
        min_size = round(target_signal_length * 0.02)  # Minimum chunk size as 5% of the target signal length
        max_size = round(target_signal_length * 0.08)   # Maximum chunk size as 10% of the target signal length
        min_chunks = 10  # Minimum number of chunks desired

        new_db = []
        for arr in split_noise:
            total_size = len(arr)
            chunks = []
            start_idx = 0
            while start_idx < total_size:
                remaining_chunks = min_chunks - len(chunks)
                remaining_size = total_size - start_idx

                # Determine the size of the next chunk
                chunk_size = np.random.randint(min_size, max_size + 1)

                # Ensure we do not exceed the remaining size or number of chunks
                if chunk_size > remaining_size:
                    chunk_size = remaining_size
                if remaining_chunks == 1:
                    chunk_size = remaining_size

                # Append the chunk to the list
                chunks.append(arr[start_idx:start_idx + chunk_size])
                start_idx += chunk_size

            if noise_augmentation != 1: #
                for repeat in range(0,noise_augmentation):
                    chunks = chunks + chunks
            
            if self.random_order_chunks:
                random.shuffle(chunks)

            new_db.append(chunks)

        return new_db


    def add_white_noise(self, X, SNR=20, fs=50):
        '''
        
        '''

        X_noisy, _ = addwhitenoise(X, SNR=SNR, fs=fs)

        return X_noisy 

    

    def load_physionet_signals(self, type_noise='em', noise_augmentation=False):
        '''
        This functions loads signals from Physionet database MIT-BIH Noise Stress Test Database
        It can load three types of noise:
        * baseline wander (in record 'bw')
        * muscle artifact (in record 'ma')
        * electrode motion artifact (in record 'em')
        
        '''
        record_name = type_noise  # Nombre del registro
        path_noise_models = 'tools_/noise_models'
        record = wfdb.rdrecord(f'{path_noise_models}/{record_name}', sampfrom=self.fs, channels=[0])

        noise = record.p_signal

        if noise_augmentation:

            # Si se quiere alargar la señal flat de ruido, concatenando al final la misma señal.

            noise.extend(noise)

        return noise
    



    
 


        






