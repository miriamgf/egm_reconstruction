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
from tools import *


# %% Path Models
# %% Path Models
current = os.path.dirname(os.path.realpath(__file__))
torsos_dir = "../../../Labeled_torsos/"
directory = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"

class NoiseSimulation:

    def __init__(self, signal, SNR_em_noise = 20, SNR_white_noise=20, oclusion =None, fs=500, random_order_chuncks=True):

        self.signal = signal
        self.SNR_em = True
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

    def add_noise(self, X, AF_model_i, noise_dic, distribution_noise_mode = 2, n_episodes_in_lead=3, n_noise_chunks_per_signal = 4):

        # 1) En cada electrodo se asigna solo una realización de ruido
        if distribution_noise_mode ==1:
            pass

        # 2) En un mismo electrodo se agregan varias realizaciones en diferentes instantes
        if distribution_noise_mode ==2:
            # EM
            if self.SNR_em_noise != None:
                
                em_noise = noise_dic['em'][AF_model_i]
                num_noise_chunks = len(em_noise)
                #if the patch size is fixed to (2x2), then the maximum number of patches is the total number of chunks //4
                min_num_patches_2_2 = num_noise_chunks//4 #this is the number of patches if only 1 noise chunks is assigned per lead
                num_patches = min_num_patches_2_2//n_noise_chunks_per_signal

                #distribute a clusters of electrodes of available noise chunks in patches of size (2,2)
                binary_map_2d = self.generate_array_with_non_overlapping_patches(array_shape=(X.shape[1], X.shape[2]),
                                                                 patch_size = (2,2), num_patches = num_patches )

                indices = np.argwhere(binary_map_2d == 1)
                #leads = X[:, indices[:, 0], indices[:, 1]]
                for column_i, row_i in indices:
                    lead_i = X[:, column_i, row_i]
                    lead_i_noisy, em_noise = self.insert_noise_in_signal(lead_i, em_noise, self.SNR_em_noise,
                                                                        n_noise_chunks_per_signal)

                    X[:, indices[:, 0], indices[:, 1]] == lead_i_noisy #Assign new noisy value

        return lead_i_noisy

    def insert_noise_in_signal(self, clean_signal, noise_list, SNR, num_chunks_per_clean_signal= 3, normalize_noise=False):

        #Create a disperse signal of same length with N noise realizations
        noise_signal = np.zeros(len(clean_signal))
        onset_index = random.randint(0, round(len(clean_signal)*0.2))

        for chunk in range(0,num_chunks_per_clean_signal):

            chunk_i = noise_list.pop(0)
            chunk_i= chunk_i.flatten()

            #normalize
            plt.figure()
            plt.plot(chunk_i, label='before norm')
            if normalize_noise:
                chunk_i = normalize_array(chunk_i, high=1, low=-1, axis_n=0)
                
                plt.plot(chunk_i, label = 'after norma')
                plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Noise_module/norm.png')
                plt.show()

            #if onset and chunk addition surpasses the clean signal length, then it tries to add it just after the last chunk
            if onset_index + len(chunk_i) > len(clean_signal) and onset_index_previous + len(chunk_i) <= len(clean_signal):
                remaining_samples = len(clean_signal) - onset_index
                chunk_i = chunk_i[0:remaining_samples]
            #if it is not possible because either way it surpasses the clean signal length, it does not add more chunks
            elif onset_index + len(chunk_i) > len(clean_signal) and onset_index_previous + len(chunk_i) > len(clean_signal):
                break
            #Id the onset index is directly out of range, no more chunks are used
            elif onset_index > len(clean_signal):
                break
            else:
                noise_signal[onset_index:onset_index+len(chunk_i)] = chunk_i
            onset_index_previous = onset_index

            #Update onset
            onset_index= onset_index + len(chunk_i) +  random.randint(0, round(len(clean_signal)*0.3))
            if onset_index > len (clean_signal):
                print('Except: onset_index > len (clean_signal)')
                continue
        
        #scale 
        PowerInSigdB = 10 * np.log10(np.mean(np.power(np.abs(noise_signal), 2)))
        sigma = np.sqrt(np.power(10, (PowerInSigdB - SNR) / 10))
        noise_signal = sigma*noise_signal 

        
        noisy_signal = clean_signal + noise_signal

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(clean_signal)
        plt.title('clean_signal')
        plt.subplot(3, 1, 2)
        plt.plot(noise_signal)
        plt.title('noise_signal')
        plt.subplot(3, 1, 3)
        plt.plot(noisy_signal)
        plt.title('noisy_signal')
        plt.tight_layout()

        plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Noise_module/clean_noisy.png')

        plt.show()

        return noisy_signal, noise_list


    def generate_array_with_non_overlapping_patches(self, array_shape, patch_size, num_patches):
        '''
        Genera un array de ceros con parches de valor 1 en ubicaciones aleatorias sin solapamientos.

        :param array_shape: Tupla que representa la forma del array base (altura, ancho).
        :param patch_size: Tupla que representa el tamaño del parche (altura, ancho).
        :param num_patches: Número de parches a colocar en el array.
        :return: Array con los parches de valor 1.
        '''
        array = np.zeros(array_shape, dtype=int)
        array_height, array_width = array_shape
        patch_height, patch_width = patch_size

        patches_placed = 0
        while patches_placed < num_patches:
            y_start = np.random.randint(0, array_height - patch_height + 1)
            x_start = np.random.randint(0, array_width - patch_width + 1)
            
            if np.all(array[y_start:y_start + patch_height, x_start:x_start + patch_width] == 0):
                array[y_start:y_start + patch_height, x_start:x_start + patch_width] = 1
                patches_placed += 1
        
        return array




    
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

    

    def load_physionet_signals(self, type_noise='em', noise_augmentation=False, noise_normalization=True):
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

        if noise_normalization:
            noise = normalize_in_batches(noise, batch_size = 100, high=1, low=-1, axis_n=0)


        return noise

def normalize_in_batches(array, batch_size=200, high=1, low=-1, axis_n=0):
    """
    Processes an array in batches, applying normalization if needed.
    
    Args:
        array (numpy.ndarray): Array to be processed.
        batch_size (int): Size of each batch.
        noise_normalization (bool): Flag to apply normalization.

    Returns:
        numpy.ndarray: Array with batches processed and normalized.
    """
    num_samples = array.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Number of batches

    processed_array = np.empty_like(array)

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, num_samples)

        batch = array[start_index:end_index]
        
       
        batch = normalize_array(batch, high=high, low=low, axis_n=axis_n)  # Assuming noise is along axis 0
        
        processed_array[start_index:end_index] = batch

    return processed_array

    



    
 


        






