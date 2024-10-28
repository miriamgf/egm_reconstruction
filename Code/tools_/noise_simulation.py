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
from tools_.tools import *
from scipy.signal import welch
from scripts.config import DataConfig


# %% Path Models
# %% Path Models
current = os.path.dirname(os.path.realpath(__file__))
torsos_dir = "../../../Labeled_torsos/"
directory = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"


class NoiseSimulation:

    def __init__(
            
        self,
        params,
        SNR_em_noise=20,
        SNR_white_noise=20,
        oclusion=None,
        fs=500,
        random_order_chuncks=True,
    ):

        self.SNR_em = True
        self.SNR_white_noise = True
        self.fs = fs
        self.SNR_em_noise = SNR_em_noise
        self.SNR_white_noise = SNR_white_noise
        self.random_order_chunks = True
        self.params=params

    def configure_noise_database(
        self, len_target_signal, all_model_names, em=True, ma=False, gn=True
    ):
        """
        This function organizes the loaded flat noise signal into
        #N_models equal groups to encapsulate the chunks by single AF models.
        This reserves a given samples of the database for each AF model and each of its leads

        """
        n_models = len(all_model_names)
        dic_noise = {}

        if em:
            em_signal_flat = self.load_physionet_signals(
                type_noise="em", noise_normalization=True
            )

            if len(em_signal_flat) % n_models != 0:  # Truncate if necessary
                truncated_samples = (len(em_signal_flat) // n_models) * n_models
                em_signal_flat = em_signal_flat[:truncated_samples]

            em_split = np.split(em_signal_flat, n_models)
            em_split_split = self.split_into_irregular_chunks(
                em_split, target_signal_length=len_target_signal, noise_augmentation=5
            )

            dic_noise["em"] = em_split_split
            # TODO: Data augmentation: Concatenar dentro de un mismo modelo de AF
        elif ma:
            ma_signal_flat = self.load_physionet_signals(
                type_noise="ma", noise_normalization=True
            )
            ma_split = np.split(ma_signal_flat, n_models)
            dic_noise["ma"] = ma_split

        return dic_noise

    def add_noise(
        self,
        clean_signal,
        AF_model_i,
        noise_dic,
        distribution_noise_mode=2,
        n_noise_chunks_per_signal=3,
        num_patches=None,
        flatten_electrodes=False,
        Tikhonov_data_loading=False,
    ):

        # EM
        if self.SNR_em_noise != None:

            # if clean_signal.ndim ==2:
            # clean_signal = get_tensor_model(clean_signal, tensor_type="1channel")
            # flatten_electrodes = True

            em_noise = noise_dic["em"][AF_model_i]
            num_noise_chunks = len(em_noise)

            if distribution_noise_mode == 2:

                # if the patch size is fixed to (2x2), then the maximum number of patches is the total number of chunks //4
                min_num_patches_2_2 = (
                    num_noise_chunks // 4
                )  # this is the number of patches if only 1 noise chunks is assigned per lead
                if num_patches == None:
                    num_patches = min_num_patches_2_2 // n_noise_chunks_per_signal

                if (
                    Tikhonov_data_loading
                ):  # 3900 Nodes --> data augmentation. Apply noise to random nodes
                    binary_map = self.create_1d_binary_map(
                        array_shape=(clean_signal[0], clean_signal[1]),
                        n_leads=num_patches * 4,
                    )

                else:
                    # distribute a clusters of electrodes of available noise chunks in patches of size (2,2)
                    binary_map = self.generate_array_with_non_overlapping_patches(
                        array_shape=(clean_signal.shape[1], clean_signal.shape[2]),
                        patch_size=(2, 2),
                        num_patches=num_patches,
                    )
                indices = np.argwhere(binary_map == 1)

            elif distribution_noise_mode == 1:
                indices = np.argwhere(binary_map >= 1)
                n_noise_chunks_per_signal = 1

            noisy_signal = clean_signal.copy()  # np.zeros(clean_signal.shape)

            # leads = clean_signal[:, indices[:, 0], indices[:, 1]]
            if self.params['fs_sub'] == 100:
                num_chunks_per_clean_signal = 3
            elif self.params['fs_sub'] == 200:
                num_chunks_per_clean_signal = 3

            if (
                Tikhonov_data_loading
            ):  # 3900 Nodes --> data augmentation. Apply noise to random nodes
                for row_i, column_i in indices:
                    lead_i = clean_signal[:, column_i]
                    lead_i_noisy, em_noise = self.insert_noise_in_signal(
                        lead_i,
                        em_noise,
                        self.SNR_em_noise,
                        num_chunks_per_clean_signal=n_noise_chunks_per_signal,
                        normalize_noise=False,
                    )
                    noisy_signal[:, column_i] = lead_i_noisy  # Assign new noisy value

            else:

                for column_i, row_i in indices:
                    lead_i = clean_signal[:, column_i, row_i]
                    lead_i_noisy, em_noise = self.insert_noise_in_signal(
                        lead_i,
                        em_noise,
                        self.SNR_em_noise,
                        num_chunks_per_clean_signal=n_noise_chunks_per_signal,
                        normalize_noise=False,
                    )
                    noisy_signal[:, column_i, row_i] = (
                        lead_i_noisy  # Assign new noisy value
                    )

            if flatten_electrodes:
                noisy_signal = noisy_signal.reshape(
                    clean_signal.shape[0], clean_signal.shape[1] * clean_signal.shape[2]
                ).T

        """
        if self.SNR_white_noise != None:
            noisy_signal_before = noisy_signal.copy()
            noisy_signal = self.add_white_noise(noisy_signal, SNR=self.SNR_white_noise, fs=self.fs)
            plt.figure(figsize =(20,5))
            plt.plot(noisy_signal[:, 0, 4], label='EM + WN')
            plt.plot(noisy_signal_before[:, 0, 4], label='only EM')
            plt.plot(clean_signal[:, 0, 4], label='clean')
            plt.legend()
            os.makedirs('output/figures/Noise_module/', exist_ok=True)
            plt.savefig('output/figures/Noise_module/phases_filtering1.png')
        """
        return noisy_signal, binary_map

    def insert_noise_in_signal(
        self,
        clean_signal,
        noise_list,
        SNR,
        num_chunks_per_clean_signal=3,
        normalize_noise=True,
    ):

        # Create a disperse signal of same length with N noise realizations
        noise_signal = np.zeros(len(clean_signal))
        onset_index = random.randint(0, round(len(clean_signal) * 0.2))
        onset_index_previous = onset_index

        for chunk in range(0, num_chunks_per_clean_signal):
            try:
                chunk_i = noise_list.pop(0)
            except:

                break
            chunk_i = chunk_i.flatten()

            # normalize
            # plt.figure()
            # plt.plot(chunk_i, label='before norm')
            if normalize_noise:
                # chunk_i = normalize_array(chunk_i, high=1, low=-1, axis_n=0)
                chunk_i = self.normalize_in_batches(
                    chunk_i, batch_size=10, high=5, low=-5, axis_n=0
                )

                # plt.plot(chunk_i, label = 'after norma')
                os.makedirs("output/figures/Noise_module/", exist_ok=True)

                # plt.savefig('output/figures/Noise_module/norm.png')
                # plt.show()

            # if onset and chunk addition surpasses the clean signal length, then it tries to add it just after the last chunk
            if onset_index + len(chunk_i) > len(
                clean_signal
            ) and onset_index_previous + len(chunk_i) <= len(clean_signal):
                remaining_samples = len(clean_signal) - onset_index
                chunk_i = chunk_i[0:remaining_samples]
            # if it is not possible because either way it surpasses the clean signal length, it does not add more chunks
            elif onset_index + len(chunk_i) > len(
                clean_signal
            ) and onset_index_previous + len(chunk_i) > len(clean_signal):
                break
            # Id the onset index is directly out of range, no more chunks are used
            elif onset_index > len(clean_signal):
                print(
                    "Warning: onset_index > len (clean_signal). Further chunks excluded."
                )
                break
            else:
                noise_signal[onset_index : onset_index + len(chunk_i)] = chunk_i
            onset_index_previous = onset_index

            # Update onset
            onset_index = (
                onset_index
                + len(chunk_i)
                + random.randint(0, round(len(clean_signal) * 0.3))
            )

        # scale

        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean(noise_signal**2)

        snr_linear = 10 ** (SNR / 10)
        desired_noise_power = signal_power / snr_linear
        scaling_factor = np.sqrt(desired_noise_power / noise_power)
        noise_signal_SNR = noise_signal * scaling_factor

        noisy_signal = clean_signal + 2 * noise_signal_SNR
        indices_noise_addition = [i for i, x in enumerate(noise_signal) if x != 0]

        # plt.figure()
        # plt.subplot(3, 1, 1)
        # plt.plot(noisy_signal, label ='noisy signal')
        # plt.plot(clean_signal, label = 'clean signal')
        # plt.xlabel('Samples')
        # plt.ylabel('Amplitude (norm)')

        # plt.title('noisy vs clean_signal')
        # plt.subplot(3, 1, 2)
        # plt.plot(noise_signal, label = 'noise (norm)')
        # plt.plot(noise_signal_SNR, label='SNR')
        # plt.xlabel('Samples')
        # plt.ylabel('Amplitude (norm)')
        # plt.legend(loc='upper right')
        # plt.title('noise_signal at ' + str(SNR) + ' dB')
        # plt.subplot(3, 1, 3)
        # plt.plot(clean_signal[indices_noise_addition[0]:indices_noise_addition[0]+500], label = 'clean signal')
        # plt.plot(noise_signal_SNR[indices_noise_addition[0]:indices_noise_addition[0]+500], label = 'noise')
        # plt.xlabel('Samples')
        # plt.ylabel('Amplitude (norm)')
        # plt.legend(loc='upper right')
        # plt.title('Zoom in first chunk inclusion')
        # plt.tight_layout()
        # os.makedirs('output/figures/Noise_module/', exist_ok=True)
        # plt.savefig('output/figures/Noise_module/clean_noisy.png')

        # plt.show()

        return noisy_signal, noise_list

    def generate_array_with_non_overlapping_patches(
        self, array_shape, patch_size, num_patches
    ):
        """
        Genera un array de ceros con parches de valor 1 en ubicaciones aleatorias sin solapamientos.

        :param array_shape: Tupla que representa la forma del array base (altura, ancho).
        :param patch_size: Tupla que representa el tamaño del parche (altura, ancho).
        :param num_patches: Número de parches a colocar en el array.
        :return: Array con los parches de valor 1.
        """
        array = np.zeros(array_shape, dtype=int)
        array_height, array_width = array_shape
        patch_height, patch_width = patch_size

        patches_placed = 0
        while patches_placed < num_patches:
            y_start = np.random.randint(0, array_height - patch_height + 1)
            x_start = np.random.randint(0, array_width - patch_width + 1)

            if np.all(
                array[y_start : y_start + patch_height, x_start : x_start + patch_width]
                == 0
            ):
                array[
                    y_start : y_start + patch_height, x_start : x_start + patch_width
                ] = 1
                patches_placed += 1

        return array

    def split_into_irregular_chunks(
        self,
        split_noise,
        target_signal_length=2500,
        max_chunks=20,
        noise_augmentation=3,
    ):
        """
        Splits a list of arrays into irregularly sized chunks based on the specified signal length (number of samples).

        Parameters:
        split_noise (list of np.ndarray): A list of NumPy arrays to be split into chunks. Each array will be processed individually.
        target_signal_length (int): The target length of the signal used to determine the minimum and maximum chunk sizes. Default is 2500.

        Returns:
        list of list of np.ndarray: A list where each element is a list of chunks (NumPy arrays) derived from the corresponding array in `split_noise`.

        Description:
        This function divides each array in `split_noise` into chunks of varying sizes, ensuring that each chunk's size falls between a minimum and maximum value.
        The minimum and maximum chunk sizes are dynamically determined based on the `target_signal_length`.

        - `min_size`: Minimum chunk size, calculated as 5% of `target_signal_length`.
        - `max_size`: Maximum chunk size, calculated as 10% of `target_signal_length`.

        Chunks are created such that their sizes are randomly determined within the specified range, but the function ensures that the total size of chunks does not exceed
        the length of the original array. If fewer chunks are left to create, the last chunk will absorb any remaining elements to fit the exact size of the array.


        """

        # Automatically configures sizes of chunks based on signal length
        min_size = round(
            target_signal_length * 0.02
        )  # Minimum chunk size as 5% of the target signal length
        max_size = round(
            target_signal_length * 0.08
        )  # Maximum chunk size as 10% of the target signal length

        new_db = []
        for arr in split_noise:
            total_size = len(arr)
            chunks = []
            start_idx = 0
            while start_idx < total_size:
                remaining_size = total_size - start_idx

                # Determine the size of the next chunk
                chunk_size = np.random.randint(min_size, max_size + 1)

                # Ensure we do not exceed the remaining size or number of chunks
                if chunk_size > remaining_size:
                    chunk_size = remaining_size

                # Append the chunk to the list
                chunks.append(arr[start_idx : start_idx + chunk_size])
                start_idx += chunk_size

            if noise_augmentation != 1:  #
                for repeat in range(0, noise_augmentation):
                    chunks = chunks + chunks

            if self.random_order_chunks:
                random.shuffle(chunks)

            new_db.append(chunks)

        return new_db

    def add_white_noise(self, X, SNR=20, fs=200):
        """ """

        X_noisy, _ = addwhitenoise(X, SNR=SNR, fs=fs)

        return X_noisy

    def load_physionet_signals(self, type_noise="em", noise_normalization=True):
        """
        This functions loads signals from Physionet database MIT-BIH Noise Stress Test Database
        It can load three types of noise:
        * baseline wander (in record 'bw')
        * muscle artifact (in record 'ma')
        * electrode motion artifact (in record 'em')

        Original fs = 360 Hz

        """
        record_name = type_noise  # Nombre del registro
        path_noise_models = "tools_/noise_models"
        record = wfdb.rdrecord(
            f"{path_noise_models}/{record_name}", sampfrom=0, channels=[0]
        )

        noise = record.p_signal
        noise_original = noise.copy()

        self.compute_periodogram_of_noise(noise_original)

        # resample to target signal fs
        #if self.fs_sub > 360:
            #raise ValueError(
                #"Error: cannot add EM noise from MIT DB Database. Fs cannot match."
            #)
        # resample to extend the original noise fs = 360 to 500 Hz
        #TODO: Revisar, ahora mismo se interpolar para aumentar la fs, pero ver si reubicar este módulo a preprocessing

        noise = signal.resample_poly(noise, 500, 360)
        noise = signal.resample_poly(noise, self.fs, 500)

        if noise_normalization:
            noise = self.normalize_in_batches(
                noise, batch_size=50, high=1, low=-1, axis_n=0
            )

        return noise

    def compute_periodogram_of_noise(self, noise_signal, fs=360, nperseg=50):

        signal = noise_signal.T
        signal = signal.flatten()
        f, Pxx = welch(signal, fs, nperseg=nperseg, noverlap=nperseg // 2)
        # Graficar el periodograma de Welch
        plt.figure(figsize=(10, 6))
        plt.semilogy(f, Pxx)
        plt.title("Periodograma de Welch")
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Densidad espectral de potencia [V**2/Hz]")
        plt.grid(True)
        os.makedirs("output/figures/Noise_module/", exist_ok=True)
        plt.savefig("output/figures/Noise_module/welch_noise.png")

    def normalize_in_batches(self, array, batch_size=200, high=1, low=-1, axis_n=0):
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
            batch = normalize_array(
                batch, high=high, low=low, axis_n=axis_n
            )  # Assuming noise is along axis 0

            processed_array[start_index:end_index] = batch

        return processed_array

    def create_1d_binary_map(self, array_shape, n_leads):

        # Crear un array de ceros
        binary_map = np.zeros((1, len(array_shape[1])))

        # Seleccionar N posiciones únicas aleatorias en el array
        pos = np.random.choice(len(array_shape[1]), n_leads, replace=False)

        # Asignar el valor 1 a esas posiciones
        binary_map[0, pos] = 1

        return binary_map


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
