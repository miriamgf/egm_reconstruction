# This script was developed Miriam Gutiérrez Fernández

import sys
sys.path.append("../Code")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import argparse
import datetime
import os
import pickle
import random
import time
import json
import cv2
import imageio


import matplotlib.pyplot as plt
#import mlflow
import scipy
from scipy.signal import resample
import tensorflow as tf
import tools_
import tools_.tools as tools
import tools_.oclusion
from evaluate_function import evaluate_function_multioutput, evaluate_function_multioutput
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer
from tools_.load_dataset import LoadDataset_BSPS


from numpy import *
from scipy.io import savemat
import h5py

import tools_
import tools_.tools
 

from tools_.df_mapping import *
from tools_.tools import *

import argparse
import datetime
import time

import tensorflow as tf
from config import ParseHiperparams, GetMetadata
from keras import backend as K
from tensorflow.keras.models import load_model
from config import str_to_bool

from tools_.load_dataset import LoadDataset
from tools_.preprocess_data import Preprocess_Dataset
from tools_.preprocessing_compression import *

import platform
print("NumPy version:", np.__version__)
print("SciPy version:", scipy.__version__)
print("Python version:", platform.python_version())
np.show_config()


class Inference_Heartlab:
    def __init__(self):

        gpus = tf.config.experimental.list_physical_devices('GPU')
        print('Available GPUS:', gpus)
        #Load hyperparams
        self.algorithm_ID= "OMAMI_no_filt_testing2_repeated"
        self.experiment_dir = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{self.algorithm_ID}/"
        self.weights_path = self.experiment_dir + "model_weights.h5"
        params_path= self.experiment_dir + 'hyperparams.json'
        with open(params_path) as file:
            params = json.load(file)
            self.params=params
            # Clear GPU
        K.clear_session()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        self.data_dir = "/home/pdi/miriamgf/tesis/Autoencoders/Real_data/HEartLab/data_E18_F02_R02_selection.mat"
        self.SEED = 42
    
    def remove_mean(self,signal):
        """
        Remove mean from signal

        Parameters:
            signal (array): signal to process

        Returns:
            signotmean: signal with its mean removed
        """
        signal=signal.T

        signotmean = np.zeros(signal.shape)

        for index in range(0, signal.shape[0]):
            signotmean[index, :] = sigproc.detrend(signal[index, :], type="constant")
        return signotmean.T
    
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

        sig_temp = self.remove_mean(signal)
        # sig_temp = signal

        # Bandpass filtering
        b, a = sigproc.butter(
            order,
            [f_low / round((self.params["fs_sub"] / 2)), f_high / round((self.params["fs_sub"] / 2))],
            btype="bandpass",
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
    
    def preprocess_data(self, data, params, type_data):
        '''
        This function preprocesses the data
        1. Interpolate
        2. Subsample
        3. Normalize
        4. Batch generation
        '''
 

        # Subsample
        downsampling_factor = int(4000 / params["fs_sub"])
        data_downsampled = data[::downsampling_factor]

        #truncate length
        if data_downsampled.shape[0] % self.params["batch_size"] != 0:
            trunc_val = np.floor_divide(data_downsampled.shape[0], self.params["batch_size"])
            data_truncated = data_downsampled[0 : self.params["batch_size"] * trunc_val, ...]
        
        #filter

        filtered_signal=self.ECG_filtering(data_truncated, fs=params["fs_sub"], order=2, f_low=5, f_high=30)

        # Normalize -1, 1
        data_normalized = tools.normalize_array(filtered_signal, high=1, low=-1, axis_n=0)

        plt.figure(figsize=(20, 10))
        plt.subplot(3, 1, 1)
        plt.plot(data[0:1000, 0])
        plt.title('Original signal')
        plt.subplot(3, 1, 2)
        plt.plot(data_downsampled[0:100, 0])
        plt.title('Original signal (only downsampling)')
        plt.subplot(3, 1, 3)
        plt.plot(filtered_signal[0:100, 0])
        plt.title('Filtered signal')
        plt.suptitle(f"{type_data}")
        plt.savefig(f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/{type_data}.png")
        print('saved image at ', f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/{type_data}.png")
    
        plt.close()

        
        return data_normalized
    
    
    
    def assign_nodes_bsps(self, bspms, tank_el_position):

        matrix_2D=[[ 145, 146, 155, 156, 165, 166, 129, 130, 139, 140, 181, 182], 
                   [ 147, 147, 157, 157, 167, 167, 131, 131, 141, 141, 183, 183],
                   [ 148, 149, 158, 159, 168, 169, 132, 133, 142, 143, 184, 185],
                   [ 150, 151, 160, 161, 170, 171, 134, 135, 144, 177, 186, 187],
                   [ 152, 152, 162, 162, 172, 172, 136, 136, 178, 178, 188, 188],
                   [ 153, 154, 163, 164, 173, 174, 137, 138, 179, 180, 189, 190]]
        
        bspms_reshaped=bspms.reshape(bspms.shape[0], 12, 5)
        bspms_interpol_array=[]
        for element in range(bspms_reshaped.shape[0]):

            element_interp = cv2.resize(element, (32, 12), interpolation=cv2.INTER_CUBIC)
            bspms_interpol_array.append(element_interp)
        bspms_reshaped=np.array(bspms_interpol_array)
            

        '''
        bspms_reshaped
        for instant_i in range(bspms.shape[0]):
            bspms_instant=bspms[instant_i, :]
            id_to_signal = dict(zip(tank_el_position.flatten(), bspms_instant))
            mapped_signal_matrix = np.vectorize(lambda x: id_to_signal.get(x, np.nan))(matrix_2D)
            bspms_reshaped.append(mapped_signal_matrix)
        bspms_reshaped=np.array(bspms_reshaped)
        '''
        return bspms_reshaped

    def main(self):

        # Load data
        #definir semilla
        self.params["seed"] = self.SEED  # Agregar la semilla a los parámetros  

        X_1channel=loadmat(self.data_dir)["signal_tank"]
        egm_tensor_RA=loadmat(self.data_dir)["signal_MEA1_RA"]
        egm_tensor_LA=loadmat(self.data_dir)["signal_MEA3_LA"]
        tank_el_position=loadmat(self.data_dir)["tank_el_position"]

        print("Tamaño de X_1channel", X_1channel.shape)
        print("Tamaño de egm_tensor_RA", egm_tensor_RA.shape)
        print("Tamaño de egm_tensor_LA", egm_tensor_LA.shape)

        egm_tensor=np.concatenate((egm_tensor_LA,egm_tensor_RA ), axis=1)

        #X_1channel_reshaped=self.assign_nodes_bsps(X_1channel, tank_el_position)


        #egm_tensor= stack((egm_tensor_RA, egm_tensor_LA), axis=-1)

        #Preprocess data
        X_1channel_pre=self.preprocess_data(X_1channel, self.params, "bspms")
        egm_pre=self.preprocess_data(egm_tensor, self.params, "egm")

        #egm_tensor_batches=self.preprocess_data(egm_tensor, self.params, "egms")

        X_1channel_reshaped=self.assign_nodes_bsps(X_1channel_pre, tank_el_position)
        bspms_reshaped=X_1channel_pre.reshape(X_1channel_pre.shape[0], 12, 5)

        frames = []
        for instant in range(0, 1000):
            # Crear la figura
            fig, ax = plt.subplots()
            ax.imshow(bspms_reshaped[instant, :, :], cmap='gray')  # Puedes cambiar el colormap
            ax.set_title('Video BSPMS - One frame')
            ax.axis('off')  # Ocultar ejes
            output_dir="/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/"
            # Guardar temporalmente la imagen
            temp_path = os.path.join(output_dir, f"bspm_frame_{instant}.png")
            plt.savefig(temp_path)
            plt.close()
    
            # Cargar imagen y añadirla a la lista de frames
            frames.append(imageio.imread(temp_path))
        gif_path=os.path.join(output_dir, "bspm_video.gif")
        # Guardar el GIF
        imageio.mimsave(gif_path, frames, duration=0.6)  # Ajusta 'duration' para cambiar la velocidad

        print(f"GIF guardado en: {gif_path}")

        rows = X_1channel_reshaped.shape[0]
        n_batch=self.params["batch_size"]
        divisible_rows = (rows // n_batch) * n_batch
        # Batch generation
        X_1channel_in_batches = reshape(
                X_1channel_reshaped,
                (
                    int(len(X_1channel_reshaped) / n_batch),
                    n_batch,
                    X_1channel_reshaped.shape[1],
                    X_1channel_reshaped.shape[2],
                    1,
                ),
            )
        
        #Remove Nans
        X_1channel_in_batches = np.nan_to_num(
            X_1channel_in_batches, nan=0.0
        )  # Nans generated during noise addition

        egm_in_batches = reshape(
                egm_pre,
                (
                    int(len(egm_pre) / n_batch),
                    n_batch,
                    egm_pre.shape[1],
                    
                    1,
                ),
            )

        # Inference
        try:
            model = load_model(self.weights_path)
        except:
            model = load_model(self.weights_path, custom_objects={"SamplingLayer": SamplingLayer})


        pred = model.predict(X_1channel_in_batches, batch_size=1)
        pred_autoencoder, pred_egm = pred[0], pred[1]
        egms=np.squeeze(egm_in_batches, axis=-1)
        
        data_resampled = resample(pred_egm, num=32, axis=-1)  # Redimensionar último eje a 36

        
        egms_flat = reshape_tensor(egms, n_dim_input=egms.ndim, n_dim_output=2)

        reconstruction_flat_test = reshape_tensor(
            data_resampled, n_dim_input=data_resampled.ndim, n_dim_output=2
        )

        estimate_egms_n=tools.normalize_array(reconstruction_flat_test, high=1, low=-1, axis_n=0)
        corr = self.correlation_by_node(estimate_egms_n, egms_flat)

        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.plot(egms_flat[0:1000, 0], label='GT')
        plt.plot(estimate_egms_n[0:1000, 0], label='Reconstruction')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(X_1channel_reshaped[0:1000, 2, 0], label='BSPM')
        plt.title('Reconstruction vs GT')
        plt.savefig(f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/reconstruction_vs_gt.png")
        print('saved image at ', f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/reconstruction_vs_gt.png")
        plt.close()
        print("corr_mean", np.mean(corr))

    def correlation_by_node(self, array1, array2):
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
                print('Computing correlation in :', n_cols, 'nodes')
                corr = np.zeros(n_cols)
                for i in range(n_cols):
                    corr[i], _ = spearmanr(array1[:, i], array2[:, i]) # or pearsonr

                return corr
    


if __name__ == "__main__":
    Inference_Heartlab_obj=Inference_Heartlab()
    Inference_Heartlab_obj.main()  
