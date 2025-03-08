# This script was developed Miriam Gutiérrez Fernández

import sys
sys.path.append("../Code")
import os
import json
import imageio
import scipy.signal as sigproc
from scipy.io import loadmat
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from inference_utils import ECG_filtering, correlation_by_node, reshape_tensor, bspm_to_images, batch_generation, normalize_array

import tensorflow as tf

import tools_.tools as tools
import tools_.oclusion
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer


from numpy import *
from scipy.io import savemat
import argparse
import datetime
import time

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import load_model
from tools_.preprocessing_compression import *

import platform
gpus = tf.config.experimental.list_physical_devices('GPU')
print('Available GPUS:', gpus)


class Inference_Heartlab:
    def __init__(self):

        #Define data source
        self.data_dir = "/home/pdi/miriamgf/tesis/Autoencoders/Real_data/HEartLab/data_E18_F02_R02_selection.mat"
        self.output_path_figs= "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/"

        #Define experiment parameters
        self.algorithm_ID= "OMAMI_no_filt_testing2_repeated"
        self.experiment_dir = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{self.algorithm_ID}/"
        self.weights_path = self.experiment_dir + "model_weights.h5"
        params_path= self.experiment_dir + 'hyperparams.json'
        with open(params_path) as file:
            params = json.load(file)
            self.params=params

        self.SEED = 42
        self.fs= 4000
    
    def load_data(self):
        mat = loadmat(self.data_dir)
        return mat["signal_tank_filt"], mat["signal_MEA1_RA_filt"], mat["signal_MEA3_LA_filt"], mat["tank_el_position"], mat["matrix_signal_MEA1_RA"], mat["matrix_signal_MEA3_LA"], mat["matrix_signal_tank"]
    
    def preprocess_data(self, data, params, type_data):
        '''
        This function preprocesses the input data following the next steps:
        
        1. Downsampling
        2. Truncate length for batch gen
        3. Filtering (detrend + passband butterworth 3.30 Hz)
        3. Normalize
        '''
 

        # Subsample
        downsampling_factor = int(self.fs / params["fs_sub"])
        data_downsampled = data[::downsampling_factor]

        #truncate length
        if data_downsampled.shape[0] % self.params["batch_size"] != 0:
            trunc_val = np.floor_divide(data_downsampled.shape[0], self.params["batch_size"])
            data_truncated = data_downsampled[0 : self.params["batch_size"] * trunc_val, ...]
        
        #filter

        filtered_signal=ECG_filtering(data_truncated, fs=params["fs_sub"], order=1, f_low=3, f_high=30)

        # Normalize -1, 1
        data_normalized = normalize_array(filtered_signal, high=1, low=-1, axis_n=0)

        if type_data=="bspms_32_32":
            plt.figure(figsize=(20, 10), tight_layout=True)
            plt.subplot(3, 1, 1)
            plt.plot(data[0:self.fs, 0, 0])
            plt.title('Original signal')
            plt.xlabel('Samples')
            plt.subplot(3, 1, 2)
            plt.plot(data_downsampled[0:params["fs_sub"], 0, 0])
            plt.xlabel('Samples')
            plt.title('Downsampled signal')
            plt.subplot(3, 1, 3)
            plt.plot(filtered_signal[0:params["fs_sub"], 0, 0])
            plt.title('Filtered signal')
            plt.xlabel('Samples')
            plt.suptitle(f"{type_data}")
            plt.savefig(f"{self.output_path_figs}{type_data}.png")
            print('saved image at ', f"{self.output_path_figs}{type_data}.png")
        
            plt.close()
        else:
            
            plt.figure(figsize=(20, 10), tight_layout=True)
            plt.subplot(3, 1, 1)
            plt.plot(data[0:self.fs, 0])
            plt.title('Original signal')
            plt.xlabel('Samples')
            plt.subplot(3, 1, 2)
            plt.plot(data_downsampled[0:params["fs_sub"], 0])
            plt.xlabel('Samples')
            plt.title('Downsampled signal')
            plt.subplot(3, 1, 3)
            plt.plot(filtered_signal[0:params["fs_sub"], 0])
            plt.title('Filtered signal')
            plt.xlabel('Samples')
            plt.suptitle(f"{type_data}")
            plt.savefig(f"{self.output_path_figs}{type_data}.png")
            print('saved image at ', f"{self.output_path_figs}{type_data}.png")
        
            plt.close()


        
        return data_normalized
    


    def run_inference(self, X_1channel_batches, egm_tensor_batches):
        '''
        This function loads pretrained AI model and performs inference 
        on loaded data in batches
        '''
        
        # Cargar modelo
        try:
            model = load_model(self.weights_path)
        except:
            model = load_model(self.weights_path, custom_objects={"SamplingLayer": SamplingLayer})

        #Predict
        pred = model.predict(X_1channel_batches, batch_size=1)
        #Matching shapes 

        egms=np.squeeze(egm_tensor_batches, axis=-1)
        
        
        pred_egm = sigproc.resample(pred[1], num=32, axis=-1) #TODO Explore more options

        egms_flat = reshape_tensor(egms, egms.ndim, 2)
        reconstruction_flat_test = reshape_tensor(pred_egm, pred_egm.ndim, 2)

        return reconstruction_flat_test, egms_flat

    
    def main(self):

        # Load data

        X_1channel,egm_tensor_RA,egm_tensor_LA,tank_el_position, matrix_RA, matrix_LA, matrix_BSPMS=self.load_data()

        print("Shape X_1channel", X_1channel.shape)
        print("Shape egm_tensor_RA", egm_tensor_RA.shape)
        print("Shape egm_tensor_LA", egm_tensor_LA.shape)

        #TODO Explore more options
        egm_tensor=np.concatenate((egm_tensor_LA,egm_tensor_RA ), axis=1)

        #Preprocess data
        X_1channel_pre=self.preprocess_data(matrix_BSPMS, self.params, "bspms")
        egm_pre=self.preprocess_data(egm_tensor, self.params, "egm")

        #X_1channel_reshaped=bspm_to_images(X_1channel_pre, tank_el_position)
        
        frames = []
        for instant in range(0, 4000):
            # Crear la figura
            fig, ax = plt.subplots()
            ax.imshow(X_1channel_pre[instant, :, :], cmap='gray')  # Puedes cambiar el colormap
            ax.set_title('Video BSPMS')
            ax.axis('off')  # Ocultar ejes
            # Guardar temporalmente la imagen
            temp_path = os.path.join(self.output_path_figs, f"bspm_frame_{instant}.png")
            plt.savefig(temp_path)
            plt.close()
    
            # Cargar imagen y añadirla a la lista de frames
            frames.append(imageio.imread(temp_path))
        gif_path=os.path.join(self.output_path_figs, "bspm_video_32_32.gif")
        # Guardar el GIF
        imageio.mimsave(gif_path, frames, duration=5)  # Ajusta 'duration' para cambiar la velocidad

        print(f"GIF guardado en: {gif_path}")

        frames = []
        
        for instant in range(0, 400):
            # Crear la figura
            fig, ax = plt.subplots()
            ax.imshow(X_1channel_reshaped[instant, :, :], cmap='gray')  # Puedes cambiar el colormap
            ax.set_title('Video BSPMS')
            ax.axis('off')  # Ocultar ejes
            # Guardar temporalmente la imagen
            temp_path = os.path.join(self.output_path_figs, f"bspm_frame_{instant}.png")
            plt.savefig(temp_path)
            plt.close()
    
            # Cargar imagen y añadirla a la lista de frames
            frames.append(imageio.imread(temp_path))
        gif_path=os.path.join(self.output_path_figs, "bspm_video.gif")
        # Guardar el GIF
        imageio.mimsave(gif_path, frames, duration=5)  # Ajusta 'duration' para cambiar la velocidad

        print(f"GIF guardado en: {gif_path}")

        X_1channel_in_batches=batch_generation(X_1channel_reshaped, batch_size=self.params["batch_size"], type_data="bspm")
        egm_in_batches=batch_generation(egm_pre, batch_size=self.params["batch_size"], type_data="egms")

        reconstruction_flat_test,egms_flat= self.run_inference(X_1channel_in_batches, egm_in_batches )

        estimate_egms_n=normalize_array(reconstruction_flat_test, high=1, low=-1, axis_n=0)
        corr = correlation_by_node(estimate_egms_n, egms_flat)
        print("corr_mean", np.mean(corr))

        # Plotting reconstructions

        for node in range(egms_flat.shape[1]):
            length_plot=400
            egms_to_plot=egms_flat[0:length_plot, node]
            reconstruction_to_plot=estimate_egms_n[0:length_plot, node]
            bsps_to_plot=X_1channel_reshaped[0:length_plot, node, node]

            egms_to_plot_norm=normalize_array(egms_to_plot, high=1, low=-1, axis_n=0)
            reconstruction_to_plot_norm=normalize_array(reconstruction_to_plot, high=1, low=-1, axis_n=0)
            bsps_to_plot_norm=normalize_array(bsps_to_plot, high=1, low=-1, axis_n=0)


            plt.figure(figsize=(20, 10))
            plt.subplot(2, 1, 1)
            plt.plot(egms_to_plot_norm, label='GT')
            plt.plot(reconstruction_to_plot_norm, label='Reconstruction')
            plt.xlabel('Samples')
            plt.title('Reconstruction vs GT')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(bsps_to_plot_norm, label='BSPM')
            plt.xlabel('Samples')
            plt.title('Tank electrodes')
            plt.savefig(f"{self.output_path_figs}reconstruction_vs_gt_{node}.png")
            print('saved image at ', f"{self.output_path_figs}reconstruction_vs_gt_{node}.png")
            plt.close()
        

if __name__ == "__main__":
    Inference_Heartlab_obj=Inference_Heartlab()
    Inference_Heartlab_obj.main()  
