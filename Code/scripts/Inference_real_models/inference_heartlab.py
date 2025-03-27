# This script was developed Miriam Gutiérrez Fernández

import sys
sys.path.append("../Code")
import os
import json
import imageio
import scipy.signal as sigproc
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from inference_utils import ECG_filtering, correlation_by_node, reshape_tensor, bspm_to_images, batch_generation, normalize_array



class Inference_Heartlab:
    """
    This class performs inference on BSPMs using a trained deep learning model.
    It includes methods for loading data, preprocessing, running inference, and visualizing results.
    """
    def __init__(self):
        """
        Initializes the class with predefined paths and experiment parameters.
        """

        #Define data source
        self.data_dir = "/home/pdi/miriamgf/tesis/Autoencoders/Real_data/HEartLab/data_E18_F02_R02_selection.mat"

        #Define experiment parameters
        self.input_data_path= "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/scripts/Inference_real_models/" 
        
        #Load weights
        self.weights_path = self.input_data_path + "model_weights.h5"

        
        #Load params config
        params_path= self.input_data_path + 'hyperparams.json'
        with open(params_path) as file:
            params = json.load(file)
            self.params=params
        
        #Default params
        self.SEED = 42
        self.fs= 4000

        self.onset_cut_matrix=int(3.5503*self.fs)
        self.end_cut_matrix=int(11.5503*self.fs)

        # Experiment config ################################

        self.matrix_egms=False
        self.bspm_matrix=False 
        self.from_12_5 = False
        self.model_id = "AE_Baseline"
        self.custom_layout=True

        if self.matrix_egms:
            self.experiment_name= f"{self.model_id}_inference_egms_matrix"
        else:
            self.experiment_name= f"{self.model_id}_inference_egms_matrix"
        if self.bspm_matrix:
            self.experiment_name= self.experiment_name + "_bspm_matrix"
        else:
            self.experiment_name= self.experiment_name + "_bspm_flat"

        if self.custom_layout:
            self.experiment_name= self.experiment_name + "_custom_layout_flipped_cols"

        ####################################################

        #Experiment_dir
        self.experiment_dir= f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/CINC25_Heartlab/{self.experiment_name}/"
        os.makedirs(self.experiment_dir, exist_ok=True)
        print(f"Saving experiment at {self.experiment_dir}")
        self.output_path_figs= self.experiment_dir

    def load_data(self):
        """
        Loads data from a .mat file and extracts relevant signals.
        
        Returns:
            tuple: Containing various signals and electrode positions.
        """
        mat = loadmat(self.data_dir)

        return mat["signal_tank_filt"], mat["signal_MEA1_RA_filt"], mat["signal_MEA3_LA_filt"], mat["tank_el_position"], mat["matrix_signal_MEA1_RA"], mat["matrix_signal_MEA3_LA"], mat["matrix_signal_tank"]
    
    def preprocess_data(self, data, params, type_data):
        """
        Preprocesses the input data by performing the following steps:
        
        1. Downsampling
        2. Truncating length for batch generation
        3. Filtering (detrend + passband Butterworth 3-30 Hz)
        4. Normalization (-1, 1)
        
        Args:
            data (ndarray): Input signal data.
            params (dict): Dictionary containing preprocessing parameters.
            type_data (str): Type of data being processed.
        
        Returns:
            ndarray: Normalized and preprocessed data.
        """

        if type_data== "bspm_matrix":
            data=data[self.onset_cut_matrix:self.end_cut_matrix+1, :]

        # Subsample
        downsampling_factor = int(self.fs / params["fs_sub"])
        data_downsampled = data[::downsampling_factor]

        #truncate length
        if data_downsampled.shape[0] % self.params["batch_size"] != 0:
            trunc_val = np.floor_divide(data_downsampled.shape[0], self.params["batch_size"])
            data_truncated = data_downsampled[0 : self.params["batch_size"] * trunc_val, ...]
        
        #filter
        filtered_signal=ECG_filtering(data_truncated, fs=params["fs_sub"], order=1, f_low=1, f_high=30)

        # Normalize -1, 1
        data_normalized = normalize_array(filtered_signal, high=1, low=-1, axis_n=0)


        # plot 1D
        plt.figure(figsize=(20, 10), tight_layout=True)
        plt.subplot(3, 1, 1)
        plt.plot(data[0:self.fs, 10])
        plt.title('Original signal')
        plt.xlabel('Samples')
        plt.subplot(3, 1, 2)
        plt.plot(data_downsampled[0:params["fs_sub"], 10])
        plt.xlabel('Samples')
        plt.title('Downsampled signal')
        plt.subplot(3, 1, 3)
        plt.plot(filtered_signal[0:params["fs_sub"], 10])
        plt.title('Filtered signal')
        plt.xlabel('Samples')
        plt.suptitle(f"{type_data}")
        plt.savefig(f"{self.output_path_figs}{type_data}.png")
        print('saved image at ', f"{self.output_path_figs}{type_data}.png")
        plt.close()
        
        return data_normalized
    


    def run_inference(self, X_1channel_batches, egm_tensor_batches):
        """
        Runs inference using a pretrained deep learning model.
        
        Args:
            X_1channel_batches (ndarray): Preprocessed BSPM data.
            egm_tensor_batches (ndarray): Preprocessed EGM tensor data.
        
        Returns:
            tuple: Reconstructed EGM signals and ground truth EGM signals.
        """
        
        # Cargar modelo
  
        model = load_model(self.weights_path)

        #Predict
        pred = model.predict(X_1channel_batches, batch_size=1)
        #Matching shapes 

        egms=np.squeeze(egm_tensor_batches, axis=-1)
        reconstruction_2048=pred[1]
        
        pred_egm = sigproc.resample(pred[1], num=32, axis=-1) #TODO Explore more options

        egms_flat = reshape_tensor(egms, egms.ndim, 2)
        reconstruction_2048_flat=reshape_tensor(reconstruction_2048, reconstruction_2048.ndim, 2)
        reconstruction_flat_test = reshape_tensor(pred_egm, pred_egm.ndim, 2)

        return reconstruction_flat_test, egms_flat, reconstruction_2048_flat
    
    def plot_interpol(self, X_1channel_reshaped, X_1channel_pre, bspms_before_interpol):
        '''
        This function plots 2D BSPMs before and after interpolation
        
        '''
        if self.from_12_5:
            X_1channel_pre_reshaped=X_1channel_pre.reshape(X_1channel_pre.shape[0], 12, 5)
        else:
            X_1channel_pre_reshaped=X_1channel_pre
        
        if self.custom_layout:
            X_1channel_pre_reshaped=bspms_before_interpol
        
        # Plot BSPMs before interpolation
        frames = []
        for instant in range(0, 400):
            fig, ax = plt.subplots()
            ax.imshow(X_1channel_pre_reshaped[instant, :, :], cmap='gray')  
            ax.set_title('BSPMS - Video Format (before interpolation)')
            ax.axis('off')  
            temp_path = os.path.join(self.output_path_figs, f"bspm_frame_{instant}.png")
            plt.savefig(temp_path)
            plt.close()
            #Save frame and then remove
            frames.append(imageio.imread(temp_path))
            os.remove(temp_path)

        # configure video file name
        if self.from_12_5:
            gif_path=os.path.join(self.output_path_figs, "bspm_video_12_32_before_inter.gif")
        elif self.custom_layout:
            gif_path=os.path.join(self.output_path_figs, "bspm_video_custom_layout_before_inter.gif")
        else:
            gif_path=os.path.join(self.output_path_figs, "bspm_video_25_25_before_inter.gif")
        
        # Save GIF
        imageio.mimsave(gif_path, frames, duration=5)  
        print(f"GIF saved at: {gif_path}")
        
        # Plot BSPMs after interpolation
        frames = []
        for instant in range(0, 400):
            fig, ax = plt.subplots()
            ax.imshow(X_1channel_reshaped[instant, :, :], cmap='gray')  
            ax.set_title('BSPMS - Video Format (after interpolation)')
            ax.axis('off')  
            temp_path = os.path.join(self.output_path_figs, f"bspm_frame_{instant}.png")
            plt.savefig(temp_path)
            #Save frame and then remove
            frames.append(imageio.imread(temp_path)) 
            os.remove(temp_path)
        
        #configure video file name
        if self.from_12_5:
            gif_path=os.path.join(self.output_path_figs, "bspm_video_12_5_interpol.gif")
        elif self.custom_layout:
            gif_path=os.path.join(self.output_path_figs, "bspm_video_custom_layout_after_inter.gif")
        else:
            gif_path=os.path.join(self.output_path_figs, "bspm_video_flat_interpol.gif")

        # Save GIF
        imageio.mimsave(gif_path, frames, duration=5)  

        print(f"GIF Saved at: {gif_path}")

    
    def main(self):

        # Load data

        X_1channel,egm_tensor_RA,egm_tensor_LA,tank_el_position, matrix_RA, matrix_LA, matrix_BSPMS=self.load_data()

        print("Shape X_1channel", X_1channel.shape)
        print("Shape egm_tensor_RA", egm_tensor_RA.shape)
        print("Shape egm_tensor_LA", egm_tensor_LA.shape)

        plt.figure(figsize=(20, 10))
        plt.plot(X_1channel[0:4000, 0:50], label= "Flat")
        plt.plot(matrix_BSPMS[0:4000, 0, 0], label="Matrix")
        plt.legend()
        plt.savefig(self.output_path_figs+"BSPM_samples.png")
        print(self.output_path_figs+"BSPM_samples.png")
        plt.close()


        #TODO Explore more options
        if not self.matrix_egms:
            egm_tensor=np.concatenate((egm_tensor_LA,egm_tensor_RA ), axis=1)
        else:
            pass


        #Preprocess data

 
        if self.bspm_matrix:
            bspms=matrix_BSPMS
        else:
            bspms=X_1channel

        if self.bspm_matrix:

            X_1channel_pre=self.preprocess_data(bspms, self.params, type_data="bspm_matrix")
        else:
            X_1channel_pre=self.preprocess_data(bspms, self.params, type_data="bspm_flat")


        egm_pre=self.preprocess_data(egm_tensor, self.params, type_data="egm")

        # Signal to image-video transformation
        X_1channel_reshaped, bspms_before_interpol =bspm_to_images(X_1channel_pre, tank_el_position, from_12_5= self.from_12_5, custom_layout=True)

        #Save 
        self.plot_interpol(X_1channel_reshaped, X_1channel_pre, bspms_before_interpol)

        X_1channel_in_batches=batch_generation(X_1channel_reshaped, batch_size=self.params["batch_size"], type_data="bspm")
        egm_in_batches=batch_generation(egm_pre, batch_size=self.params["batch_size"], type_data="egms")

        #Plot BSPM and EGM sample batch 0 (1D)
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.plot(X_1channel_in_batches[0, :, 1, 1, 0])
        plt.title('BSPM')
        plt.subplot(2, 1, 2)
        plt.plot(egm_in_batches[0, :, 1, 0])
        plt.title('EGM')
        plt.savefig(self.output_path_figs+"batch_1.png")
        print(self.output_path_figs+"batch_1.png")
        plt.close()

        #plot BSPM batch 0
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.plot(X_1channel_in_batches[0, 0:400, 0, 0], label= "Flat")
        plt.title('BSPM')
        plt.subplot(2, 1, 2)
        plt.plot(egm_in_batches[0, 0:400, 0, 0], label="Matrix")
        plt.title('EGM')
        plt.savefig(self.output_path_figs+"batch_0.png")
        print(self.output_path_figs+"batch_0.png")
        plt.close()


        reconstruction_flat_test,egms_flat, reconstruction_2048= self.run_inference(X_1channel_in_batches, egm_in_batches)

        estimate_egms_n=normalize_array(reconstruction_flat_test, high=1, low=-1, axis_n=0)

        #Save GT and reconstruction to json

        with open(f"{self.experiment_dir}arrays_gt_egm.json", "w") as f:
            json.dump({"reconstruction_flat_test": reconstruction_flat_test.tolist(),
                        "egms_flat": egms_flat.tolist(),
                        "reconstruction_2048": reconstruction_2048.tolist()}, f)

        corr = correlation_by_node(estimate_egms_n, egms_flat)
        print("corr_mean", np.mean(corr))

        # Plotting reconstructions

        for node in range(egms_flat.shape[1]):
            length_plot=400
            egms_to_plot=egms_flat[0:length_plot, node]
            reconstruction_to_plot=estimate_egms_n[0:length_plot, node]
            bsps_to_plot=X_1channel_reshaped[0:length_plot, 0, 0]

            egms_to_plot_norm=normalize_array(egms_to_plot, high=1, low=-1, axis_n=0)
            reconstruction_to_plot_norm=normalize_array(reconstruction_to_plot, high=1, low=-1, axis_n=0)
            bsps_to_plot_norm=normalize_array(bsps_to_plot, high=1, low=-1, axis_n=0)


            plt.figure(figsize=(20, 10))
            plt.subplot(2, 1, 1)
            plt.plot(egms_to_plot_norm, label='GT')
            plt.plot(reconstruction_to_plot_norm, label='Reconstruction')
            plt.xlabel('Samples (1 second)')
            plt.title('Reconstruction vs GT')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(bsps_to_plot_norm, label='BSPM')
            plt.xlabel('Samples (1 second)')
            plt.title('Tank electrodes')
            plt.savefig(f"{self.output_path_figs}reconstruction_vs_gt_{node}.png")
            print('Image saved at ', f"{self.output_path_figs}reconstruction_vs_gt_{node}.png")
            plt.close()
        

if __name__ == "__main__":
    Inference_Heartlab_obj=Inference_Heartlab()
    Inference_Heartlab_obj.main()  
