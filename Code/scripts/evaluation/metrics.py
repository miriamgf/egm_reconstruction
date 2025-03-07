import sys
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time
from scipy.signal import welch, coherence, csd, butter, filtfilt
import os
from scipy.ndimage import uniform_filter1d
import tools_.tools as tools
from tools_.tools_inference import postprocess_prediction
from sklearn.metrics.pairwise import cosine_similarity

import tools_.tools as tools
from scripts.evaluation.tools_evaluate import deflexion_detection,compare_r_peaks, bandpass_filter, compute_HR_from_RR_dist, custom_coherence

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

class Metrics:  
    def __init__(self, algorithm_ID=None, model_name=None, tik=False):
        self.algorithm_ID = algorithm_ID
        self.model_name = model_name
        self.tik = tik
        self.output_directory="/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/evaluation/metrics_figures"
        


    def correlation_by_node(self,array1, array2):
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



    def rmse_by_node(self,array1, array2):
            """
            Calcula el RMSE entre las columnas de dos arrays.

            Args:
                array1: un array de numpy de dimensión (n,m)
                array2: otro array de numpy de dimensión (n,m)

            Returns:
                Un array de numpy de dimensión (m,) que contiene la RMSE 
                de las columnas de array1 y array2.
            """

            # Verificar si ambos arrays tienen las mismas dimensiones
            assert (
                array1.shape == array2.shape
            ), "Los arrays deben tener las mismas dimensiones."

            n_cols = array1.shape[1]
            print('Computing rmse in', n_cols, 'columns')
            rmse = np.zeros(n_cols)
            for i in range(n_cols):
                rmse[i]=np.sqrt(np.mean((array1[:, i] - array2[:, i]) ** 2))

            return rmse

    def peak_detector_classif(self,prediction, y_label, fs,d=0.1, prominence_val=0.2, plot = False):
        '''

        This function computes the precision, recall and error of the peak detection algorithm
        The algorithm is based on the comparison of the real and predicted peaks

        Args:   
            prediction: array with the predicted EGM signal
            y_label: array with the real EGM signal
            fs: sampling frequency
            d: margin in seconds to consider a peak as a true positive
            plot: boolean to plot the results
        Returns:
            precision_list: list with the precision of the peak detection algorithm in each channel
            recall_list: list with the recall of the peak detection algorithm in each channel
            error_list: list with the error of the peak detection algorithm in each channel
        
        '''

            
        # Verificar si ambos arrays tienen las mismas dimensiones
        assert (
            y_label.shape == prediction.shape
        ), "Los arrays deben tener las mismas dimensiones."

        peak_list=deflexion_detection(y_label, fs=fs, prominence_value=prominence_val)
        peak_list_pred=deflexion_detection(prediction, fs=fs, prominence_value=0.2)

        precision_list=[]
        recall_list=[]
        error_list=[]

        for lead in range(0,y_label.shape[1]):
            
            peaks_i=peak_list[lead]
            lead_i=y_label[:, lead]
            peaks_pred_i=peak_list_pred[lead]
            lead_pred_i=prediction[:, lead]

            #compute HR from real EGM
            HR=compute_HR_from_RR_dist(peaks_i, fs=fs)
            n_beats_HR= (HR*(len(lead_i)/fs))/60
            try:
                T_samples=int(len(lead_i)/n_beats_HR)
            except:
                T_samples=int(0.15*fs)
            T_seconds= T_samples/fs

            #Compute relative refraction time using a 'k' customized % of T (physiological 30-40%)
            k=0.3
            distance_in_samples = int(T_samples*k)
            metrics, matching_peaks, matched_peaks_r=compare_r_peaks(peaks_i, peaks_pred_i,lead_i,lead_pred_i, tolerance_samples=distance_in_samples)
            
            precision_list.append(metrics['Precision'])
            recall_list.append(metrics['Sensitivity'])
            error_list.append(metrics['Error'])
        
        best_node_recall=np.argmax(recall_list)
        best_node_precision=np.argmax(precision_list)
        worst_node_recall=np.argmin(recall_list)
        worst_node_precision=np.argmin(precision_list)

        nodes_to_plot=[best_node_recall,best_node_precision, worst_node_recall, worst_node_precision ]

        if plot:
            
            for lead in nodes_to_plot:
                print(lead)
                if lead==best_node_recall:
                    target='best_node_recall'
                elif lead==best_node_precision:
                    target='best_node_precision'
                elif lead==worst_node_precision:
                    target='worst_node_precision'
                elif lead==worst_node_recall:
                    target='worst_node_recall'

                peaks_i=peak_list[lead]
                lead_i=y_label[:, lead]
                peaks_pred_i=peak_list_pred[lead]
                lead_pred_i=prediction[:, lead]
                metrics, matching_peaks, matched_peaks_r=compare_r_peaks(peaks_i, peaks_pred_i,lead_i,lead_pred_i, tolerance_samples=distance_in_samples)
                
                plt.figure(figsize=(20, 10), tight_layout=True)
                plt.subplot(2, 1, 1)
                plt.plot(y_label[:, lead], color='royalblue')
                plt.scatter(peaks_i, lead_i[peaks_i], color='purple', marker='o', label='Real peaks')
                plt.scatter(matched_peaks_r, lead_i[matched_peaks_r], color='red', marker='x', s=200, label='Detected peaks')
                plt.title('Real EGM')
                plt.ylabel('Amplitude mV (normalized)')
                plt.xlabel('Samples')
                plt.legend()
                plt.grid(True)
                plt.subplot(2, 1, 2)
                plt.plot(prediction[:, lead], color='orange')
                #plt.scatter(peaks_i, lead_pred_i[peaks_i], c='purple', marker='o', label='Real peaks')
                plt.scatter(peaks_pred_i, lead_pred_i[peaks_pred_i], c='g', marker='o', label='Prediction peaks')
                plt.scatter(matching_peaks, lead_pred_i[matching_peaks], color='red', marker='x', s=200, label='Detected peaks')
                plt.ylabel('Amplitude mV (normalized)')
                plt.xlabel('Samples')
                plt.legend()
                plt.title(f" margin = {distance_in_samples} samples. Recall: {str(np.round(metrics['Sensitivity'], 2))}. Precision: {str(np.round(metrics['Precision'], 2))} Error: {str(np.round(metrics['Error'], 2))}") 
                plt.grid(True)
                plt.plot(y_label[:, lead], alpha=0.5,  color='royalblue')
                plt.scatter(peaks_i, lead_i[peaks_i], color='purple', marker='o', label='Picos real', alpha=0.3)
                plt.suptitle(f"Peak detection {self.algorithm_ID}   {self.model_name}.")
                if self.tik:
                    path_to_save=f"{self.output_directory}/{self.algorithm_ID}/{self.model_name}/tik/peak_detection_{target}.png"
                else:                    
                    path_to_save=f"{self.output_directory}/{self.algorithm_ID}/{self.model_name}/peak_detection_{target}.png"
                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
                plt.savefig(path_to_save)
                print("Peak detection figure saved in: ", path_to_save)
                plt.close()

        return precision_list, recall_list, error_list
    
    def dynamic_time_warping(self, prediction, y_label, plot=False):
        '''
        
        This function computes the dynamic time warping between two multivariate signals and returns the distance between them
        DTW is computed for each channel and the mean distance is returned

        Args:
            prediction: array with the predicted EGM signal
            y_label: array with the real EGM signal
            plot: boolean to plot the results
        Returns:
            distance_list: list with the distance of the DTW algorithm in each channel
        
        
        '''

        print('Computing DTW...')
        time_start=time.time()

        distance_list = []
        for channel in range(0,y_label.shape[1]):
            print('Computing DTW in channel:', channel)
            y_label_channel = y_label[:, channel]
            prediction_channel = prediction[:, channel]
        
            # Convertir las señales a una lista de tuplas para DTW
            y_label_tuples = [(y,) for y in y_label_channel]
            prediction_tuples = [(p,) for p in prediction_channel]
            distance, path = fastdtw(y_label_tuples, prediction_tuples, dist=euclidean)
            distance_list.append(distance)

        time_end=time.time()
        print('DTW computation time:', time_end-time_start)

        if plot:
            
            min_dist_channel = np.argmin(distance_list)
            max_dist_channel = np.argmax(distance_list)
          
            #BEST CHANNEL

            y_label_channel = y_label[:, min_dist_channel]
            prediction_channel = prediction[:, min_dist_channel]
        
            # Convertir las señales a una lista de tuplas para DTW
            y_label_tuples = [(y,) for y in y_label_channel]
            prediction_tuples = [(p,) for p in prediction_channel]
            distance, path = fastdtw(y_label_tuples, prediction_tuples, dist=euclidean)
            distance_list.append(distance)
            
            # Visualizar la alineación temporal
            plt.figure(figsize=(12, 6), tight_layout=True)
            plt.subplot(2, 1, 1)
            plt.plot(y_label_tuples, label="Señal Real", color="blue")
            plt.plot(prediction_tuples, label="Señal Predicha", color="red", alpha=0.7)

            # Añadir líneas que conecten las muestras alineadas
            for (i, j) in path:
                plt.plot([i, j], [y_label_tuples[i], prediction_tuples[j]], color="gray", alpha=0.5)

            plt.title(f"DTW of Best Channel: {min_dist_channel}. Distance = {np.round(distance, 2)} (Mean= {np.round(np.mean(distance_list), 2)}). Patient: {self.model_name}. Algorithm: {self.algorithm_ID}")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude (normalized)")
            plt.legend()
            plt.grid(alpha=0.3)
            

            #WORST CHANNEL

            y_label_channel = y_label[:, max_dist_channel]
            prediction_channel = prediction[:, max_dist_channel]
        
            # Convertir las señales a una lista de tuplas para DTW
            y_label_tuples = [(y,) for y in y_label_channel]
            prediction_tuples = [(p,) for p in prediction_channel]
            distance, path = fastdtw(y_label_tuples, prediction_tuples, dist=euclidean)
            distance_list.append(distance)
            
            # Visualizar la alineación temporal
            plt.subplot(2, 1, 2)

            plt.plot(y_label_tuples, label="Señal Real", color="blue")
            plt.plot(prediction_tuples, label="Señal Predicha", color="red", alpha=0.7)

            # Añadir líneas que conecten las muestras alineadas
            for (i, j) in path:
                plt.plot([i, j], [y_label_tuples[i], prediction_tuples[j]], color="gray", alpha=0.5)

            plt.title(f"DTW of Worst Channel: {max_dist_channel}. Distance = {np.round(distance, 2)} (Mean= {np.round(np.mean(distance_list), 2)}). Patient: {self.model_name}. Algorithm: {self.algorithm_ID}")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude (normalized)")
            plt.legend()
            plt.grid(alpha=0.3)
            if self.tik:
                path_to_save=f"{self.output_directory}/{self.algorithm_ID}/{self.model_name}/tik/DTW.png"
            else:  
                path_to_save=f"{self.output_directory}/{self.algorithm_ID}/{self.model_name}/DTW.png"
            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
            plt.savefig(path_to_save)
            print("DTW figure saved in: ", path_to_save)
            plt.close()
            plt.show()

        return distance_list
    
    def cosine_similarity(self,prediction, y_label,fs, nperseg_val=256, plot=True):

        cosim_list = []
        for channel in range(0,y_label.shape[1]):
            _, Pxx = welch(prediction[:, channel], fs, nperseg=nperseg_val)
            _, Pyy = welch(y_label[:, channel], fs, nperseg=nperseg_val)
            Pxx = Pxx / np.linalg.norm(Pxx)
            Pyy = Pyy / np.linalg.norm(Pyy)
            similarity = cosine_similarity(Pxx.reshape(1, -1), Pyy.reshape(1, -1))[0, 0]
            cosim_list.append(similarity)
        best_channel=np.argmax(cosim_list)
        worst_channel=np.argmin(cosim_list)

        channels_to_plot=[best_channel, worst_channel]
        if plot:
            for channel in channels_to_plot:
                if channel == best_channel:
                    id='best'
                else:
                    id='worst'
            


        return cosim_list


    def compute_spectral_coherence(self,prediction, y_label, fs, ROI_freq=[0.5,30], nperseg_val=256, plot=False, custom_path=None):
        '''
        This function computes the spectral coherence between two signals and returns the mean coherence in the ROI_freq range
        

        Args:
            prediction: array with the predicted EGM signal
            y_label: array with the real EGM signal
            fs: sampling frequency
            ROI_freq: list with the lower and upper limit of the frequency range to compute the coherence
            nperseg_val: number of samples per segment to compute the coherence
        Returns:
            Cxy_roi_mean: mean coherence in the ROI_freq range
        
        '''

        print('Computing spectral coherence...')

        time_start=time.time()

        coh_list = []
        for channel in range(0,y_label.shape[1]):
        
            y_label_filtered = bandpass_filter(y_label[:, channel], fs, ROI_freq[0], ROI_freq[1])
            y_pred_filtered = bandpass_filter(prediction[:, channel], fs, ROI_freq[0], ROI_freq[1])

            # Alinear las señales
            lag = np.argmax(np.correlate(y_label_filtered, y_pred_filtered, mode="full")) - len(y_label_filtered)
            y_pred_filtered = np.roll(y_pred_filtered, lag)

            y_label_filtered=tools.normalize_array(y_label_filtered, high=1, low=-1, axis_n=0)
            y_pred_filtered=tools.normalize_array(y_pred_filtered, high=1, low=-1, axis_n=0)
        
            #f_coh, Cxy = custom_coherence(y_pred_filtered, y_label_filtered, fs=fs, nperseg=nperseg_val)
            f_coh, Cxy = coherence(y_pred_filtered, y_label_filtered, fs=fs, nperseg=nperseg_val)

            # Suavizar coherencia para reducir ruido
            Cxy_smoothed = uniform_filter1d(Cxy, size=5)

            # Calcular promedio de coherencia en el ROI
            ROI_indices = (f_coh >= ROI_freq[0]) & (f_coh <= ROI_freq[1])
            coherence_mean_ROI = np.mean(Cxy_smoothed[ROI_indices])
            coh_list.append(coherence_mean_ROI)
        
        best_channel=np.argmax(coh_list)
        worst_channel=np.argmin(coh_list)

        channels_to_plot=[best_channel, worst_channel]
        if plot:
            for channel in channels_to_plot:
                if channel == best_channel:
                    id='best'
                else:
                    id='worst'
                y_label_filtered = bandpass_filter(y_label[:, channel], fs, ROI_freq[0], ROI_freq[1])
                y_pred_filtered = bandpass_filter(prediction[:, channel], fs, ROI_freq[0], ROI_freq[1])

                y_label_filtered=tools.normalize_array(y_label_filtered, high=1, low=-1, axis_n=0)
                y_pred_filtered=tools.normalize_array(y_pred_filtered, high=1, low=-1, axis_n=0)

                # Calcular los periodogramas de Welch para ambas señales
                f1, Pxx = welch(y_pred_filtered, fs, nperseg=nperseg_val)
                f2, Pyy = welch(y_label_filtered, fs, nperseg=nperseg_val)
                f_csd, Pxy = csd(y_pred_filtered, y_label_filtered, fs=fs, nperseg=nperseg_val)

                f_coh, Cxy = coherence(y_pred_filtered, y_label_filtered, fs=fs, nperseg=nperseg_val)

                # Suavizar coherencia para reducir ruido
                Cxy_smoothed = uniform_filter1d(Cxy, size=5)

                plt.figure(figsize=(24, 12),tight_layout=True)
                plt.plot(f_coh, Cxy, label="Coherence (Original)")
                plt.plot(f_coh, Cxy_smoothed, label="Coherence (Smoothed)")
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("Coherence")
                plt.title("Coherence with Butterworth Filtering")
                plt.xlim([0, 40])
                plt.ylim([0, 1])
                plt.grid()
                if self.tik:
                    path_to_save=f"{self.output_directory}/{self.algorithm_ID}/{self.model_name}/tik/coh_Coherence_{id}.png"
                elif not self.tik:  
                    path_to_save=f"{self.output_directory}/{self.algorithm_ID}/{self.model_name}/coh_Coherence_{id}.png"
                if custom_path:
                    path_to_save=f"{custom_path}/coh_Coherence_{id}.png"
                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
                plt.savefig(path_to_save)
                print('Saved in ', path_to_save)
                plt.close()


                plt.figure(figsize=(24, 12),tight_layout=True)
                plt.plot(f1, Pxx, label="Prediction")
                plt.plot(f2, Pyy, label="Ground truth")
                plt.plot(f_csd, Pxy, label="Cross Spectrum Density")
                plt.xlim([0, 40])
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("Power spectral density")
                plt.title("Power spectral density")
                plt.legend()
                plt.grid()
                if self.tik:
                    path_to_save=f"{self.output_directory}/{self.algorithm_ID}/{self.model_name}/tik/coh_psd_{id}.png"
                elif not self.tik:  
                    path_to_save=f"{self.output_directory}/{self.algorithm_ID}/{self.model_name}/coh_psd_{id}.png"
                if custom_path:
                    path_to_save=f"{custom_path}/coh_psd_{id}.png"
                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
                plt.savefig(path_to_save)
                print('Saved in ', path_to_save)
                plt.close()

                plt.figure(figsize=(24, 12),tight_layout=True)
                plt.plot(y_label_filtered, label="Ground truth")
                plt.plot(y_pred_filtered, label="Prediction")
                plt.ylabel("Amplitude (normalized)")
                plt.xlabel("Samples")
                plt.title("Time domain")
                plt.legend()
                plt.grid()
                if self.tik:
                    path_to_save=f"{self.output_directory}/{self.algorithm_ID}/{self.model_name}/tik/coh_time_{id}.png"
                elif not self.tik:  
                    path_to_save=f"{self.output_directory}/{self.algorithm_ID}/{self.model_name}/coh_time_{id}.png"
                if custom_path:
                    path_to_save=f"{custom_path}/coh_tim_{id}.png"

                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
                plt.savefig(path_to_save)
                print('Saved in ', path_to_save)
                plt.close()
             
        return coh_list

    def compute_metrics(self,prediction, y_label, fs):
        """
        Computa las métricas de correlación y RMSE para un conjunto de pacientes.

        Returns:
            Un diccionario con las métricas de correlación y RMSE.
        """

        try:
            assert prediction[:, 0].max() == 1
        except AssertionError:
            print('Prediction not normalized!')
            sys.exit()

        try:
            assert y_label[:, 0].max() == 1
        except AssertionError:
            print('Label not normalized!')
            sys.exit()

        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.plot(prediction[0:500, 0], label='egm')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(y_label[0:500, 0], label='egm gt')
        plt.legend()
        plt.savefig("/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/evaluation_trash/"+'preprocessed_signals_feat_opt_evaluate.png')
        print('saved image at ', "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/evaluation_trash/"+'preprocessed_signals_feat_opt_evaluate.png')
        plt.close()

        corr = self.correlation_by_node(prediction, y_label)
        rmse = self.rmse_by_node(prediction, y_label)
        recall, precision, error  = self.peak_detector_classif(prediction, y_label, fs, d=0.05, plot = True)
        dtw=self.dynamic_time_warping(prediction, y_label, plot=True)
        error_no_nan=[x for x in error if str(x) != 'nan']
        error_peak_det_norm=tools.normalize_array(error_no_nan, high=1, low=0, axis_n=0)
        coh_list=self.compute_spectral_coherence(prediction, y_label, fs, ROI_freq=[1.5, 10], nperseg_val=fs*2, plot=True)

        metrics={"name": self.model_name,"Correlation": np.mean(corr),
                "RMSE": np.mean(rmse),
                "Peak_detector_Recall": np.mean(recall),
                "Peak_detector_Precision": np.mean(precision),
                "Peak_detector_Error": np.mean(error_peak_det_norm), 
                "DTW": np.mean(dtw), 
                "Coherence": np.mean(coh_list)}
    
        metrics_all_nodes={"name": self.model_name,
            "Correlation": list(corr),
            "RMSE": list(rmse),
            "Peak_detector_Recall": recall,
            "Peak_detector_Precision": precision,
            "Peak_detector_Error": list(error_peak_det_norm), 
            "DTW":dtw, 
            "Coherence": coh_list}
                
        return metrics, metrics_all_nodes
    
