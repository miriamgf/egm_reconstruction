from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time
from scipy.signal import welch, coherence

from scripts.evaluation.tools_evaluate import deflexion_detection, compare_r_peaks, normalize_array

class Metrics:  
    def __init__(self, algorithm_ID, model_name):
         self.algorithm_ID = algorithm_ID
         self.model_name = model_name
         self.output_directory="/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/scripts/output/metrics_figures/"

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

    def peak_detector_classif(self,prediction, y_label, fs,d=0.1, plot = False):
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

        distance_in_samples = int(d * fs)  

        peak_list=deflexion_detection(y_label, fs=fs, prominence_value=0.3)
        peak_list_pred=deflexion_detection(prediction, fs=fs, prominence_value=0.3)

        precision_list=[]
        recall_list=[]
        error_list=[]

        for lead in range(0,y_label.shape[1]):
            
            peaks_i=peak_list[lead]
            lead_i=y_label[:, lead]
            peaks_pred_i=peak_list_pred[lead]
            lead_pred_i=prediction[:, lead]
            metrics, matching_peaks, matched_peaks_r=compare_r_peaks(peaks_i, peaks_pred_i,lead_i,lead_pred_i, tolerance_samples=distance_in_samples)
            
            precision_list.append(metrics['Precision'])
            recall_list.append(metrics['Sensitivity'])
            error_list.append(metrics['Error'])

            if plot:
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
                path_to_save=self.output_directory + f"peak_detection_{lead}.png"
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
            path_to_save=self.output_directory + f"DTW_{self.algorithm_ID}_{self.model_name}.png"

            plt.savefig(path_to_save)
            print("DTW figure saved in: ", path_to_save)
            plt.close()
            plt.show()

        return distance_list


    def compute_spectral_coherence(self,prediction, y_label, fs, ROI_freq=[0,30], nperseg_val=256):
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
         
        # Calcular los periodogramas de Welch para ambas señales
        f1, Pxx1 = welch(prediction, fs, nperseg=nperseg_val, noverlap=nperseg_val)  # Señal 1
        f2, Pxx2 = welch(y_label, fs, nperseg=nperseg_val, noverlap=nperseg_val)  # Señal 2

        # Calcular la coherencia espectral entre las dos señales
        f_coh, Cxy = coherence(prediction, y_label, fs=fs, nperseg=512)#, noverlap=256)

        f_max_coh=f_coh[np.argmax(Cxy)] #frecuencia con mayor coherencia
        f_roi=f_coh[f_coh > ROI_freq[0] and f_coh < ROI_freq[1] ] #frecuencias en el ROI
        indices_filtered = np.where(f < 30)[0] #indices de las frecuencias ROI
        Cxy_roi_mean=np.mean(Cxy[indices_filtered]) #coherencia media en el ROI


         return Cxy_roi_mean

    def compute_metrics(self,prediction, y_label, fs):
        """
        Computa las métricas de correlación y RMSE para un conjunto de pacientes.

        Returns:
            Un diccionario con las métricas de correlación y RMSE.
        """
        # Calcular correlación y RMSE para cada paciente

            
        corr = self.correlation_by_node(prediction, y_label)
        rmse = self.rmse_by_node(prediction, y_label)
        recall, precision, error  = self.peak_detector_classif(prediction, y_label, fs, d=0.05, plot = False)
        dtw=self.dynamic_time_warping(prediction, y_label, plot=True)
        error_no_nan=[x for x in error if str(x) != 'nan']
        error_peak_det_norm=normalize_array(error_no_nan, high=1, low=0, axis_n=0)
        compute_spectral_coherence

        metrics={"name": self.model_name,"Correlation": np.mean(corr),
                "RMSE": np.mean(rmse),
                "Peak_detector_Recall": np.mean(recall),
                "Peak_detector_Precision": np.mean(precision),
                "Peak_detector_Error": np.mean(error_peak_det_norm), 
                "DTW": np.mean(dtw)}
    
        metrics_all_nodes={"name": self.model_name,
            "Correlation": corr,
            "RMSE": rmse,
            "Peak_detector_Recall": recall,
            "Peak_detector_Precision": precision,
            "Peak_detector_Error": error_peak_det_norm, 
            "DTW":dtw}
                
        return metrics, metrics_all_nodes
    
