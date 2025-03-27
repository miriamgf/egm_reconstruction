import os
import sys
sys.path.append("../Code")
import os
import json
from scipy.io import loadmat
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
from tensorflow.keras.models import load_model
import time
import argparse
from numpy import reshape

from tools_.preprocess_data import Preprocess_Dataset
from tools_.load_dataset import LoadDataset_BSPS
from scripts.Tikhonov.compute_tik import TikhonovReconstruction
from scripts.evaluation.metrics import Metrics
from models.attention import LuongAttention
from models.multioutput_VAE import  SamplingLayer
from scripts.evaluate_function import *
from tools_.tools_inference import *
import seaborn as sns
from scipy.signal import spectrogram
import tools_.tools as tools
from tools_.tools_inference import normalize_array


class Visualize2D:
    def __init__(self, test_patients, experiment_ID_list, list_metrics, torso_num=2, TIK_ON=True):
        self.test_patients = test_patients
        self.experiment_ID_list = experiment_ID_list
        self.torso_num = torso_num
        self.TIK_ON = TIK_ON
        self.metrics_obj = None
        self.fs = None
        self.n_batch = None
        self.output_directory = None
        self.data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
        self.torsos_dir="/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"
        self.list_metrics = list_metrics
        self.nperseg=600
        self.nperseg_value=self.nperseg
        self.load_tik_array=True

    def create_output_directory(self, algorithm_ID, model_name):
        """ Verifica si la carpeta de salida existe, si no, la crea. """
        self.output_directory = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/renderizations/{algorithm_ID}/{model_name}"
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def save_figure(self, fig, filename):
        """Guarda la figura en el directorio de salida."""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)
        fig.savefig(os.path.join(self.output_directory, filename), dpi=300)
        plt.close(fig)
        print(f"Saved at {os.path.join(self.output_directory, filename)} ")

    def process_patient(self, model_name, algorithm_ID):
        """Carga, preprocesa los datos y computa métricas para un paciente y un algoritmo."""
        print(f"Processing patient {model_name} with algorithm {algorithm_ID}")
        self.create_output_directory(algorithm_ID, model_name[0])
        self.metrics_obj = Metrics(algorithm_ID=algorithm_ID, model_name=model_name[0])

        # Rutas de experimentos y modelos
        experiment_dir = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{algorithm_ID}/"
        model_path = experiment_dir + f"reconstructions_by_model_{algorithm_ID}.mat"
        weights_path = experiment_dir + "model_weights.h5"
        params_path = experiment_dir + "hyperparams.json"

        # Cargar parámetros
        with open(params_path) as file:
            params = json.load(file)

        self.n_batch=params["batch_size"]
        self.fs_original=params["fs"]
        self.fs=params["fs_sub"]
        params["filter_EGM"] = True

        #Load geometry

        SNR_em_noise = None
        SNR_white_noise = 100
        patches_oclussion = "PT"
        experiment_number = 0
        unfold_code = 1

        # Cargar datos del modelo y la geometría
        all_torsos_names = []
        for subdir, dirs, files in os.walk(self.torsos_dir):
            for file in files:
                if file.endswith(".mat"):
                    all_torsos_names.append(file)

                # Load test model
        (
            X_1channel,
            Y,
            Y_model,
            egm_tensor,
            length_list,
            AF_models,
            all_model_names,
            transfer_matrices,
            y_list
        ) = LoadDataset_BSPS(
            params,
            directory=self.data_dir,
            data_type="1channelTensor",
            n_classes=params["n_classes"],
            downsampling=False,
            fs=params["fs"],
            norm=False,
            SR=True,
            n_batch=params["batch_size"],
            sinusoid=False,
            SNR_em_noise=SNR_em_noise,
            SNR_white_noise=SNR_white_noise,
            patches_oclussion=patches_oclussion,
            unfold_code=unfold_code,
            inference=False,
            select_model = model_name
        )()

        #Unpack
        torso_name = f"Torso{self.torso_num}_mod.mat"
        torso_index = all_torsos_names.index(torso_name)
        bspm_signal = y_list[torso_index]['y']
        transfer_matrix= transfer_matrices[torso_index]#[0]
        transfer_matrix_flat=transfer_matrices[torso_index][0]
        transfer_matrix_64=transfer_matrix[0][transfer_matrix[1].ravel(), :]
        bspm_signal_64=bspm_signal[transfer_matrix[1].ravel(), :]

        #Select only specified torso signals
        egm_single=np.split(egm_tensor, 10)[torso_index]
        X_1channel_single=np.split(X_1channel, 10)[torso_index]
        AF_models_single=np.split(np.array(AF_models), 10)[torso_index]
        Y_model_single=np.split(np.array(Y_model), 10)[torso_index]

        try:
            print(params["filter_EGM"])
        except:
            params["filter_EGM"] = True


        print(params)
        print('fs:', self.fs, ' batch size: ', self.n_batch)

        # PREPROCESS for DL Prediction

        dic_vars={}

        # Preprocess data
        (
        X_1channel, egm_tensor, AF_models, Y_model
        ) = Preprocess_Dataset(
            params,
            X_1channel_single,
            egm_single,
            list(AF_models_single),
            Y_model,
            dic_vars,
            Y,
            all_model_names,
            transfer_matrices,
            experiment_dir,
            norm_egm=True,
            inference=True
        )()

        #batch gen
        rows = X_1channel.shape[0]
        divisible_rows = (rows // self.n_batch) * self.n_batch

        X_1channel = X_1channel[:divisible_rows]
        egm_tensor=egm_tensor[:divisible_rows]

        bsps_batches = reshape(
                        X_1channel,
                        (
                            int(len(X_1channel) / self.n_batch),
                            self.n_batch,
                            X_1channel.shape[1],
                            X_1channel.shape[2],
                            1,
                        ),
                    )
        egm_batches = reshape(
                        egm_tensor,
                        (
                            int(len(egm_tensor) / self.n_batch),
                            self.n_batch,
                            egm_tensor.shape[1],
                            1,
                        ),
                    )
        
        #inference
        try:
            model = load_model(weights_path)
        except:
            try:
                model = load_model(weights_path, 
                                custom_objects={'LuongAttention': LuongAttention, 'SamplingLayer': SamplingLayer})
                print('loading sampling layer')
            except:
                model = load_model(weights_path, 
                custom_objects={'SamplingLayer': SamplingLayer})
                print('loading sampling layer')
        #model = load_model(weights_path)

        prediction = model.predict(
            bsps_batches, batch_size=1
        )  # x_test=[#batches, batch_size, 12, 32, 1]
        prediction = prediction[1]
        prediction_flat = prediction.reshape(
            (prediction.shape[0] * prediction.shape[1], prediction.shape[2])
        )
        egm_flat = egm_batches.reshape(
            (prediction.shape[0] * prediction.shape[1], prediction.shape[2])
        )
        bsp_flat = bsps_batches.reshape(
            (bsps_batches.shape[0] * bsps_batches.shape[1], bsps_batches.shape[2]* bsps_batches.shape[3])
        )
        #Preprocess before plotting

        prediction_norm = normalize_by_models(prediction_flat, Y_model)
        egm_norm = normalize_by_models(egm_flat, Y_model)
        X_1channel_norm = normalize_by_models(bsp_flat, Y_model)

        prediction_centered= prediction_norm-np.mean(prediction_norm)
        egm_centered=egm_flat-np.mean(egm_flat)
        X_1channel_centered=X_1channel_norm-np.mean(X_1channel_norm)
        bspm_signal_norm = normalize_array(bspm_signal_64.T, high=1, low=-1, axis_n=1) 

        TIK_ON=True
        if TIK_ON==True:

            #Preprocess bspm (normalization)
            bspm_signal_norm = normalize_array(bspm_signal_64, high=1, low=-1, axis_n=0) 

            #bspm_signal_norm = normalize_array(bspm_signal, high=1, low=-1, axis_n=0) 
            #Compute Tikhonov

            ObjTik=TikhonovReconstruction(bspm_signal_norm, transfer_matrix_64, order=0)
            if self.load_tik_array:
                try:
                    path_to_object=f"{experiment_dir}{model_name[0]}_tik_array.json"
                    with open(path_to_object, "r") as f:
                        tik_dict = json.load(f)
                        tik_rec=np.array(tik_dict["tik_rec"])
                except:
                    print("Tikhonov Object not available. Computing ZOT inference...")
                    ObjTik=TikhonovReconstruction(bspm_signal_norm, transfer_matrix_64, order=0)
                    tik_rec=ObjTik() 
            else:
                tik_rec=ObjTik() 
            #tik_rec=ObjTik() 
            tik_batches=ObjTik.tik_post_process_to_plot(tik_rec, self.fs, divisible_rows, self.n_batch)
            tik_flat = tik_batches.reshape(
                (tik_batches.shape[0] * tik_batches.shape[1], tik_batches.shape[2])
            )
            tik_rec_norm = normalize_array(tik_flat, high=1, low=-1, axis_n=0) 
            tik_rec_centered=tik_rec_norm-np.mean(tik_rec_norm)

        return tik_rec_centered, prediction_centered, egm_centered, X_1channel_centered
    

    def best_channels_by_metric(self, tik_rec_centered, prediction_centered, egm_centered):

        # Inicializar diccionario de mejores nodos
        best_nodes = {}

        # Computar métricas
        for metric in self.list_metrics:
            metric = metric[0]  # Extraer el string dentro de la lista

            if metric == "Correlation":
                corr_DL = self.metrics_obj.correlation_by_node(prediction_centered, egm_centered)
                corr_tik = self.metrics_obj.correlation_by_node(tik_rec_centered, egm_centered) if tik_rec_centered is not None else None

                best_nodes["best_DL_corr"] = np.argmax(corr_DL) if corr_DL is not None else None
                best_nodes["best_Tik_corr"] = np.argmax(corr_tik) if corr_tik is not None else None

                self.plot_violin_distribution_nodes(metric, corr_DL, corr_tik)

            elif metric == "RMSE":
                RMSE_DL = self.metrics_obj.rmse_by_node(prediction_centered, egm_centered)
                RMSE_tik = self.metrics_obj.rmse_by_node(tik_rec_centered, egm_centered) if tik_rec_centered is not None else None

                best_nodes["best_DL_rmse"] = np.argmin(RMSE_DL) if RMSE_DL is not None else None
                best_nodes["best_Tik_rmse"] = np.argmin(RMSE_tik) if RMSE_tik is not None else None

                self.plot_violin_distribution_nodes(metric, RMSE_DL, RMSE_tik)

            elif metric == "DTW":
                dtw_DL = self.metrics_obj.dynamic_time_warping(prediction_centered, egm_centered, plot=True)
                dtw_tik = self.metrics_obj.dynamic_time_warping(tik_rec_centered, egm_centered, plot=True) 

                best_nodes["best_DL_dtw"] = np.argmin(dtw_DL) if dtw_DL is not None else None
                best_nodes["best_Tik_dtw"] = np.argmin(dtw_tik) if dtw_tik is not None else None

                self.plot_violin_distribution_nodes(metric, dtw_DL, dtw_tik)

            elif metric == "Coherence":
                coh_DL = self.metrics_obj.compute_spectral_coherence(prediction_centered, egm_centered, self.fs, ROI_freq=[0.5, 30], nperseg_val=256, plot=True)
                coh_tik = self.metrics_obj.compute_spectral_coherence(tik_rec_centered, egm_centered, self.fs, ROI_freq=[0.5, 30], nperseg_val=256, plot=True)

                best_nodes["best_DL_coh"] = np.argmax(coh_DL) if coh_DL is not None else None
                best_nodes["best_Tik_coh"] = np.argmax(coh_tik) if coh_tik is not None else None

                self.plot_violin_distribution_nodes(metric, coh_DL, coh_tik)

            elif metric == "PeakDet": 

                recall, precision_DL, error = self.metrics_obj.peak_detector_classif(prediction_centered, egm_centered, self.fs, d=0.05, plot=True)
                recall, precision_tik, error = self.metrics_obj.peak_detector_classif(tik_rec_centered, egm_centered, self.fs, d=0.05, plot=True)

                best_nodes["best_DL_peakdet"] = np.argmax(precision_DL) if precision_DL is not None else None
                best_nodes["best_Tik_peakdet"] = np.argmax(precision_tik) if precision_tik is not None else None

                self.plot_violin_distribution_nodes(metric, precision_DL, precision_tik)

        return best_nodes
 

    def plot_time_series(self, best_nodes, prediction, egm, tik_rec, bspm):
        """Genera y guarda gráficas de series temporales para los mejores nodos."""
        for metric, node_num in best_nodes.items():
            if node_num is None:
                continue
            time_axis = np.arange(prediction.shape[0]) / self.fs
            fig, axs = plt.subplots(3, 1, figsize=(12, 6), tight_layout=True)
            axs[0].plot(time_axis, prediction[:, node_num], label='Predicted', color="blue")
            axs[0].plot(time_axis, egm[:, node_num], label='Real', color="red")
            axs[0].set_title(f"AI Prediction | Node {node_num} | {metric}")
            axs[0].legend()

            if tik_rec is not None:
                axs[1].plot(time_axis, tik_rec[:, node_num], label='Predicted', color="green")
                axs[1].plot(time_axis, egm[:, node_num], label='Real', color="red")
                axs[1].set_title(f"ZOT Prediction | Node {node_num} | {metric}")
                axs[1].legend()


            axs[2].plot(time_axis, bspm[:, :3], label='BSPM')
            axs[2].set_title(f"BSPM | Node {node_num}")

            self.save_figure(fig, f"time_series_best_{metric}2.png")
            #print(f"Figure saved at time_series_best_{metric}.png")
    #Welch 

    def compute_welch_periodogram(self,signal, fs, nperseg_value, title):

        fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)

        for height in range(0, signal.shape[1], 5):
            if height >= signal.shape[1]:  # Prevención de index error
                break
            f, Pxx_den = scipy.signal.welch(
                signal[:, height].flatten(),  # Asegurar que es unidimensional
                fs,
                nperseg=nperseg_value,
                #noverlap=nperseg_value // 2,
                scaling="density",
                detrend="linear"
            )
            ax.plot(f, Pxx_den, linewidth=0.5)

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [V²/Hz]")
        ax.set_ylim([0, 0.25])
        ax.set_xlim([0, 50])  # Corregido, no puede haber frecuencias negativas en PSD
        ax.set_title(title)

        # Guardar la figura correctamente
        self.save_figure(fig, f"{title}.png")
        plt.close(fig)  # Cerrar la figura para evitar consumo excesivo de memoria

        print(f"{title}.png")
    

    def compute_spectrogram(self, signal, fs, title, nperseg_value ):

        frequencies, times, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg_value)

        # Graficar el espectrograma
        plt.figure(figsize=(5, 3), tight_layout= True)
        plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
        plt.colorbar(label='PSD (dB/Hz)')
        plt.title(title)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')
        plt.ylim(0, 30)  # Mostrar solo hasta la frecuencia de Nyquist
        plt.grid()
        # Guardar la figura correctamente
       
    

    
    def plot_violin_distribution_nodes(self, metric, metric_dl, metric_tik):
  
        data = [metric_dl, metric_tik]  # Usa np.asarray() para evitar copias innecesarias
        labels = ['DL', 'ZOT']

        # Crear figura y ejes
        fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)

        # Crear gráfico de violín
        sns.violinplot(data=data, palette=["blue", "green"], ax=ax)

      
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title("Distribution of correlation between nodes", fontsize=12)
        ax.set_ylabel("Correlation", fontsize=10)
        ax.set_xlabel("Arrays", fontsize=10)
        
        # Ajustar límites
        if metric in ["Correlation", "RMSE", "Coherence"] :
            ax.set_ylim([-1.2, 1.2])

        self.save_figure(fig, f"Violin_{metric}aa.png")
        plt.close(fig)  # Cerrar la figura para evitar consumo excesivo de memoria


    def plotting(self, best_nodes, prediction_centered, egm_centered, tik_rec_centered, X_1channel_centered):
        '''
        This function plots

        - Time series
        - Welch periodogram
        - Violin plots of node distribution per metric
        
        '''
        # Guardar gráficas
        #Time series per metric (best channel)
        self.plot_time_series(best_nodes, prediction_centered, egm_centered, tik_rec_centered, X_1channel_centered)

        #Welch
        self.compute_welch_periodogram(X_1channel_centered, fs=self.fs, nperseg_value=self.nperseg_value, title=f" Welch Periodogram BSPM. nperseg = {self.nperseg_value}")
        self.compute_welch_periodogram(egm_centered, fs=self.fs, nperseg_value=self.nperseg_value, title=f"Welch Periodogram EGM Real. nperseg = {self.nperseg}")
        self.compute_welch_periodogram(prediction_centered, fs=self.fs, nperseg_value=self.nperseg_value, title=f"Welch Periodogram Deep Learning EGM Predicted. nperseg = {self.nperseg}")
        self.compute_welch_periodogram(tik_rec_centered, fs=self.fs, nperseg_value=self.nperseg_value, title=f"Welch Periodogram ZOT EGM Predicted. nperseg = {self.nperseg}")


    def run_all(self):
        """Ejecuta el procesamiento para todos los pacientes y algoritmos."""
        start=time.time()
        cont=0
        for model_name in self.test_patients:

            print(f"Loading patient {cont}/{len(self.test_patients)}" )
            cont+=1

            for algorithm_ID in self.experiment_ID_list:
                tik_rec_centered, prediction_centered, egm_centered, X_1channel_centered=self.process_patient(model_name, algorithm_ID[0])
                best_nodes=self.best_channels_by_metric(tik_rec_centered, prediction_centered, egm_centered)
                self.plotting(best_nodes, prediction_centered, egm_centered, tik_rec_centered, X_1channel_centered)
        
                end=time.time()

                print('Execution time 1 patient', (end-start)/60 , 'minutes')



if __name__ == "__main__":

    list_metrics= [["Correlation"], ["RMSE"], ["DTW"], ["Coherence"], ["PeakDet"]]
    #list_metrics= [["Correlation"]], ["RMSE"]#, ["DTW"], ["Coherence"], ["PeakDet"]]

    test_patients = [["Simulation_01_200212_001_  5"],  
                ["Simulation_01_210119_001_001"], 
                ["Simulation_01_200428_001_010"],["Simulation_01_200212_001_ 10"]]
    
    test_patients = [
            ["LA_PLAW_140711_arm"], ["LA_RSPV_CAF_150115"],
           ["Simulation_01_200212_001_  5"], ["Simulation_01_200212_001_ 10"],
            ["Simulation_01_200316_001_  3"], ["Simulation_01_200316_001_  4"],
            ["Simulation_01_200316_001_  8"], ["Simulation_01_200428_001_004"],
            ["Simulation_01_200428_001_008"], ["Simulation_01_200428_001_010"],
            ["Simulation_01_210119_001_001"], ["Simulation_01_210208_001_002"]
        ]

    try:
        print("Parsing bash params")
        parser = argparse.ArgumentParser(description="params")
        parser.add_argument("--algorithm_ID", type=str, help="experiment name", required=True)
        

        args = parser.parse_args()
        algorithm_ID = args.algorithm_ID
        experiment_ID_list=[[algorithm_ID]]

        print(algorithm_ID)
    
    except:
        experiment_ID_list=[['OMAMI_no_filt_testing2_repeated_no_filt_l2_tm']]

    vis = Visualize2D(test_patients, experiment_ID_list, list_metrics)
    vis.run_all()
