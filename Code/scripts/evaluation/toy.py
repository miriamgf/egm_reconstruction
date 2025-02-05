import sys
sys.path.append("../Code")
import os
import json
sys.path.append("../Code")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import scipy
import pywt
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import welch, coherence
from scipy.ndimage import uniform_filter1d

from tools_.preprocess_data import Preprocess_Dataset
from tools_.load_dataset import LoadDataset_BSPS
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer
from tools_.tools_inference import postprocess_prediction
from scripts.evaluation.tools_evaluate import normalize_array, downsampling, bandpass_filter
from scripts.evaluate_function import *
from tools_.tools_inference import *
from scripts.evaluation.metrics import Metrics

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scripts.evaluation.metrics import deflexion_detection
from scripts.evaluation.tools_evaluate import *
import time
start = time.time()

#---------------------------------------------------------------------------------------------------------------------
# CONFIGURE
#---------------------------------------------------------------------------------------------------------------------

plot_BSP = True
plot_Tikhonov = True
plot_DL= True
plot_correlation_DL = True
plot_correlation_tik = True
plot_rmse_DL = True
plot_rmse_tik = True

plot_DF_maps_DL = False
plot_DF_maps_tik = False

torso_num=2
test_patients = ["Simulation_01_200316_001_  3","Simulation_01_200212_001_ 10",
            "LA_PLAW_140711_arm", "LA_RSPV_CAF_150115",
            "Simulation_01_200212_001_  5", "Simulation_01_200212_001_ 10",
            "Simulation_01_200316_001_  3", "Simulation_01_200316_001_  4",
            "Simulation_01_200316_001_  8", "Simulation_01_200428_001_004",
            "Simulation_01_200428_001_008", "Simulation_01_200428_001_010",
            "Simulation_01_210119_001_001", "Simulation_01_210208_001_002"
        ]

#model_name = ["Simulation_01_200212_001_  5"]
#model_name = ["Simulation_01_200316_001_  3"]
algorithm_ID= ["OMAMI_VAE_no_filt", "OMAMI_no_filt"]
algorithm_ID= "OMAMI_VAE_no_filt"


torso_path=f"/home/pdi/miriamgf/tesis/Autoencoders/Labeled_torsos/Torso{torso_num}_mod.mat"
geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
geom_path_edgar= "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_Edgar/Atria.mat"
output_directory = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/scripts/output/metrics_figures/"
os.makedirs(output_directory, exist_ok=True)
data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"

experiment_dir=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{algorithm_ID}/"
model_path_DL=experiment_dir+f"reconstructions_by_model_{algorithm_ID}.mat"
weights_path = experiment_dir + "model_weights.h5"
params_path=experiment_dir+'hyperparams.json'

#Load params dictionary
with open(experiment_dir+"hyperparams.json") as file:
    params = json.load(file)  # Load the JSON data into a dictionary

if params["algorithm"]=="OMAMI_VAE" or params["algorithm"]=="OMAMI_VAE_no_filt":
    fs=100
    n_batch=200
elif params["algorithm"]=="OMAMI" or params["algorithm"]=="OMAMI_no_filt":
    fs=200
    n_batch=400

print(params)
print('fs:', fs, ' batch size: ', n_batch)


#---------------------------------------------------------------------------------------------------------------------
# LOAD DATA
#---------------------------------------------------------------------------------------------------------------------



# Cargar datos del modelo y la geometría

all_torsos_names = []
for subdir, dirs, files in os.walk(torsos_dir):
    for file in files:
        if file.endswith(".mat"):
            all_torsos_names.append(file)

#Load geometry

SNR_em_noise = None
SNR_white_noise = 100
patches_oclussion = "PT"
experiment_number = 0
unfold_code = 1
global_better_settings=[]
for patient in test_patients:
    model_name=[patient]
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
        directory=data_dir,
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

    X_1channel_or=X_1channel.copy()
    y_list_or=y_list.copy()

    #Unpack
    torso_name = f"Torso{torso_num}_mod.mat"
    torso_index = all_torsos_names.index(torso_name)
    bspm_signal = y_list[torso_index]['y']
    transfer_matrix= transfer_matrices[torso_index][0]

    #Select only specified torso signals
    egm_single=np.split(egm_tensor, 10)[torso_index]
    X_1channel_single=np.split(X_1channel, 10)[torso_index]
    AF_models_single=np.split(np.array(AF_models), 10)[torso_index]
    Y_model_single=np.split(np.array(Y_model), 10)[torso_index]

    #normalize 
    bspm_signal_norm = normalize_array(bspm_signal.T, high=1, low=-1, axis_n=1) 
    egm_single_norm = normalize_array(egm_single, high=1, low=-1, axis_n=0) 

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

    rows = X_1channel.shape[0]
    divisible_rows = (rows // n_batch) * n_batch
    #batch gen
    bsps_batches = reshape(
                    X_1channel,
                    (
                        int(len(X_1channel) / n_batch),
                        n_batch,
                        X_1channel.shape[1],
                        X_1channel.shape[2],
                        1,
                    ),
                )
    egm_batches = reshape(
                    egm_tensor,
                    (
                        int(len(egm_tensor) / n_batch),
                        n_batch,
                        egm_tensor.shape[1],
                        1,
                    ),
                )



    print("Computing inference")

    # Inference
    try:
        model = load_model(weights_path)
    except:
        model = load_model(weights_path, custom_objects={'SamplingLayer': SamplingLayer})


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

    # Postprocess prediction
    prediction_post=postprocess_prediction(prediction_flat, fs=fs,cutoff_DC=1.5)
    y_label = tools.remove_mean(egm_flat, cutoff=1.5) 
    y_label=normalize_array(egm_flat, high=1, low=-1, axis_n=1)

    custom_path= "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/evaluation/toy/coherence"
    coh_list=Metrics().compute_spectral_coherence(prediction_post, y_label, fs, ROI_freq=[1.5,10], nperseg_val=200, plot=True)
    print(np.mean(coh_list))
    sys.exit()

























'''
    wavelet_list= ['db6','db8', 'bior3.5', 'coif3', 'bior3.3', 'bior4.4']
    threshold_list=[0.2,0.5, 0.8, 1, 1.2]
    level_list=[2, 3, 4]
    better_settings=[]
    cont=0
    cont_0=0
    for wav in wavelet_list:
        for level in level_list:
            for thr in threshold_list:  

                #prediction_filt=wavelet_filter(prediction,wav,level=level, threshold_mult=thr)
                prediction_filt=wavelet_filter(prediction_post,wav,level=level, threshold_mult=thr)

                recall, precision, error=Metrics().peak_detector_classif(prediction_post, y_label, fs=fs,  prominence_val=0.3)
                recall_filt, precision_filt, error=Metrics().peak_detector_classif(prediction_filt, y_label, fs=fs, prominence_val=0.3)

                print([wav, thr, level])
                print('Precision not filt:', np.mean(precision))
                print('Precision filt:', np.mean(precision_filt))

                if np.mean(precision_filt)<np.mean(precision):
                    cont_0+=1
                elif np.mean(precision_filt)>np.mean(precision):
                    cont+=1
                    better_settings.append([wav, thr, level]) #[np.mean(precision), np.mean(precision_filt)]])


                    # Assuming precision and precision_filt are lists of length 2048
                    precision = np.array(precision)  # Convert to numpy array if they are lists
                    precision_filt = np.array(precision_filt)

                    # Find the indices where precision_filt is greater than precision
                    best_node = np.where(precision_filt > precision)[0][0]

                    lead=best_node

                    peak_list=deflexion_detection(y_label, fs=fs, prominence_value=0.3)
                    peak_list_pred=deflexion_detection(prediction, fs=fs, prominence_value=0.3)
                    peak_list_pred_filt=deflexion_detection(prediction_filt, fs=fs, prominence_value=0.3)

                    peaks_i=peak_list[lead]
                    lead_i=y_label[:, lead]

                    peaks_pred_i=peak_list_pred[lead]
                    lead_pred_i=prediction[:, lead]

                    peaks_pred_i_filt=peak_list_pred_filt[lead]
                    lead_pred_i_filt=prediction_filt[:, lead]

                    #

                    HR=compute_HR_from_RR_dist(peaks_i, fs=fs)

                    n_beats_HR= (HR*(len(lead_i)/fs))/60

                    T_samples=int(len(lead_i)/n_beats_HR)
                    T_seconds= T_samples/fs
                    distance_in_samples = int(T_samples*0.3)  #30% del periodo
                    print('Tolerance:', distance_in_samples)

                    metrics, matching_peaks, matched_peaks_r=compare_r_peaks(peaks_i, peaks_pred_i,lead_i,lead_pred_i, tolerance_samples=distance_in_samples)
                    metrics_filt, matching_peaks_filt, matched_peaks_r_filt=compare_r_peaks(peaks_i, peaks_pred_i_filt,lead_i,lead_pred_i_filt, tolerance_samples=distance_in_samples)

                    plt.figure(figsize=(30, 20), tight_layout=True)
                    plt.subplot(2, 1, 1)
                    plt.plot(y_label[:, lead], color='royalblue')
                    plt.scatter(peaks_i, lead_i[peaks_i], color='purple', marker='o', label='Real peaks')
                    plt.scatter(matched_peaks_r, lead_i[matched_peaks_r], color='green', marker='x', s=200, label='Detected peaks original')
                    plt.scatter(matched_peaks_r_filt, lead_i[matched_peaks_r_filt], color='red', marker='x', s=200, label='Detected peaks wavelet')

                    plt.title('Real EGM')
                    plt.ylabel('Amplitude mV (normalized)')
                    plt.xlabel('Samples')
                    plt.legend()
                    plt.grid(True)

                    plt.subplot(2, 1, 2)
                    plt.plot(prediction[:, lead], color='green', alpha=0.5, label="pred")
                    plt.plot(prediction_post[:, lead], color='grey', label='pred postprocessed')

                    plt.plot(prediction_filt[:, lead], color='red', label='pred post filtered')
                    #plt.plot(prediction_post_filt[:, lead], color='grey', label='pred postprocessed filtered')

                    #plt.scatter(peaks_i, lead_pred_i[peaks_i], c='purple', marker='o', label='Real peaks')
                    plt.scatter(peaks_pred_i_filt, lead_pred_i_filt[peaks_pred_i_filt], color='red', marker='o', label='Prediction peaks filt')
                    plt.scatter(peaks_pred_i, lead_pred_i[peaks_pred_i], c='g', marker='o', label='Prediction peaks in original')
                    plt.scatter(matching_peaks_filt, lead_pred_i_filt[matching_peaks_filt], color='red', marker='x', s=200, label='Detected peaks filt')
                    plt.scatter(matching_peaks, lead_pred_i[matching_peaks], color='green', marker='x', s=100, label='Detected peaks in original')

                    plt.ylabel('Amplitude mV (normalized)')
                    plt.xlabel('Samples')
                    plt.legend()
                    plt.title(f" margin = {distance_in_samples} samples. Recall: {str(np.round(metrics['Sensitivity'], 2))}vs. filt:{str(np.round(metrics_filt['Sensitivity'], 2))}. Precision: {str(np.round(metrics['Precision'], 2))} vs filt:{str(np.round(metrics_filt['Precision'], 2))} Error: {str(np.round(metrics['Error'], 2))}") 
                    plt.grid(True)
                    plt.plot(y_label[:, lead], alpha=0.5,  color='royalblue', label="Ground truth (EGMs)")
                    plt.scatter(peaks_i, lead_i[peaks_i], color='purple', marker='o', label='Picos real', alpha=0.3)
                    plt.suptitle(f"Peak detection patient: {algorithm_ID}   {model_name}. HR={HR}. Wavelet: {wav} threshold: {thr} label:{level}")
                    path=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/evaluation/toy/peak_det_filt_{wav}_{thr}_{level}.png"
                    plt.savefig(path)
                    print("Peak detection figure saved in: ", path)
                    plt.close()
                
    print(cont,':', cont_0, 'better:worse')
    print(better_settings)
    global_better_settings.append([better_settings])
    
global_better_settings
patient_settings = [set(tuple(setting) for setting in patient[0]) for patient in global_better_settings]

# Encontrar la intersección de todas las listas de configuraciones
common_settings = set.intersection(*patient_settings)

# Convertir de nuevo a lista para visualización
common_settings = [list(setting) for setting in common_settings]
print('common setings:', common_settings)

'''
