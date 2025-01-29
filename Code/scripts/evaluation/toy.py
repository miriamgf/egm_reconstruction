import sys
sys.path.append("../Code")
import os
import json
sys.path.append("../Code")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import scipy

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import welch, coherence
from scipy.ndimage import uniform_filter1d

from tools_.preprocess_data import Preprocess_Dataset
from tools_.load_dataset import LoadDataset_BSPS
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer
from scripts.evaluation.tools_evaluate import normalize_array, downsampling, bandpass_filter
from scripts.evaluate_function import *
from tools_.tools_inference import *
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean





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
model_name = ["Simulation_01_200212_001_  5"]
#model_name = ["LA_RSPV_CAF_150115"]

algorithm_ID= "OMAMI_VAE_Optuna_1"


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

if params["algorithm"]=="OMAMI_VAE":
    fs=100
    n_batch=200
elif params["algorithm"]=="OMAMI":
    fs=200
    n_batch=400

params["filter_EGM"]=True
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

prediction = normalize_by_models(prediction_flat, Y_model)
y_label=normalize_by_models(egm_flat, Y_model)

time_duration=y_label.shape[0] # num of samples to represent

channel=0
y_label_red=y_label[:, channel]
y_pred_red=prediction[:, channel]




# Filtrar señales
y_label_filtered = bandpass_filter(y_label_red, fs, 0.5, 30)
y_pred_filtered = bandpass_filter(y_pred_red, fs, 0.5, 30)

# Alinear las señales
lag = np.argmax(np.correlate(y_label_filtered, y_pred_filtered, mode="full")) - len(y_label_filtered)
y_pred_aligned = np.roll(y_pred_filtered, lag)

# Calcular coherencia con señales alineadas
nperseg_val = 100  # Ajustar tamaño de ventana
noverlap_val = nperseg_val // 2  # 50% de solapamiento

f_coh, Cxy = coherence(y_pred_aligned, y_label_filtered, fs=fs, nperseg=nperseg_val, noverlap=noverlap_val)


# Suavizar coherencia para reducir ruido
Cxy_smoothed = uniform_filter1d(Cxy, size=5)

# Calcular promedio de coherencia en el ROI
ROI_indices = (f_coh >= 0.5) & (f_coh <= 30)
coherence_mean_ROI = np.mean(Cxy[ROI_indices])
print(f"Mean Coherence in ROI (0.5–30 Hz): {coherence_mean_ROI:.3f}")

# Graficar coherencia
plt.figure()
plt.plot(f_coh, Cxy, label="Coherence (Original)")
plt.plot(f_coh, Cxy_smoothed, label="Coherence (Smoothed)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Coherence")
plt.title("Coherence with Butterworth Filtering")
plt.xlim([0, 40])
plt.ylim([0, 1])
plt.legend()
plt.grid()
plt.savefig(output_directory + f"coh_Coherence_smooth.png")
print('Saved in ', output_directory + "coh_Power_Spectral_Density.png")
plt.show()

# Graficar señales originales y filtradas
plt.figure()
plt.plot(y_label_red, label="Ground truth (Original)", alpha=0.5)
plt.plot(y_label_filtered, label="Ground truth (Filtered)")
plt.plot(y_pred_filtered, label="Prediction (Filtered)")
plt.plot(y_pred_aligned, label="Prediction (Aligned)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("Signals (Original, Filtered, and Aligned)")
plt.legend()
plt.grid()
plt.show()



nperseg_range=[25,50, 100, 125, 150, 200, 250, 300, 350]
for nperseg_i in nperseg_range:
    nperseg_val=nperseg_i

    f_coh, Cxy = coherence(y_pred_aligned, y_label_filtered, fs=fs, nperseg=nperseg_val, noverlap=nperseg_val//2)

    # Suavizar coherencia para reducir ruido
    Cxy_smoothed = uniform_filter1d(Cxy, size=5)

    f1, Pxx1 = scipy.signal.welch(
                y_pred_filtered,
                fs,
                nperseg=nperseg_val,
                noverlap=nperseg_val // 2,
                scaling="density",
                detrend="linear"
            )

    f2, Pxx2 = scipy.signal.welch(
                y_label_filtered,
                fs,
                nperseg=nperseg_val,
                noverlap=nperseg_val // 2,
                scaling="density",
                detrend="linear"
            )


    # Calcular la coherencia espectral entre las dos señales
    f_coh, Cxy = coherence(y_pred_filtered, y_label_filtered, fs=fs, nperseg=nperseg_val)#,noverlap=nperseg_val // 2  )#, noverlap=256)


    # Gráfica 2: Densidad espectral de potencia
    plt.figure(tight_layout=True)
    plt.plot(f1, Pxx1, label="Prediction")
    plt.plot(f2, Pxx2, label="Ground truth")
    plt.xlim([0, 40])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectral density")
    plt.title("Power spectral density")
    plt.legend()
    plt.grid()
    plt.savefig(output_directory + f"coh_Power_Spectral_Density{nperseg_i}.png")
    print('Saved in ', output_directory + "coh_Power_Spectral_Density.png")
    plt.close()

    # Calcular promedio de coherencia en el ROI
    ROI_indices = (f_coh >= 0.5) & (f_coh <= 30)
    coherence_mean_ROI = np.mean(Cxy[ROI_indices])
    print(f"Mean Coherence in ROI (0.5–30 Hz): {coherence_mean_ROI:.3f}")

    # Graficar coherencia
    plt.figure()
    plt.plot(f_coh, Cxy, label="Coherence (Original)")
    plt.plot(f_coh, Cxy_smoothed, label="Coherence (Smoothed)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Coherence")
    plt.title("Coherence with Butterworth Filtering")
    plt.xlim([0, 40])
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()
    plt.savefig(output_directory + f"coh_Coherence_smooth_{nperseg_i}.png")
    print('Saved in ', output_directory + "coh_Power_Spectral_Density.png")
    plt.show()

# Graficar señales originales y filtradas
plt.figure()
plt.plot(y_label_red, label="Ground truth (Original)", alpha=0.5)
plt.plot(y_label_filtered, label="Ground truth (Filtered)")
plt.plot(y_pred_filtered, label="Prediction (Filtered)")
plt.plot(y_pred_aligned, label="Prediction (Aligned)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("Signals (Original, Filtered, and Aligned)")
plt.legend()
plt.grid()
plt.show()

    