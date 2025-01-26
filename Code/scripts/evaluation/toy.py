import sys
sys.path.append("../Code")
import os
import json
sys.path.append("../Code")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from tools_.preprocess_data import Preprocess_Dataset
from tools_.load_dataset import LoadDataset_BSPS
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer
from scripts.evaluation.tools_evaluate import normalize_array, downsampling
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
y_label_red=y_label[0:100, :]
y_pred_red=prediction[0:100, :]


import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Parámetros
n_samples = 400  # Número de muestras
fs = 1000  # Frecuencia de muestreo (Hz)
t = np.arange(n_samples) / fs  # Vector de tiempo

# Crear señales senoidales con desfase y amplitud sinusoidal
frequency = 10  # Frecuencia de las señales (Hz)
base_amplitude = 1  # Amplitud base de las señales

# Desfase sinusoidal, cambia rápidamente
A_phase = np.pi / 4  # Amplitud del desfase
omega_phase = 2 * np.pi * 0.5  # Frecuencia del desfase (mayor para cambio rápido)

# Amplitud sinusoidal, cambia con el tiempo
A_amp = 0.5  # Amplitud de la modulación de amplitud
omega_amp = 2 * np.pi * 3  # Frecuencia de la modulación de la amplitud

# Desfase y amplitud sinusoidal
phase_shift_sine = A_phase * np.sin(omega_phase * t)  # Desfase sinusoidal
amplitude_sine = base_amplitude + A_amp * np.sin(omega_amp * t)  # Amplitud sinusoidal
# Señal "real" (sin desfase y amplitud variable)
y_label_channel = base_amplitude * np.sin(2 * np.pi * frequency * t)

# Señal "predicha" (con desfase y amplitud sinusoidal)
prediction_channel = amplitude_sine * np.sin(2 * np.pi * frequency * t + phase_shift_sine)

# Convertir las señales a una lista de tuplas para DTW (requiere formato de tuplas)
y_label_tuples = [(y,) for y in y_label_channel]
prediction_tuples = [(p,) for p in prediction_channel]

# Calcular DTW para el canal
distance, path = fastdtw(y_label_tuples, prediction_tuples, dist=euclidean)

# Mostrar resultados
print(f"Distancia DTW: {distance}")
print(f"Camino óptimo (primeros 10 pares): {path[:10]}")

# Visualizar la alineación temporal
plt.figure(figsize=(12, 6))
plt.plot(t, y_label_channel, label="Señal Real", color="blue")
plt.plot(t, prediction_channel, label="Señal Predicha", color="orange", alpha=0.7)

# Añadir líneas que conecten las muestras alineadas
for (i, j) in path:
    plt.plot([t[i], t[j]], [y_label_channel[i], prediction_channel[j]], color="gray", alpha=0.5)

plt.title(f"Alineación Temporal con DTW. Distancia = {distance}")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(alpha=0.3)

# Guardar la imagen
output_directory = "./scripts/output/metrics_figures/"  # Asegúrate de tener un directorio de salida
path_to_save = output_directory + f"Sinusoidal_DTW_Distance_senoid_{distance:.2f}.png"
plt.savefig(path_to_save)
print('saved in:', path_to_save)
plt.close()
plt.show()
