import sys
sys.path.append("../Code")
import os
import json
sys.path.append("../Code")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from tools_.preprocess_data import Preprocess_Dataset
from scripts.visualization.utils.renderizer import EGMRenderer_BSP
from scripts.config import ParseHiperparams
from tools_.load_dataset import LoadDataset_BSPS
from scripts.visualization.utils.bsp_3d_plotter import BSP_3D_PLOTTER
from scripts.visualization.utils.egm_3d_plotter import EGM_3D_PLOTTER
from scripts.visualization.utils.corr_3d_plotter import CORRELATION_3D_PLOTTER
from scripts.visualization.utils.df_map_3d_plotter import DF_MAPS_3D_PLOTTER
from scripts.Tikhonov.compute_tik import TikhonovReconstruction
from scripts.evaluate_function import *
from tools_.tools import corr_pearson_cols
from tools_.tools_inference import *
from tools_ import freq_phase_analysis as freq_pha

def normalize_array(array, high=1, low=-1, axis_n=0):
    mins = np.min(array, axis=axis_n)
    maxs = np.max(array, axis=axis_n)
    rng = maxs - mins
    if axis_n == 1:
        array = array.T
    norm_array = high - (((high - low) * (maxs - array)) / rng)
    if axis_n == 1:
        norm_array = norm_array.T
    return norm_array

#INDEX 

#I) ESPACIAL
#1. Voltage maps
#2. Correlation maps
#3. RMSE maps
#4. DF Maps

#II) TEMPORAL --> Notebook
# BSPM, EGM, EGM rec mismo plot
# Espectros


torso_num=2
model_path=f"/home/pdi/miriamgf/tesis/Autoencoders/Labeled_torsos/Torso{torso_num}_mod.mat"
model_name = ["Simulation_01_200316_001_  3"]
time_duration=5

# Cargar datos
#model_path = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/OMAMI_weighted/reconstructions_by_model_OMAMI_weighted.mat"

geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
geom_path_edgar= "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_Edgar/Atria.mat"
model_path_database= "/home/pdi/miriamgf/tesis/Autoencoders/Data/modelLA_RSPV_CAF_150115/EGMs.mat"
output_directory = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/renderized_heart/OMAMI"
os.makedirs(output_directory, exist_ok=True)
data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"

experiment_dir="/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/OMAMI_repeated/"
model_path_DL=experiment_dir+"reconstructions_by_model_OMAMI_repeated.mat"
geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
geom_path_edgar= "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_Edgar/Atria.mat"
weights_path = experiment_dir + "model_weights.h5"
params_path=experiment_dir+'hyperparams.json'



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

params = ParseHiperparams().parse_default_hyperparams()
params['fs_sub']=100

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

#BSPM
print("Plotting BSP...")

BSP_3D_PLOTTER(torso_num,
            all_torsos_names,
            y_list,
            X_1channel,
            output_directory,
            model_path,
            time=time_duration)()

try:
    with open(params_path, "r") as file:
        params = json.load(file)  # Load the JSON data into a dictionar
    n_batch=params["batch_size"]
except:
    n_batch=200


#Tikhonov

bspm_signal_norm = normalize_array(bspm_signal, high=1, low=-1, axis_n=1) 
tik_rec=TikhonovReconstruction(bspm_signal_norm, transfer_matrix, order=0)()
tik_rec2=TikhonovReconstruction(bspm_signal, transfer_matrix, order=0)()

tik_rec_norm = normalize_array(tik_rec, high=1, low=-1, axis_n=1) 
tik_rec_norm2 = normalize_array(tik_rec2, high=1, low=-1, axis_n=1) 

egm_single_norm = normalize_array(egm_single, high=1, low=-1, axis_n=0) 



print("Plotting Tikhonov...")
EGM_3d_object=EGM_3D_PLOTTER(model_name,
            model_path_DL,
            geom_path_CF,
            output_directory,
            labels_mode=False,
            tikhonov=True,
            time=time_duration)

_, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()

EGM_3d_object.plot_3d_mesh_prediction(tik_rec_norm2.T, egm_single_norm, faces_heart, vertices_heart)

#DL Predictions

print("Computing inference")
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
# Inference
weights_path="/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/OMAMI_repeated/model_weights.h5"
model = load_model(weights_path)

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

print("Plotting DL reconstruction")
EGM_3d_object.plot_3d_mesh_prediction(prediction, y_label, faces_heart, vertices_heart)

#Correlation
print("Plotting Correlation Maps")
# Creación del objeto para correlación
CorrelationObject = CORRELATION_3D_PLOTTER(
    model_name,
    model_path,
    geom_path_CF,
    output_directory,
    labels_mode=False,
    tikhonov=False,
    time=time_duration
)

# Calcular correlación por nodo
corr = CorrelationObject.correlation_by_node(prediction, y_label)

# Usar el método del objeto para plotear

CorrelationObject.plot_3d_mesh_label(
    corr, faces_heart, vertices_heart, np.min(corr), np.max(corr)
)

# Cálculo de frecuencia y fase
df_reconstructed, sig_k_rec, phase_rec = freq_pha.kuklik_DF_phase(prediction.T, fs=params["fs_sub"])
df_label, sig_k_rec, phase_rec = freq_pha.kuklik_DF_phase(y_label, fs=params["fs_sub"])


# Creación del objeto para mapas DF
DFMapObject = DF_MAPS_3D_PLOTTER(
    model_name,
    model_path,
    geom_path_CF,
    output_directory,
    labels_mode=False,
    tikhonov=False,
    time=time_duration
)

# Usar el método del objeto para plotear
DFMapObject.plot_3d_mesh_label(
    df_reconstructed, df_label,  vertices_heart,faces_heart, np.min(df_reconstructed), np.max(df_reconstructed)
)
