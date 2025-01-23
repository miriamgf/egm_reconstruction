import sys
sys.path.append("../Code")
import os
import json
import pandas as pd
import argparse


import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from tools_.preprocess_data import Preprocess_Dataset
from scripts.visualization.utils.renderizer import EGMRenderer_BSP
from scripts.config import ParseHiperparams
from tools_.load_dataset import LoadDataset_BSPS
from scripts.visualization.utils.bsp_3d_plotter import BSP_3D_PLOTTER
from scripts.visualization.utils.egm_3d_plotter import EGM_3D_PLOTTER
from scripts.visualization.utils.corr_3d_plotter import CORRELATION_3D_PLOTTER
from scripts.visualization.utils.rmse_3d_plotter import RMSE_3D_PLOTTER
from scripts.visualization.utils.df_map_3d_plotter import DF_MAPS_3D_PLOTTER
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer
from scripts.evaluation.tools_evaluate import normalize_array, downsampling
from scripts.evaluation.metrics import rmse_by_node, correlation_by_node

from scripts.Tikhonov.compute_tik import TikhonovReconstruction
from scripts.evaluate_function import *
from tools_.tools import corr_pearson_cols
from tools_.tools_inference import *
from tools_ import freq_phase_analysis as freq_pha

os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"


try:
    print("Parsing")
    parser = argparse.ArgumentParser(description="Noise params")
    parser.add_argument("--algorithm_ID", type=str, help="experiment name", required=True)
    args = parser.parse_args()
    algorithm_ID = args.algorithm_ID

except:
    algorithm_ID = "OMAMI_VAE"

#---------------------------------------------------------------------------------------------------------------------
# CONFIGURE
#---------------------------------------------------------------------------------------------------------------------


torso_num=2
time_duration=500 # num of samples to represent


torso_path=f"/home/pdi/miriamgf/tesis/Autoencoders/Labeled_torsos/Torso{torso_num}_mod.mat"
geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
geom_path_edgar= "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_Edgar/Atria.mat"
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

test_patients = [
                        "LA_PLAW_140711_arm",
                        "LA_RSPV_CAF_150115",
                        "Simulation_01_200212_001_  5",
                        "Simulation_01_200212_001_ 10",
                        "Simulation_01_200316_001_  3",
                        "Simulation_01_200316_001_  4",
                        "Simulation_01_200316_001_  8",
                        "Simulation_01_200428_001_004",
                        "Simulation_01_200428_001_008",
                        "Simulation_01_200428_001_010",
                        "Simulation_01_210119_001_001",
                        "Simulation_01_210208_001_002",
                    ]


corr_list = []
rmse_list = []

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

    prediction = normalize_by_models(prediction_flat, Y_model)
    y_label=normalize_by_models(egm_flat, Y_model)

    # Calcular correlación por nodo
    corr = correlation_by_node(prediction, y_label)
    RMSE = rmse_by_node(prediction, y_label)

    corr_list.append(np.mean(corr))
    rmse_list.append(np.mean(RMSE))


df_metrics = pd.DataFrame({
    "name": test_patients,
    "mean correlation": corr_list,  # Corrige el nombre de la columna eliminando el typo ("orrelation" a "correlation")
    "mean RMSE": rmse_list
})
df_metrics.head()
# Guardar el DataFrame como un archivo CSV
output_path = experiment_dir + "metrics_dl.csv" 
df_metrics.to_csv(output_path, index=False)
print('Metrics saved in',  output_path)

