import sys
sys.path.append("../Code")
import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd


from tools_.preprocess_data import Preprocess_Dataset
from tools_.load_dataset import LoadDataset_BSPS
from scripts.evaluation.tools_evaluate import normalize_array
from scripts.evaluation.metrics import rmse_by_node, correlation_by_node

from scripts.Tikhonov.compute_tik import TikhonovReconstruction
from scripts.evaluate_function import *
from tools_.tools_inference import *

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
data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"

experiment_dir=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{algorithm_ID}/"
output_directory = experiment_dir
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
# 
#---------------------------------------------------------------------------------------------------------------------

corr_list = []
rmse_list = []

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

for patient in test_patients:
    # Cargar datos del modelo y la geometría

    model_name=[patient]

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


    egm_flat = egm_batches.reshape(
        (egm_batches.shape[0] * egm_batches.shape[1], egm_batches.shape[2])
    )

    y_label=normalize_by_models(egm_flat, Y_model)

    #BSPM

    rows = X_1channel.shape[0]
    divisible_rows = (rows // n_batch) * n_batch

        
    ObjTik=TikhonovReconstruction(bspm_signal_norm.T, transfer_matrix, order=0)
    tik_rec=ObjTik() 
    tik_batches=ObjTik.tik_post_process_to_plot(tik_rec, fs, divisible_rows, n_batch)
    tik_flat = tik_batches.reshape(
    (tik_batches.shape[0] * tik_batches.shape[1], tik_batches.shape[2])
    )
    tik_rec_norm = normalize_array(tik_flat, high=1, low=-1, axis_n=0) 

    # Calcular correlación por nodo
    corr = correlation_by_node(tik_rec_norm, y_label)
    RMSE = rmse_by_node(tik_rec_norm, y_label)

    corr_list.append(np.mean(corr))
    rmse_list.append(np.mean(RMSE))


df_metrics = pd.DataFrame({
    "name": test_patients,
    "mean correlation": corr_list,  # Corrige el nombre de la columna eliminando el typo ("orrelation" a "correlation")
    "mean RMSE": rmse_list
})
df_metrics.head()
# Guardar el DataFrame como un archivo CSV
output_path = output_directory + "metrics_tik.csv"  # Asegúrate de que `output_directory` termine con "/"
df_metrics.to_csv(output_path, index=False)
print('Metrics saved in',  output_path)