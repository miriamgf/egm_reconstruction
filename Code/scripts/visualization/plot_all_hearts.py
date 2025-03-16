import sys
sys.path.append("../Code")
import os
import json
from scipy.io import loadmat
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import ast
import argparse
from tools_.preprocess_data import Preprocess_Dataset
from scripts.visualization.utils.renderizer import EGMRenderer_BSP
from scripts.config import ParseHiperparams
from tools_.load_dataset import LoadDataset_BSPS
from scripts.visualization.utils.bsp_3d_plotter import BSP_3D_PLOTTER
from scripts.visualization.utils.egm_3d_plotter import EGM_3D_PLOTTER
from scripts.visualization.utils.corr_3d_plotter import CORRELATION_3D_PLOTTER
from scripts.visualization.utils.rmse_3d_plotter import RMSE_3D_PLOTTER
from scripts.visualization.utils.df_map_3d_plotter import DF_MAPS_3D_PLOTTER
from scripts.visualization.utils.metric_3d_plotter import METRIC_3D_PLOTTER
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer
import tools_.tools as tools
from scripts.evaluation.metrics import Metrics
from tools_.tools_inference import postprocess_prediction
import pandas as pd



from scripts.Tikhonov.compute_tik import TikhonovReconstruction
from scripts.evaluate_function import *
from tools_.tools import corr_pearson_cols
from tools_.tools_inference import *
from tools_ import freq_phase_analysis as freq_pha

os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"

import time


#---------------------------------------------------------------------------------------------------------------------
# CONFIGURE
#---------------------------------------------------------------------------------------------------------------------



torso_num=2
all_patients = []
directory = "/home/profes/miriamgf/tesis/Autoencoders/Data/"

for subdir, dirs, files in os.walk(directory):
    # print(subdir, directory, files)

    if subdir != directory:
        model_name = subdir.split("/")[-1]
        all_patients.append([model_name])


try:
        print("Parsing bash params")
        parser = argparse.ArgumentParser(description="params")
        parser.add_argument("--algorithm_ID", type=str, help="experiment name", required=True)
        

        args = parser.parse_args()
        algorithm_ID = args.algorithm_ID
        experiment_ID_list=[[algorithm_ID]]

        print(algorithm_ID)
    
except:
        experiment_ID_list=[["OMAMI_no_filt_testing2_repeated"]]


testing_id=0
start = time.time()
cont=0


for model_name in all_patients:

    print(f"Loading patient {cont}/{len(all_patients)}")
    cont+=1

    start_=time.time()

    model_name=model_name
    algorithm_ID=experiment_ID_list[0][0]

    torso_path=f"/home/pdi/miriamgf/tesis/Autoencoders/Labeled_torsos/Torso{torso_num}_mod.mat"
    geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
    geom_path_edgar= "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_Edgar/Atria.mat"
    output_directory = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/renderizations/{algorithm_ID}/{model_name[0]}"
    os.makedirs(output_directory, exist_ok=True)
    data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
    torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"
    regions_path = "/home/pdi/miriamgf/tesis/Autoencoders/Regions/regions.mat"

    experiment_dir=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{algorithm_ID}/"
    model_path_DL=experiment_dir+f"reconstructions_by_model_{algorithm_ID}.mat"
    weights_path = experiment_dir + "model_weights.h5"
    params_path=experiment_dir+'hyperparams.json'

    #Load params dictionary
    with open(experiment_dir+"hyperparams.json") as file:
        params = json.load(file)  # Load the JSON data into a dictionary

    fs=params["fs_sub"]
    n_batch=params["batch_size"]

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
    bspm_signal_norm = tools.normalize_array(bspm_signal.T, high=1, low=-1, axis_n=0) 
    egm_single_norm = tools.normalize_array(egm_single, high=1, low=-1, axis_n=0) 

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

    egm_flat = egm_batches.reshape(
        (egm_batches.shape[0] * egm_batches.shape[1], egm_batches.shape[2])
    )

    y_label=egm_flat
    y_label=normalize_by_models(egm_flat, Y_model)

    time_duration=y_label.shape[0] # num of samples to represent

    EGM_3d_object=EGM_3D_PLOTTER(model_name,
                model_path_DL,
                geom_path_CF,
                output_directory="/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/videos_all_patients/",
                labels_mode=False,
                tikhonov=False,
                time=time_duration)

    _, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()

    print("Plotting DL reconstruction")
    EGM_3d_object.plot_only_label(y_label, faces_heart, vertices_heart, normalizar=False, frames=10)

