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
from models.gen_vae import Gen_VAE
import tools_.tools as tools
from scripts.evaluation.metrics import Metrics
from tools_.tools_inference import postprocess_prediction
import pandas as pd
import tensorflow as tf



from scripts.Tikhonov.compute_tik import TikhonovReconstruction
from scripts.evaluate_function import *
from tools_.tools import corr_pearson_cols
from tools_.tools_inference import *
from tools_ import freq_phase_analysis as freq_pha
from models.attention import LuongAttention


os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"

import time


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
plot_coherence_DL = True
plot_coherence_tik = True
plot_DTW_DL = True
plot_DTW_tik = True

plot_DF_maps_DL = False
plot_DF_maps_tik = False

load_tik_array=True

torso_num=2

test_patients = [["Simulation_01_200212_001_  5"]]

#PRECONFIG
params = ParseHiperparams().parse_default_hyperparams()
params["split_mode"] = "stratified"
params["oversampling"] = False
params["classes_to_oversample"]= [0,1, 5]
params["latent_dim"]=250

try:
        print("Parsing bash params")
        parser = argparse.ArgumentParser(description="params")
        parser.add_argument("--algorithm_ID", type=str, help="experiment name", required=True)
        

        args = parser.parse_args()
        algorithm_ID = args.algorithm_ID
        experiment_ID_list=[[algorithm_ID]]

        print(algorithm_ID)
    
except:
        experiment_ID_list=[["OMAMI_VAE_baseline"]]


testing_id=0
start = time.time()
cont=0


for model_name in test_patients:

    print(f"Loading patient {cont}/{len(test_patients)}")
    cont+=1

    start_=time.time()
    for algorithm_ID in experiment_ID_list: 

        model_name=model_name
        algorithm_ID=algorithm_ID[0]

        torso_path=f"/home/pdi/miriamgf/tesis/Autoencoders/Labeled_torsos/Torso{torso_num}_mod.mat"
        geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
        geom_path_edgar= "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_Edgar/Atria.mat"
        output_directory = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/renderizations/synthetic_gen/{algorithm_ID}/{model_name[0]}"
        os.makedirs(output_directory, exist_ok=True)
        data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
        torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"
        regions_path = "/home/pdi/miriamgf/tesis/Autoencoders/Regions/regions.mat"

        experiment_dir=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/synthetic_generation/{algorithm_ID}/"
        model_path_DL=experiment_dir+f"reconstructions_by_model_{algorithm_ID}.mat"
        weights_path = experiment_dir + "model_weights.h5"
        params_path=experiment_dir+'hyperparams.json'


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
            inference=True, 
            split_mode="stratified"
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
        egm_batches = egm_batches.squeeze(-1)
        vae = Gen_VAE(
            params=params,
            input_shape_=egm_batches.shape[1:], 
            n_nodes=2048,
            latent_dim=params["latent_dim"],
            tensorboard_logs=experiment_dir + "tb_logs/"
        )

        vae.model.load_weights(experiment_dir + "model_weights.h5")
        decoder = vae.build_decoder_from_latent()


        # Samplear del espacio latente
        z = tf.random.normal((1, vae.latent_dim))  # 10 muestras aleatorias

        # Generar señales sintéticas
        synthetic = decoder.predict(z)
        print(synthetic.shape)  # → (10, 2048)


    


        #DL Predictions
        #---------------------------------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------------------------------

        if plot_DL:

            EGM_3d_object=EGM_3D_PLOTTER(model_name,
                        model_path_DL,
                        geom_path_CF,
                        output_directory,
                        labels_mode=False,
                        tikhonov=False,
                        time=100)

            _, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()

            print("Plotting DL reconstruction")
            if synthetic.ndim == 3:
                synthetic = synthetic.squeeze(0)
            EGM_3d_object.plot_3d_mesh_prediction(synthetic, synthetic, faces_heart, vertices_heart)

            
end = time.time()
        
print('Execution time TOTAL: ', end-start, 'min')
