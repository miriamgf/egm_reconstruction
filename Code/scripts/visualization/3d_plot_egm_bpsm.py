import sys
sys.path.append("../Code")
import os
import json
from scipy.io import loadmat
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
from scripts.visualization.utils.rmse_3d_plotter import RMSE_3D_PLOTTER
from scripts.visualization.utils.df_map_3d_plotter import DF_MAPS_3D_PLOTTER
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer
from scripts.evaluation.tools_evaluate import normalize_array, downsampling
from scripts.evaluation.metrics import Metrics
from tools_.tools_inference import postprocess_prediction


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

plot_BSP = False
plot_Tikhonov = False
plot_DL= True
plot_correlation_DL = False
plot_correlation_tik = False
plot_rmse_DL = False
plot_rmse_tik = False

plot_DF_maps_DL = True
plot_DF_maps_tik = False

torso_num=2

test_patients = [
            ["LA_PLAW_140711_arm"], ["LA_RSPV_CAF_150115"],
            ["Simulation_01_200212_001_  5"], ["Simulation_01_200212_001_ 10"],
            ["Simulation_01_200316_001_  3"], ["Simulation_01_200316_001_  4"],
            ["Simulation_01_200316_001_  8"], ["Simulation_01_200428_001_004"],
            ["Simulation_01_200428_001_008"], ["Simulation_01_200428_001_010"],
            ["Simulation_01_210119_001_001"], ["Simulation_01_210208_001_002"]
        ]
test_patients = [["Simulation_01_200212_001_  5"],  
                ["Simulation_01_210119_001_001"], 
                ["Simulation_01_200428_001_010"],["Simulation_01_200212_001_ 10"]]

test_patients=[["Simulation_01_200212_001_  5"]]

experiment_ID_list=[["OMAMI_repeated"], ["OMAMI_VAE_Optuna_1"], ['OMAMI_no_filt'], ['OMAMI_VAE_no_filt']]
experiment_ID_list=[["OMAMI_VAE_Optuna_1"]]
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

        prediction=postprocess_prediction(prediction_flat, Y_model)
        time_duration=y_label.shape[0] # num of samples to represent


        #METRIC OBJECT INSTANCIATION

        MetricsObj = Metrics(algorithm_ID=algorithm_ID, model_name=model_name)
        #BSPM
        #---------------------------------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------------------------------
        if plot_BSP:
            print("Plotting BSP...")

            BSP_3D_PLOTTER(torso_num,
                        all_torsos_names,
                        y_list_or,
                        X_1channel_or,
                        output_directory,
                        torso_path,
                        time=time_duration)()

        #Tikhonov
        #---------------------------------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------------------------------
        if plot_Tikhonov or plot_correlation_tik or plot_rmse_tik or plot_correlation_tik or plot_DF_maps_tik:

            ObjTik=TikhonovReconstruction(bspm_signal_norm.T, transfer_matrix, order=0)
            tik_rec=ObjTik() 

            tik_batches=ObjTik.tik_post_process_to_plot(tik_rec, fs, divisible_rows, n_batch)
            tik_flat = tik_batches.reshape(
            (tik_batches.shape[0] * tik_batches.shape[1], tik_batches.shape[2])
        )
            tik_rec_norm = normalize_array(tik_flat, high=1, low=-1, axis_n=0) 

            EGM_3d_object=EGM_3D_PLOTTER(model_name,
                        model_path_DL,
                        geom_path_CF,
                        output_directory,
                        labels_mode=False,
                        tikhonov=True,
                        time=time_duration)

            _, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()

        if plot_Tikhonov:

            print("Plotting Tikhonov...")
            EGM_3d_object.plot_3d_mesh_prediction(tik_rec_norm, y_label, faces_heart, vertices_heart)


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
                        time=time_duration)

            _, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()

            print("Plotting DL reconstruction")
            EGM_3d_object.plot_3d_mesh_prediction(prediction, y_label, faces_heart, vertices_heart)



        #DF Mapping tik
        #---------------------------------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------------------------------

        if plot_DF_maps_tik:


            df_reconstructed, sig_k_rec, phase_rec = freq_pha.kuklik_DF_phase(tik_rec_norm.T, fs=params["fs_sub"])
            df_label, sig_k_rec, phase_rec = freq_pha.kuklik_DF_phase(y_label.T, fs=params["fs_sub"]) 

            df_reconstructed = np.squeeze(df_reconstructed)
            df_label = np.squeeze(df_label)
            print("Plotting DF Maps for ZOT")
            # Creación del objeto para mapas DF
            DFMapObject = DF_MAPS_3D_PLOTTER(
                model_name,
                torso_path,
                geom_path_CF,
                output_directory,
                labels_mode=False,
                tikhonov=True,
                time=time_duration
            )

            EGM_3d_object=EGM_3D_PLOTTER(model_name,
                        model_path_DL,
                        geom_path_CF,
                        output_directory,
                        labels_mode=False,
                        tikhonov=True,
                        time=time_duration)

            _, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()

            # Usar el método del objeto para plotear
            DFMapObject.plot_3d_mesh_label(
                df_reconstructed, df_label,  vertices_heart,faces_heart, np.min(df_reconstructed), np.max(df_reconstructed)
            )


        #DF Mapping
        #---------------------------------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------------------------------

        if plot_DF_maps_DL:

            print("Plotting DF Maps for DL")
            df_reconstructed, sig_k_rec, phase_rec = freq_pha.kuklik_DF_phase(prediction.T, fs=params["fs_sub"])
            df_label, sig_k_label, phase_label = freq_pha.kuklik_DF_phase(y_label.T, fs=params["fs_sub"])

            df_reconstructed = np.squeeze(df_reconstructed)
            df_label = np.squeeze(df_label)

            # Creación del objeto para mapas DF
            DFMapObject = DF_MAPS_3D_PLOTTER(
                model_name,
                torso_path,
                geom_path_CF,
                output_directory,
                labels_mode=False,
                tikhonov=False,
                time=time_duration
            )
            EGM_3d_object=EGM_3D_PLOTTER(model_name,
                        model_path_DL,
                        geom_path_CF,
                        output_directory,
                        labels_mode=False,
                        tikhonov=False,
                        time=time_duration)

            _, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()

            # Usar el método del objeto para plotear
            DFMapObject.plot_3d_mesh_df(
                df_reconstructed, df_label,  vertices_heart,faces_heart, np.min(df_reconstructed), np.max(df_reconstructed)
            )

            DFMapObject.plot_3d_mesh_phase(
                phase_rec, phase_label, vertices_heart,faces_heart, np.min(phase_label), np.max(phase_label)
            )

        #Correlation DL
        #---------------------------------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------------------------------
        if plot_correlation_DL:
            print("Plotting Correlation Maps")
            # Creación del objeto para correlación
            CorrelationObject = CORRELATION_3D_PLOTTER(
                model_name,
                torso_path,
                geom_path_CF,
                output_directory,
                labels_mode=False,
                tikhonov=False,
                time=time_duration
            )

            # Calcular correlación por nodo
            corr = MetricsObj.correlation_by_node(prediction, y_label)

            EGM_3d_object=EGM_3D_PLOTTER(model_name,
                        model_path_DL,
                        geom_path_CF,
                        output_directory,
                        labels_mode=False,
                        tikhonov=True,
                        time=time_duration)

            _, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()

            # Usar el método del objeto para plotear

            CorrelationObject.plot_3d_mesh_label(
                corr, faces_heart, vertices_heart, min_val_value=-1, max_val_value=1
            )

        #Correlation TIK
        #---------------------------------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------------------------------
        if plot_correlation_tik:
            print("Plotting Correlation Maps tik")
            # Creación del objeto para correlación
            CorrelationObject = CORRELATION_3D_PLOTTER(
                model_name,
                torso_path,
                geom_path_CF,
                output_directory,
                labels_mode=False,
                tikhonov=True,
                time=time_duration
            )

            # Calcular correlación por nodo
            corr = MetricsObj.correlation_by_node(tik_rec_norm, y_label)

            # Usar el método del objeto para plotear
            EGM_3d_object=EGM_3D_PLOTTER(model_name,
                        model_path_DL,
                        geom_path_CF,
                        output_directory,
                        labels_mode=False,
                        tikhonov=True,
                        time=time_duration)

            _, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()
            CorrelationObject.plot_3d_mesh_label(
                corr, faces_heart, vertices_heart, min_val_value=-1, max_val_value=1
            )



        #RMSE maps DL
        #---------------------------------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------------------------------


        if plot_rmse_DL:
            print("Plotting RMSE Maps for DL")
            # Creación del objeto para correlación
            RMSEObject = RMSE_3D_PLOTTER(
                model_name,
                torso_path,
                geom_path_CF,
                output_directory,
                labels_mode=False,
                tikhonov=False,
                time=time_duration
            )

            # Calcular correlación por nodo
            RMSE = MetricsObj.rmse_by_node(prediction, y_label)

            EGM_3d_object=EGM_3D_PLOTTER(model_name,
                        model_path_DL,
                        geom_path_CF,
                        output_directory,
                        labels_mode=False,
                        tikhonov=False,
                        time=time_duration)

            _, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()


            # Usar el método del objeto para plotear

            RMSEObject.plot_3d_mesh_label(
                RMSE, faces_heart, vertices_heart, min_val_value=0, max_val_value=1
            )

        #RMSE maps DL
        #---------------------------------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------------------------------


        if plot_rmse_tik:
            print("Plotting RMSE Maps for Tik")
            # Creación del objeto para correlación
            RMSEObject = RMSE_3D_PLOTTER(
                model_name,
                torso_path,
                geom_path_CF,
                output_directory,
                labels_mode=False,
                tikhonov=True,
                time=time_duration
            )

            # Calcular correlación por nodo
            RMSE = MetricsObj.rmse_by_node(tik_rec_norm, y_label)

            # Usar el método del objeto para plotear

            EGM_3d_object=EGM_3D_PLOTTER(model_name,
                        model_path_DL,
                        geom_path_CF,
                        output_directory,
                        labels_mode=False,
                        tikhonov=False,
                        time=time_duration)

            _, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()

        print('Chapao')
        end_ = time.time()
        print('Execution time one test example : ', end_-start_, 'min')
        sys.exit()

end = time.time()
        
print('Execution time TOTAL: ', end-start, 'min')
