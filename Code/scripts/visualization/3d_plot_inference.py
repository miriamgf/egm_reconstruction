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
from scipy.interpolate import Rbf



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


algorithm= "AE_Baseline_inference_egms_matrix_bspm_flat_custom_layout_flipped_cols"
geom_path_CF = "/home/pdi/miriamgf/tesis/Autoencoders/geometries/Atria_geom/Modelos_computacionales_Carlos_Fambuena/Atria.mat"
input_directory=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/CINC25_Heartlab/{algorithm}/"
output_directory= f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/CINC25_Heartlab/{algorithm}/"
torso_num=2

plot_EGMs=False
plot_tank=True
interpolate_216=True

#####################
#Load reconstructions
if plot_EGMs:
    with open(f"{input_directory}arrays_gt_egm.json", "r") as f:
        data = json.load(f)

    label=np.array(data["egms_flat"])
    reconstruction=np.array(data["reconstruction_2048"])

    EGM_3d_object=EGM_3D_PLOTTER(model_name="Sinus",
                model_path=None,
                geom_path_CF=geom_path_CF,
                output_directory=output_directory,
                labels_mode=False,
                tikhonov=False,
                time=1000)

    _, _, faces_heart, vertices_heart=EGM_3d_object.load_geometry_and_egm()

    reconstruction_norm=normalize_array(reconstruction, high=1, low=-1, axis_n=0)


    #EGM_3d_object.plot_3d_mesh_prediction(reconstruction, label, faces_heart, vertices_heart, normalizar=False)

    EGM_3d_object.plot_only_label(reconstruction_norm, faces_heart, vertices_heart, frames=1)

elif plot_tank:

    input_data_path= "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/scripts/Inference_real_models/" 
    data_dir = "/home/pdi/miriamgf/tesis/Autoencoders/Real_data/HEartLab/data_E18_F02_R02_selection.mat"
    tank_data_dir="/home/pdi/miriamgf/tesis/Autoencoders/Real_data/HEartLab/tank_geom_E18_F02_R02_selection.mat"

    mat_tank_data = loadmat(tank_data_dir)

    #Load saved bspm and signals
    tank_faces=mat_tank_data["tank_faces"] -1
    tank_vertices=mat_tank_data["tank_vertices"]
    signal_tank=mat_tank_data["signal_tank"]
    tank_el_position=mat_tank_data["tank_el_position"][0] -1

    vertices_el_position=tank_vertices[:, tank_el_position]

    #adapt to match structure
    faces=tank_faces.T
    vertices=vertices_el_position.T
    vertices=tank_vertices.T

    bspm_signal=signal_tank.T

    scalars = np.zeros((tank_vertices.shape[1], 32001))
    scalars[tank_el_position, :] = bspm_signal 

    # Interpol 
    all_vertices = tank_vertices.T
    electrode_vertices = all_vertices[tank_el_position, :]

    if interpolate_216:
        interpolated_signals = np.zeros((all_vertices.shape[0],bspm_signal.shape[1] ))
        # Recorremos cada instante temporal y precomputamos todo fuera del bucle
        for t in range(bspm_signal.shape[1]):
            values = bspm_signal[:, t]  # Valores en los 60 electrodos
            rbf = Rbf(
                electrode_vertices[:, 0],
                electrode_vertices[:, 1],
                electrode_vertices[:, 2],
                values,
                function='linear'  # o 'multiquadric', etc.
            )
            interpolated = rbf(
                all_vertices[:, 0],
                all_vertices[:, 1],
                all_vertices[:, 2]
            )
            interpolated_signals[:, t] = np.clip(interpolated, -1, 1)  

        bspm_signal=interpolated_signals
        
    BSP_3D_PLOTTER(torso_num,
                None,
                None,
                None,
                output_directory,
                None,
                time=1000).plot_3d_mesh(bspm_signal, faces, vertices)

