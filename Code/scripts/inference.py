# This script was developed Miriam Gutiérrez Fernández

import sys
sys.path.append("../Code")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import argparse
import datetime
import os
import pickle
import random
import time
import json

import matplotlib.pyplot as plt
#import mlflow
import scipy
import tensorflow as tf
import tools_
import tools_.tools as tools
import tools_.oclusion
from evaluate_function import evaluate_function_multioutput, evaluate_function_multioutput
from models.multioutput_VAE import MultiOutput_VAE, SamplingLayer
from tools_.load_dataset import LoadDataset_BSPS


from numpy import *
from scipy.io import savemat
import h5py

import tools_
import tools_.tools
 

from tools_.df_mapping import *
from tools_.tools import *

import argparse
import datetime
import time

import tensorflow as tf
from config import ParseHiperparams, GetMetadata
from keras import backend as K
from tensorflow.keras.models import load_model
from config import str_to_bool

from tools_.load_dataset import LoadDataset
from tools_.preprocess_data import Preprocess_Dataset
from tools_.preprocessing_compression import *

import platform
print("NumPy version:", np.__version__)
print("SciPy version:", scipy.__version__)
print("Python version:", platform.python_version())
np.show_config()




# Clear GPU
K.clear_session()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

SEED = 42

def correlation_by_node(array1, array2):
            """
            Calcula la correlación de Spearman entre las columnas de dos arrays.

            Args:
                array1: un array de numpy de dimensión (n,m)
                array2: otro array de numpy de dimensión (n,m)

            Returns:
                Un array de numpy de dimensión (m,) que contiene la correlación de Spearman
                de las columnas de array1 y array2.
            """

            # Verificar si ambos arrays tienen las mismas dimensiones
            assert (
                array1.shape == array2.shape
            ), "Los arrays deben tener las mismas dimensiones."

            # Calcular la correlación de Spearman de las columnas de ambos arrays
            n_cols = array1.shape[1]
            print('Computing correlation in :', n_cols, 'nodes')
            corr = np.zeros(n_cols)
            for i in range(n_cols):
                corr[i], _ = spearmanr(array1[:, i], array2[:, i]) # or pearsonr

            return corr

gpus = tf.config.experimental.list_physical_devices('GPU')
print('Available GPUS:', gpus)

#Load hyperparams
algorithm_ID= "OMAMI_VAE_no_filt_testing"
experiment_dir = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{algorithm_ID}/"
weights_path = experiment_dir + "model_weights.h5"

params_path= experiment_dir + 'hyperparams.json'
with open(params_path) as file:
    params = json.load(file)

SNR_em_noise = None
SNR_white_noise = 100
patches_oclussion = "PT"
experiment_number = 0
unfold_code = 1
experiment_name = algorithm_ID

if params["cross_validation"]:
    experiment_name = f"{experiment_name}_fold_{params['fold']}"

if not params["filter_EGM"]:  
    experiment_name = f"{experiment_name}_no_filt"

if params["algorithm"] == "OMAMI":
    params["fs_sub"]=200
    params["batch_size"]=400
else:
    params["fs_sub"]=100
    params["batch_size"]=200
    #params["loss_weight_1"]=1
    #params["loss_weight_1"]=15


if params["optuna_optimization"]:
    experiment_name = f"{experiment_name}_Optuna"
'''
params["3D_depth"]=5
params["parallel_scope"]=True
params["early_stopping_patience"]=10
params["algorithm"] = "OMAMI_VAE"
params["num_batch_iter"]=2
params["learning_rate"]=0.001
params["batch_size"]=400
params["fs_sub"]=200
params["n_epochs"]=2
'''
params["num_batch_iter"]=1
#params["batch_size"]=400
#params["fs_sub"]=200
#params["optuna_optimization"]=True
params["n_trials"]=40
#params["n_epochs"]=1
#params["parallel_scope"]=True
#params["optuna_optimization"]=True
experiment_name = f"{experiment_name}_bs_fs_Optuna"


#params["n_epochs"]=1
print(params)


#experiment_name='pruebas interpol'
print('Experiment name: ', experiment_name)


root_logdir = "output/logs/"
log_dir = root_logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data_short/"
torsos_dir = "../../../../Labeled_torsos/"
figs_dir = "output/figures/"
models_dir = "output/model/"
dict_var_dir = "output/variables/"
dict_results_dir = "output/results/"
experiment_dir = "output/experiments/experiments_VAE/" + experiment_name + "/"

#definir semilla

params["seed"] = SEED  # Agregar la semilla a los parámetros

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)    
print('Experiment dir', experiment_dir)

dic_vars = {}
dict_results = {}

# GPU Configuration
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs:", len(physical_devices))
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

start = time.time()

all_torsos_names = []
for subdir, dirs, files in os.walk(torsos_dir):
    for file in files:
        if file.endswith(".mat"):
            all_torsos_names.append(file)
torsos_dir = "/home/profes/miriamgf/tesis/Autoencoders/Labeled_torsos/"

all_torsos_names = [
    file for _, _, files in os.walk(torsos_dir) for file in files if file.endswith(".mat")
]

all_model_names = []
directory = data_dir
for subdir, dirs, files in os.walk(directory):
    # print(subdir, directory, files)

    if subdir != directory:
        model_name = subdir.split("/")[-1]
        all_model_names.append(model_name)

all_model_names = sorted(all_model_names)
print(all_model_names)


#mlflow.set_tracking_uri(uri="http://10.110.100.78:5000")
#mlflow.autolog()

# Load data
if params["fs"] == params["fs_sub"]:
    params["fs"] = params["fs_sub"]
 

Transfer_model = False  # Transfer learning from sinusoids
sinusoids = False
'''
# Load data
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
) = LoadDataset(
    params,
    directory=directory,
    data_type="1channelTensor",
    n_classes=params["n_classes"],
    downsampling=False, #deprecated
    fs=params["fs"],
    norm=False,
    SR=True,
    n_batch=params["batch_size"],
    SNR_em_noise=SNR_em_noise,
    SNR_white_noise=SNR_white_noise,
    patches_oclussion=patches_oclussion,
    unfold_code=unfold_code,
    inference=True,
)()
'''

model_name=["Simulation_01_200212_001_  5"]

(
    X_1channel,
    Y,
    Y_model,
    egm_tensor,
    length_list,
    AF_models,
    all_model_names,
    transfer_matrices,
    y_list,
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
    select_model=model_name,
)()

print('################ CHECKING DISTRIBUTION BEFORE PREPROCESSING ################')
X_1channel_load=np.load("output/figures/evaluation_trash/X_1channel_.npy")
torso_index = all_torsos_names.index(f"Torso{1}_mod.mat")

X_1channel=np.split(X_1channel, 10)[torso_index]
egm_tensor=np.split(egm_tensor, 10)[torso_index]
AF_models=np.split(np.array(AF_models), 10)[torso_index]
Y_model=np.split(np.array(Y_model), 10)[torso_index]


plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(egm_tensor[0:1000, 0], label='egm')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(X_1channel[0:1000, 0, 0], label='bspm')
plt.legend()
plt.savefig("output/figures/evaluation_trash/X_1channel_loaded_inf.png")
print("output/figures/evaluation_trash/X_1channel_loaded_inf.png")
plt.close()

try:
    assert egm_tensor[:, 0].max() == 1, "No cumple egm_tensor[:, 0].max()==1"

except:
    print("No cumple egm_tensor[:, 0].max()==1")

try:
    assert X_1channel[:, 0, 0].max() == 1, "No cumple X_1channel[:, 0, 0].max()==1"

except:
    print("No cumple X_1channel[:, 0, 0].max()==1")
#mdic = {"egm": egm_tensor, "AF_models": AF_models, "all_model_names": all_model_names}
#with h5py.File(experiment_dir + "/egm_names_all.mat", 'w') as f:
    #for key, value in mdic.items():
        #f.create_dataset(key, data=value)

print('params after optimization', params)

# Preprocess data
dic_vars={}
(X_1channel, egm_tensor, AF_models, Y_model) = Preprocess_Dataset(
    params,
    X_1channel,
    egm_tensor,
    AF_models,
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
n_batch=params["batch_size"]
divisible_rows = (rows // n_batch) * n_batch
#batch gen
x_test = reshape(
                X_1channel,
                (
                    int(len(X_1channel) / n_batch),
                    n_batch,
                    X_1channel.shape[1],
                    X_1channel.shape[2],
                    1,
                ),
            )
y_test = reshape(
                egm_tensor,
                (
                    int(len(egm_tensor) / n_batch),
                    n_batch,
                    egm_tensor.shape[1],
                    1,
                ),
            )

#X_1channel_selected = X_1channel[
    #np.where((AF_models_test == 12))
#] 



try:
    model = load_model(weights_path)
except:
    model = load_model(weights_path, custom_objects={"SamplingLayer": SamplingLayer})

#reduce number of batches

#x_test_red = x_test[:, :, :, :]
#y_test_red = y_test[:, :, :]

X_1channel_load=np.load("output/figures/evaluation_trash/X_1channel_inference.npy")

plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(X_1channel_load[0, :, 0, 0], label='ev')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x_test[0, :, 0, 0, 0], label='original')
plt.legend()
plt.savefig("/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/evaluation_trash/"+'bspms.png')
print('saved image at ', "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/evaluation_trash/"+'bspms.png')
plt.close()

x_test_red=x_test
y_test_red=y_test

pred_test = model.predict(x_test_red, batch_size=1)

pred_test_autoencoder, pred_test_egm = pred_test[0], pred_test[1]

y_test_red=np.squeeze(y_test_red, axis=-1)
y_test_flat = reshape_tensor(y_test_red, n_dim_input=y_test_red.ndim, n_dim_output=2)

reconstruction_flat_test = reshape_tensor(
    pred_test_egm, n_dim_input=pred_test_egm.ndim, n_dim_output=2
)

estimate_egms_n = reconstruction_flat_test

#AF_models_test = AF_models_test[0:len(reconstruction_flat_test)]
#estimate_egms_n = normalize_by_models(reconstruction_flat_test, AF_models_test)

estimate_egms_n=tools.normalize_array(reconstruction_flat_test, high=1, low=-1, axis_n=0)

corr = correlation_by_node(estimate_egms_n, y_test_flat)

#correlation_array, test_models_corr, corr_df_test = correlation_by_AFModels(
    #AF_models_test, estimate_egms_n, y_test_flat, all_model_names
#)

#corr_mean = np.mean(correlation_array, axis=1)

print("corr_mean", np.mean(corr))