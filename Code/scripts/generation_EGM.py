# This script was developed Miriam Gutiérrez Fernández

import sys
sys.path.append("../Code")
import argparse
import datetime
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
#import mlflow
import scipy
import tensorflow as tf
import tools_
import tools_.oclusion
from evaluate_function import evaluate_function_multioutput, evaluate_function_multioutput
from numpy import *
from scipy.io import savemat
import h5py

import tools_
import tools_.tools
 

from tools_.df_mapping import *
from tools_.tools import *

tf.random.set_seed(42)
import argparse
import datetime
import time

import tensorflow as tf
from config import ParseHiperparams
from keras import backend as K

from tools_.load_dataset import LoadDataset
from tools_.preprocess_data import Preprocess_Dataset
from tools_.preprocessing_compression import *
from tools_.train_model_gen import TrainModelGen

print("end imports")
# Clear GPU
K.clear_session()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()


"""
# parse args
print('parsing')
parser = argparse.ArgumentParser(description="Noise params")
parser.add_argument('--SNR_em_noise', type=int, help='EM noise SNR', required=True)
parser.add_argument('--SNR_white_noise', type=int, help='white noise SNR', required=True)
parser.add_argument('--patches_oclussion', type= str, help='Oclussion patches', required=True)
parser.add_argument('--unfold_code', type=int, help='Unfolding order', required=True)
parser.add_argument('--experiment_number', type=int,  help='number of experiment', required=True)


args = parser.parse_args()


SNR_em_noise = args.SNR_em_noise
SNR_white_noise = args.SNR_white_noise
patches_oclussion = args.patches_oclussion
unfold_code =args.unfold_code
experiment_number = args.experiment_number

print(type(patches_oclussion))
#Run script IDE

"""

params = ParseHiperparams().parse_default_hyperparams()


try:
    print("parsing")
    parser = argparse.ArgumentParser(description="Noise params")
    parser.add_argument("--algorithm", type=str, help="experiment name", required=True)
    parser.add_argument("--optuna", type=str, help="True or False", required=False)
    parser.add_argument("--n_nodes", type=int, help="682, 1024", required=False)

    args = parser.parse_args()
    algorithm = args.algorithm
    optuna = args.optuna
    n_nodes = args.n_nodes
    params["algorithm"]=algorithm
    params["n_nodes_regression"]=n_nodes

    if optuna == "True":
        params["optuna_optimization"] = True

except:
    algorithm = params["algorithm"]

print('Params to train: ', params)

SNR_em_noise = None
SNR_white_noise = 100
patches_oclussion = "PT"
experiment_number = 0
unfold_code = 1

experiment_name = algorithm

experiment_name = f"{experiment_name}_mod_data"
#experiment_name='pruebas interpol'
root_logdir = "output/logs/"
log_dir = root_logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data_short/"
torsos_dir = "../../../../Labeled_torsos/"
figs_dir = "output/figures/"
models_dir = "output/model/"
dict_var_dir = "output/variables/"
dict_results_dir = "output/results/"
experiment_dir = "output/experiments/experiments_VAE/" + experiment_name + "/"


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
) = LoadDataset(
    params,
    directory=directory,
    data_type="1channelTensor",
    n_classes=params["n_classes"],
    downsampling=False,
    fs=params["fs"],
    norm=False,
    SR=True,
    n_batch=params["batch_size"],
    sinusoid=sinusoids,
    SNR_em_noise=SNR_em_noise,
    SNR_white_noise=SNR_white_noise,
    patches_oclussion=patches_oclussion,
    unfold_code=unfold_code,
    inference=False,
)()

#mdic = {"egm": egm_tensor, "AF_models": AF_models, "all_model_names": all_model_names}
#with h5py.File(experiment_dir + "/egm_names_all.mat", 'w') as f:
    #for key, value in mdic.items():
        #f.create_dataset(key, data=value)



# Preprocess data
(
    x_train,
    x_test,
    x_val,
    y_train,
    y_test,
    y_val,
    dic_vars,
    BSPM_train,
    BSPM_test,
    BSPM_val,
    AF_models_train,
    AF_models_test,
    AF_models_val,
    train_models,
    test_models,
    val_models,
) = Preprocess_Dataset(
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
)()

#borrar
'''
y_test=y_val
x_test=x_val
AF_models_test=AF_models_val
test_models=val_models
'''

params["algorithm"] = "gen_VAE"
print("Algorithm selected:", params["algorithm"])

model, history = TrainModelGen(
    params, x_train, x_test, x_val, y_train, y_test, y_val, models_dir, experiment_dir
)()
