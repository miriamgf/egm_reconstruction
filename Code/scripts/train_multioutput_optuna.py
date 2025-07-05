# This script was developed Miriam Gutiérrez Fernández

import sys
sys.path.append("../Code")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import argparse
import datetime
import time as t
import os
import pickle
import random
import time
import json

import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
from numpy import *
from scipy.io import savemat
import h5py
import argparse
import datetime
import time
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import load_model

from tools_.train_model import TrainModel
from config import ParseHiperparams, GetMetadata
from src.training.optuna_opt import OptunaOpt
from config import str_to_bool
from tools_.data_augmentation import DataAugmentation
from tools_.load_dataset import LoadDataset
from tools_.preprocess_data import Preprocess_Dataset
from evaluate_function import evaluate_function_multioutput, evaluate_function_multioutput
from tools_.preprocessing_compression import *
import tools_.tools
from tools_.df_mapping import *
from tools_.tools import *



# Clear GPU
K.clear_session()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

SEED = 42

gpus = tf.config.experimental.list_physical_devices('GPU')
print('Available GPUS:', gpus)

params = ParseHiperparams().parse_default_hyperparams()
#default params
SNR_em_noise = None

patches_oclussion = "PT"
unfold_code = 1
params["SNR_white_noise"]=100
params["filter_EGM"]=False
params['optuna_optimization']=False
print('Params to train: ', params)
params["algorithm"]="OMAMI" #default

#["algorithm"]='OMAMI_VAE'
if params["algorithm"]=='OMAMI':

    algorithm_ID_copy_config="OMAMI_no_filt_testing2_repeated"
    path_best_params=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{algorithm_ID_copy_config}/hyperparams.json"
    params=ParseHiperparams().load_best_hyperparams(path_best_params)
    params['optuna_optimization']=False
    print(f"Load OMAMI: {algorithm_ID_copy_config} Optimal hyperparams")

elif params["algorithm"]=='OMAMI_VAE':
    
    algorithm_ID_copy_config= "OMAMI_VAE_no_filt_testing_repeated"
    path_best_params=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{algorithm_ID_copy_config}/hyperparams.json"
    params=ParseHiperparams().load_best_hyperparams(path_best_params)
    params['optuna_optimization']=False
    print(f"Load OMAMI VAE: {algorithm_ID_copy_config} Optimal hyperparams")

try:
    print("Parsing bash params")
    parser = argparse.ArgumentParser(description="Noise params")
    parser.add_argument("--algorithm", type=str, help="experiment name", required=True)
    parser.add_argument("--optuna", type=str_to_bool, help="True or False", required=False)
    parser.add_argument("--n_nodes", type=int, help="682, 1024", required=False)
    parser.add_argument("--fold", type=int, help="0, 1, 2, 3, 4", required=False)
    parser.add_argument("--filter_EGM", type=str_to_bool, help="True or False", required=False)

    # Data Augmentation
    parser.add_argument("--shuffle_patient", type=str_to_bool, help="True or False", required=False)
    parser.add_argument("--time_masking", type=str_to_bool, help="True or False", required=False)
    parser.add_argument("--attention", type=str_to_bool, help="True or False", required=False)

    # Stratified split
    parser.add_argument("--split_mode", type=str, help="stratified or random", required=False)
    parser.add_argument("--oversampling", type=str_to_bool, help="True or False", required=False)
    parser.add_argument("--discard_classes", type=str_to_bool, help="True or False", required=False)


    #Noise
    parser.add_argument('--SNR_em_noise', type=int, help='EM noise SNR', required=False)
    parser.add_argument('--SNR_white_noise', type=int, help='white noise SNR', required=False)
    parser.add_argument('--patches_oclussion', type= str, help='Oclussion patches', required=False)
    parser.add_argument('--unfold_code', type=int, help='Unfolding order', required=False)


    args = parser.parse_args()
    algorithm = args.algorithm
    optuna = args.optuna
    n_nodes = args.n_nodes
    filter_EGM= args.filter_EGM
    fold=args.fold
    shuffle_patient=args.shuffle_patient
    time_masking=args.time_masking
    SNR_white_noise=args.SNR_white_noise
    SNR_em_noise=args.SNR_em_noise
    attention=args.attention
    split_mode=args.split_mode
    oversampling=args.oversampling
    discard_classes=args.discard_classes
    

    params["algorithm"]=algorithm

    #Load best hiperparams
    if params["algorithm"]=='OMAMI':

        algorithm_ID_copy_config="OMAMI_no_filt_testing2_repeated"
        path_best_params=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{algorithm_ID_copy_config}/hyperparams.json"
        params=ParseHiperparams().load_best_hyperparams(path_best_params)
        params['optuna_optimization']=False
        print(f"Load OMAMI: {algorithm_ID_copy_config} Optimal hyperparams")

    elif params["algorithm"]=='OMAMI_VAE':
        
        algorithm_ID_copy_config= "OMAMI_VAE_no_filt_testing_repeated"
        path_best_params=f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/{algorithm_ID_copy_config}/hyperparams.json"
        params=ParseHiperparams().load_best_hyperparams(path_best_params)
        params['optuna_optimization']=False
        print(f"Load OMAMI VAE: {algorithm_ID_copy_config} Optimal hyperparams")

    params["n_nodes_regression"]=n_nodes

    # Data Augmentation
    if split_mode is not None:
        params["split_mode"]=split_mode
    else:
        params["split_mode"] = "deterministic"

    if shuffle_patient is not None:
        params["shuffle_patient"]=shuffle_patient
    else:
        params["shuffle_patient"]=False
    
    if time_masking is not None:
        params["time_masking"]=time_masking
    else:
        params["time_masking"]=False
    
    if SNR_white_noise is not None:
        params["SNR_white_noise"]=SNR_white_noise
    else:
        params["SNR_white_noise"]=100

    if attention is not None:
        params["attention_layer"]=attention
    else:
        params["attention_layer"]=False

    # Stratified split

    if discard_classes:
        params["discard_classes"]=[0, 1, 3, 5]
        params["classes_to_oversample"]= [4]
        split_mode=params["split_mode"]
        

    else:
        params["discard_classes"]=[]
        params["classes_to_oversample"]= [0,1, 5]


    if split_mode is not None:
        params["split_mode"]=split_mode
    else:
        params["split_mode"]="stratified"
    if oversampling is not None:
        params["oversampling"]=oversampling
    else:
        params["oversampling"]=False
    
    #Optuna

    if optuna:
        params["optuna_optimization"] = True
        print('Optuna activated. Launching', params["n_trials"], 'trials')


    if filter_EGM is not None:  
        params["filter_EGM"] = filter_EGM
        print('EGM filtering activated (not filtering)')

    
    if fold is not None:
        fold=args.fold
        params['cross_validation']=True
        params["fold"]=fold
        print('Cross val activated with fold: ', fold)

except SystemExit as e:
    print(e)

    algorithm = params["algorithm"]
    SNR_white_noise = 100
    params['SNR_white_noise']=20
    params['attention_layer']=False

    print('Failed in parsing bash params :( ')
    pass



experiment_name = algorithm_ID_copy_config

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

if params["optuna_optimization"]:
    experiment_name = f"{experiment_name}_Optuna"

#params["n_epochs"]=1
print(params)

#Regularization experiments
try:
    print(params["shuffle_patient"])
except:
    params["shuffle_patient"]=False
try:
    print(params["time_masking"])
except:
    params["time_masking"]=False

try:
    print(params['split_mode'])
except:
    params['split_mode']="deterministic"


experiment_name = experiment_name + "_l2"
params["l2_reg"]=0.001


if params["shuffle_patient"]:
    experiment_name= experiment_name + "_shuffle_patient"
if params["time_masking"]:
    experiment_name= experiment_name + "_tm"
if params["SNR_white_noise"] != 100:
    print("SNR 20 Applied")
    experiment_name= experiment_name + "_SNR20"
if params["attention_layer"]:
    print("Attention layer added")
    experiment_name= experiment_name + "_attention"

params["early_stopping_patience"] = 40
if len(params["discard_classes"])>0:
    experiment_name= experiment_name + "_strat_2_class_overs"
else:
    experiment_name= experiment_name + "_strat_5_class_overs"

if params['split_mode'] == "deterministic":
    experiment_name= experiment_name + "_det"
else:
    pass

print(params)
print('Experiment name: ', experiment_name)


#experiment_name="TOY"
print('Experiment name: ', experiment_name)
params["experiment_name"]=experiment_name

root_logdir = "output/logs/"
log_dir = root_logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
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

start = t.time()

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
    y_list, 
    class_complexity_list
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
    inference=False,
)()



plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(egm_tensor[0:500, 0], label='egm')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(X_1channel[0:500, 0, 0], label='bspm')
plt.legend()
plt.savefig(experiment_dir+'loaded_signals_feat_opt.png')
print('saved image at ', experiment_dir+'loaded_signals_feat_opt.png')
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

if params["optuna_optimization"]:
    print("Starting Optuna optimization...")
    params = OptunaOpt(
        params=params,
        X_1channel=X_1channel,
        egm_tensor=egm_tensor,
        AF_models=AF_models,
        Y_model=Y_model,
        dic_vars=dic_vars,
        Y=Y,
        all_model_names=all_model_names,
        transfer_matrices=transfer_matrices,
        models_dir=models_dir,
        experiment_dir=experiment_dir,
    )()

print('params after optimization', params)

"""
plt.figure()
plt.plot(X_1channel[0:200, 0, 0], label='bsps')
plt.plot(egm_tensor[0:200, 0], label='egm')
plt.legend()
os.makedirs('output/figures/input_output/', exist_ok=True)
plt.savefig('output/figures/input_output/before_norm.png')
"""


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
    class_complexity_list_train, class_complexity_list_test, class_complexity_list_val,
    train_models,
    test_models,
    val_models,
) = Preprocess_Dataset(
    params,
    X_1channel,
    egm_tensor,
    AF_models,
    class_complexity_list,
    Y_model,
    dic_vars,
    Y,
    all_model_names,
    transfer_matrices,
    experiment_dir,
    split_mode=params["split_mode"],
    norm_egm=True,
    shuffle_patient= params["shuffle_patient"], 
)()


plt.figure()
plt.plot()

print('################ CHECKING DISTRIBUTION AFTER PREPROCESSING ################')

try:
    assert x_train[0, :, 0, 0, 0].max() == 1, "No cumple egm_tensor[:, 0].max()==1"

except:
    print("No cumple x_train[0, :, 0, 0, 0].max()==1")

try:
    assert y_train[0, :, 0].max() == 1, "No cumple y_train[0, :, 0].max()==1"

except:
    print("No cumple X_1channel[:, 0, 0].max()==1")

plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(x_train[0, :, 0, 0, 0], label='x_train')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(y_train[0, :, 0], label='y_train')
plt.legend()
plt.savefig(experiment_dir+'preprocessed_signals_feat_opt.png')
print('saved image at ', experiment_dir+'preprocessed_signals_feat_opt.png')
plt.close()


#Data Augmentation
if params["time_masking"]:
    x_train = DataAugmentation(params, x_train).time_masking()
    print("Data augmentation applied")

print("Algorithm selected:", params["algorithm"])
model, history = TrainModel(
    params, x_train, x_test, x_val, y_train, y_test, y_val, models_dir, experiment_dir
)()


############## INFERENCE ###############

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, (x_test, y_test))) \
    .batch(params["num_batch_iter"], drop_remainder=False)  \
    .repeat(1).cache()

        #.prefetch(tf.data.experimental.AUTOTUNE)
print('Test prediction...')

try:
    with tf.device('/GPU:0'):  
        pred_test = model.predict(x_test,batch_size=params["num_batch_iter"])#, batch_size=5)#, batch_size=5)
        print('Train prediction')
        x_train = x_train[0:50, :, :, :, :]
        y_train = y_train[0:50, :, :]
        pred_train = model.predict(x_train, batch_size=params["num_batch_iter"])

except: 
    with tf.device('/GPU:1'):  
        pred_test = model.predict(x_test,batch_size=params["num_batch_iter"])#, batch_size=5)#, batch_size=5)
        print('Train prediction')
        x_train = x_train[0:50, :, :, :, :]
        y_train = y_train[0:50, :, :]
        pred_train = model.predict(x_train, batch_size=params["num_batch_iter"])


'''
print('Evaluating...)')
results_autoencoder, results_regressor = evaluate_function_multioutput(
    x_train, y_train, x_test, y_test, pred_train, pred_test, model, batch_size=1
)

pred_test_autoencoder, pred_test_egm = pred_test[0], pred_test[1]
pred_train_autoencoder, pred_train_egm = pred_train[0], pred_train[1]

print("Results autoencoder:")
print(results_autoencoder)

print("Results regressor:")
print(results_regressor)

new_items = {
    "pred_test": pred_test,
    "pred_train": pred_train,
    "conv_autoencoder": model,
}
dic_vars.update(new_items)


y_test_flat = reshape_tensor(y_test, n_dim_input=y_test.ndim, n_dim_output=2)
reconstruction_flat_test = reshape_tensor(
    pred_test_egm, n_dim_input=pred_test_egm.ndim, n_dim_output=2
)

x_test_flat = reshape_tensor(x_test, n_dim_input=x_test.ndim, n_dim_output=2)
autoencoder_flat_test = reshape_tensor(
    pred_test_autoencoder, n_dim_input=pred_test_autoencoder.ndim, n_dim_output=2
)
estimate_egms_test = reconstruction_flat_test

# normalize reconstrutions
estimate_egm_test_r = estimate_egms_test
estimate_egms_n = reconstruction_flat_test
estimate_egms_n = normalize_by_models(reconstruction_flat_test, AF_models_test)

pred_test_egm_fl = reshape(
    pred_test_egm,
    (pred_test_egm.shape[0] * pred_test_egm.shape[1], pred_test_egm.shape[2]),
)
y_fl = reshape(y_test, (y_test.shape[0] * y_test.shape[1], y_test.shape[2]))
x_fl = reshape(
    x_test,
    (
        x_test.shape[0] * x_test.shape[1],
        x_test.shape[2] * x_test.shape[3] * x_test.shape[4],
    ),
)


# Reconstruction predictions
for i in range(0, 30):
    interv = random.randrange(1, len(pred_test_egm_fl) - 1, 50)
    node = random.randrange(1, estimate_egms_n.shape[-1], 1)
    normalize_ = True
    rango = 500
    # normalize between -1 and 1
    estimate_signal = pred_test_egm_fl[interv : interv + rango, :]
    estimate_egms_norm_represent = normalize_array(estimate_signal, 1, -1)
    plt.figure(figsize=(15, 3))
    plt.subplot(2, 1, 1)
    plt.plot(estimate_egms_norm_represent[:, node], label="Estimation Test")
    # plt.plot(estimate_egms_norm[interv: interv+200 ,node], label='Estimation Test')
    plt.plot(y_fl[interv : interv + rango, node], label="Test", alpha=0.5)
    # plt.plot(latent_vector_test[200:400, 0, 0, 0], label = 'Latent Vector')
    text = "Node {} in second {} to {}".format(
        node, interv / fs, interv / fs + rango / fs
    )
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title(text)
    plt.subplot(2, 1, 2)
    x_signal = x_fl[interv : interv + rango, :]
    x_norm_represent = normalize_array(x_signal, 1, -1)
    plt.plot(x_norm_represent[:, 0:5], label="BSP")
    # plt.plot(estimate_egms_norm[interv: interv+200 ,node], label='Estimation Test')
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title("BSPM")
    plt.savefig(experiment_dir + "EGM_Reconstructions_" + str(i) + ".png")

    plt.close()
    

time_instant = random.randint(0, params["batch_size"])
batch = random.randrange(2, x_test.shape[0]-2, 1)

# Reconstrauction Autoencoders
for i in range(0, 2):
    batch = batch + 1
    plt.figure(tight_layout=True)
    plt.subplot(3, 1, 1)
    plt.imshow(x_test[batch, 0, :, :, 0])
    plt.title("X test Autoencoder - batch" + str(batch))
    plt.colorbar(label="Colorbar Label")  # Add a colorbar with a label
    plt.subplot(3, 1, 2)
    plt.imshow(pred_test_autoencoder[batch, 0, :, :, 0])
    plt.title("Test predictions Autoencoder - batch" + str(batch))
    plt.colorbar(label="Colorbar Label")  # Add a colorbar with a label
    plt.subplot(3, 1, 3)
    difference = x_test[batch, 0, :, :, 0] - pred_test_autoencoder[batch, 0, :, :, 0]
    plt.imshow(difference)
    plt.title("Error")
    plt.colorbar(label="Colorbar Label")  # Add a colorbar with a label
    plt.savefig(experiment_dir + "Autoencoder_reconstructions" + str(i) + ".png")
    plt.show()
    plt.close()

# 2D EGM plots
plt.figure(layout="tight", figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.imshow(normalize_array(y_fl[0:1000, :].T, 1, 0, 0), cmap="Greys")
plt.title("y test (egm)")
plt.xlabel("time")
plt.ylabel("nodes")
plt.colorbar(orientation="horizontal", pad=0.2)
plt.subplot(1, 3, 2)
plt.imshow(normalize_array(pred_test_egm_fl[0:1000, :].T, 1, 0, 0), cmap="Greys")
plt.title("Reconstruction")
plt.xlabel("time")
plt.ylabel("nodes")
plt.colorbar(orientation="horizontal", pad=0.2)
plt.subplot(1, 3, 3)
dif = normalize_array(y_fl[0:1000, :].T, 1, 0, 0) - normalize_array(
    pred_test_egm_fl[0:1000, :].T, 1, 0, 0
)
dif = abs(dif)
plt.imshow(dif, cmap="Greys")
plt.title("y_test - reconstruction")
plt.xlabel("time")
plt.ylabel("nodes")
plt.colorbar(orientation="horizontal", pad=0.2)
plt.savefig(experiment_dir + "2D_EGM_predictions.png")
plt.show()
plt.close()

# PSD
nperseg_value = 2 * params["fs_sub"]
fig = plt.figure(layout="tight", figsize=(10, 6))
plt.subplot(3, 1, 3)
fs = params["fs_sub"]
# EGM reconstruction
for height in range(0, estimate_egms_n.shape[1], 5):
    f, Pxx_den = scipy.signal.welch(
        estimate_egms_n[:, height],
        fs,
        nperseg=nperseg_value,
        noverlap=nperseg_value // 2,
        scaling="density",
        detrend="linear",
    )
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    # plt.ylim([0,0.2])
    titlee = "PSD Welch of EGM signal estimation. node {}".format(height)
    plt.title("EGM signals reconstruction")

plt.subplot(3, 1, 2)
input = reshape(
    x_test,
    (
        x_test.shape[0] * x_test.shape[1],
        x_test.shape[2] * x_test.shape[3] * x_test.shape[4],
    ),
)
for height in range(0, input.shape[1], 5):
    f, Pxx_den = scipy.signal.welch(
        input[:, height],
        fs,
        nperseg=nperseg_value,
        noverlap=nperseg_value // 2,
        scaling="density",
        detrend="linear",
    )
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    # plt.ylim([0,0.2])
    titlee = "PSD of BSP (Input). node {}".format(height)
    plt.title("BSP (Input) ")

plt.subplot(3, 1, 1)
for height in range(0, y_fl.shape[1], 5):
    f, Pxx_den = scipy.signal.welch(
        y_fl[:, height],
        fs,
        nperseg=nperseg_value,
        noverlap=nperseg_value // 2,
        scaling="density",
        detrend="linear",
    )
    plt.plot(f, Pxx_den, linewidth=0.5)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    # plt.ylim([0,0.2])
    titlee = "PSD of original egm (Input). node {}".format(height)
    plt.title("EGM (Label) ")

fig.suptitle("Welch Periodogram (window size=200 samples)", fontsize=15)
plt.savefig(experiment_dir + "PSD.png")
plt.close()



# Calculate metrics DTW, RMSE and Correlation BY AF MODELS: Meand and std
# *This metrics are calculated appart because thay are not computed in evaluate_function, (...)
# (...) as they cannot be included in the tensorflow metric callback

# TODO: Arreglar DTW
dtw_array, dtw_array_random = [
    0,
    0,
]  # DTW_by_AFModels(AF_models_test, estimate_egms_n, y_test_subsample)
rmse_array, rmse_df_test = RMSE_by_AFModels(AF_models_test, estimate_egms_n, y_test_flat, all_model_names)
rmse_df_test.to_csv(experiment_dir+"/rmse_df_test.csv", index=False) # Save RMSE by nodes to csv

correlation_array, test_models_corr, corr_df_test = correlation_by_AFModels(
    AF_models_test, estimate_egms_n, y_test_flat, all_model_names
)
corr_df_test.to_csv(experiment_dir+"/corr_df_test.csv", index=False) # Save correlation by nodes to csv


# Mean and STD of Spearman Correlation, DTW and RMSE
corr_mean = np.mean(correlation_array, axis=1)
corr_std = np.std(correlation_array, axis=1)

dtw_mean = 0  # np.mean(dtw_array, axis=1)
dtw_std = 0  # np.std(dtw_array, axis=1)
dtw_mean_random = 0  # np.mean(dtw_array_random, axis=1)
dtw_std_random = 0  # np.std(dtw_array_random, axis=1)
rmse_mean = np.mean(rmse_array, axis=1)
rmse_std = np.std(rmse_array, axis=1)


new_items = {
    "corr_mean": corr_mean,
    "corr_std": corr_std,
    "rmse_mean": rmse_mean,
    "rmse_std": rmse_std,
    "dtw_mean": dtw_mean,
    "dtw_std": dtw_std,
    "dtw_mean_random": dtw_mean_random,
    "dtw_std_random": dtw_std_random,
}
dic_vars.update(new_items)

# results = pd.DataFrame(
# columns=["MSE AE", "DTW AE", "MSE Reconstruction", "TWD Reconstruction"]
# )

# Interpolation for mapping in 3D
estimate_egms_reshaped = reshape(
    estimate_egms_n, (estimate_egms_n.shape[0], estimate_egms_n.shape[1], 1, 1)
)
interpol = interpolate_reconstruction(estimate_egms_reshaped, method="bicubic")
test_estimation = reshape(interpol, (interpol.shape[0], interpol.shape[1]))
label_represent = y_test_flat[:, :]
estimate_labels_reshaped = reshape(
    label_represent, (label_represent.shape[0], label_represent.shape[1], 1, 1)
)

interpol_label = interpolate_reconstruction(estimate_labels_reshaped, method="bicubic")
label = reshape(interpol_label, (interpol_label.shape[0], interpol_label.shape[1]))


print("Saving variables...")

new_correlation_array = interpolate_fun(
    correlation_array, len(correlation_array), y_train.shape[2]
)
new_rmse_array = interpolate_fun(rmse_array, len(rmse_array), y_train.shape[2])
'''
# %%
# Save the model names in train, test and val
test_model_name = [all_model_names[index] for index in AF_models_test]
val_model_name = [all_model_names[index] for index in AF_models_val]
train_model_name = [all_model_names[index] for index in AF_models_train]

#mdic = {"reconstruction": test_estimation, "label": label}



variables = {
    "test_model_name": np.unique(test_model_name),
    "train_model_name": np.unique(train_model_name),
    "val_model_name": np.unique(val_model_name),
}

#dic_latent_space_test = {"Latent_space_test": pred_test_autoencoder}
#savemat(experiment_dir + "/autoencoder.mat", dic_latent_space_test)


# Write dictionary string representation to text file
file_path = experiment_dir + "metrics.txt"

with open(file_path, "w") as f:
    for key, value in variables.items():
        f.write(f"{key}: {value}\n")


# Crear un archivo para guardar el summary

file_name = experiment_dir + "model_summary.txt"

with open(file_name, "w") as f:
    # Redirigir la salida estándar al archivo
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# Save results to csv and export
'''
results_Autoencoder = pd.DataFrame.from_dict(
    results_autoencoder, orient="index", columns=["Autoencoder"]
)
results_Reconstruction = pd.DataFrame.from_dict(
    results_regressor, orient="index", columns=["Reconstruction"]
)
global_results = pd.concat([results_Autoencoder, results_Reconstruction], axis=1)
global_results.to_csv(experiment_dir + "/Results_MO.csv")


global_results.round(3)
'''
# Save dictionaries into pickle and .mat

#with open(dict_var_dir + "variables_MO.pkl", "wb") as fp:
    #pickle.dump(dic_vars, fp)
#with open(dict_results_dir + "dict_results_reconstruction_MO.pkl", "wb") as fp:
    #pickle.dump(results_regressor, fp)
#with open(dict_results_dir + "dict_results_autoencoder_MO.pkl", "wb") as fp:
    #pickle.dump(results_autoencoder, fp)

# savemat(dict_var_dir + "dic_vars.mat", dic_vars) #TODO: cannot be saved to .mat because now is saving a keras model
#savemat(dict_results_dir + "dict_results_autoencoder.mat", results_autoencoder)
#savemat(dict_results_dir + "dict_results_reconstruction.mat", results_regressor)


# %%
end = t.time()
params["commit_hash"]=GetMetadata().get_git_commit()

params["execution_time"] = (end - start) / 60
# Specify the file path
file_path = experiment_dir + "hyperparams.json"

# Escribe el diccionario como JSON en el archivo
with open(file_path, "w") as f:
    json.dump(params, f, indent=2)

#track git commit hash 

print((end - start) / 60, "Mins of execution")
print("-------------EXPERIMENT RED MULTIOUPUT'--------------", experiment_name)
