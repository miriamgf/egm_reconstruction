import sys

sys.path.append("../Code")
from tools_tikhonov import  load_data, load_egms_df, normalize_array
import forward_inverse_problem as fip
import filtering
from sklearn.metrics import mean_squared_error

# import data_load as dl
import precompute_matrix as pre_m
import metrics
import freq_phase_analysis as freq_pha
import sys, os
from scipy.io import loadmat
import data_load as dl
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import os
import time
import tensorflow as tf
import keras
from keras import models, layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils import shuffle
from itertools import product
from tensorflow.keras import datasets, layers, models, losses, Model
import pickle
import sys
from numpy import reshape
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import glob
from PIL import Image
import matplotlib.image
import time
import random
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.io import savemat
import forward_inverse_problem as fip
from tools import corr_spearman_cols
from tools import array_to_dic_by_models
import argparse
import math


directory = "/home/profes/miriamgf/tesis/Autoencoders/Data/"

fs = 200

#
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs:", len(physical_devices))

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

start = time.time()

all_model_names = []

for subdir, dirs, files in os.walk(directory):
    print(subdir, directory, files)

    if subdir != directory:
        model_name = subdir.split("/")[-1]
        all_model_names.append(model_name)

all_model_names = sorted(all_model_names)


# parse args
print('parsing')
parser = argparse.ArgumentParser(description="Noise params")
parser.add_argument('--SNR_em_noise', type=int, help='EM noise SNR', required=True)
parser.add_argument('--SNR_white_noise', type=int, help='white noise SNR', required=True)
parser.add_argument('--experiment_number', type=int,  help='number of experiment', required=True)


args = parser.parse_args()


SNR_em_noise = args.SNR_em_noise
SNR_white_noise = args.SNR_white_noise
experiment_number = args.experiment_number

#Run script IDE

'''
SNR_em_noise=20
SNR_white_noise=20
patches_oclussion='PT'
experiment_number=1
unfold_code=1
'''

experiment_name='_EXP_' + str(experiment_number)
experiment_dir='output/experiments/experiments_CINC/'+ experiment_name+'/'

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
    print("Directory for experiment", experiment_dir, 'created')

n_classes = 3  # 1: Rotor/no rotor ; 2: RA/LA/No rotor (2 classes) ; 3: 7 regions (3 classes) + no rotor (8 classes)


(
    X_1channel,
    Y,
    Y_model,
    egm_tensor,
    length_list,
    AF_models,
    all_model_names,
    transfer_matrices,
    AF_models_names
) = load_data(
    data_type="Flat",
    n_classes=n_classes,
    subsampling=True,
    fs_sub=fs,
    norm=False,
    SR=True,
    SNR=20,
    data_dir =directory 


)


atrial_model = 0
order = 0
x_hat_array = []
x_array = []
corr_list_mean = []
corr_list_std = []
corr_list = []

reconstruction_list = []
x_real_list = []

new_dic = {}
MSE_list = []

df = load_egms_df(directory, fs_sub=fs)


for model in range(len(all_model_names)):

    #if model ==2:
        #continue
    
    print('Computing model ', model)
    model_name= all_model_names[model]
    pos_model = np.where(np.array(AF_models_names) == model_name)
    pos_unique = np.unique(Y_model[pos_model])[0]  # Select only the first torso

    y = np.array(X_1channel[np.where(Y_model == pos_unique)])
    #y = y[:, 0:]
    af_signal_values = df.loc[df['id'] == model_name, 'AF_signal']
    af_signal_list = af_signal_values.tolist()  # Para obtener una lista
    af_signal_array = np.array(af_signal_list)
    x=af_signal_array[0]
    y = np.vstack(y).T

    plt.figure(figsize = (20, 7))
    plt.plot(y[0, :])
    plt.plot(x[0, :])

    plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Tikhonov_rec/y.png')

    #Normalize
    x= normalize_array(x, axis_n=1)
    y= normalize_array(y, axis_n=1)

    plt.figure(figsize = (20, 7))
    plt.subplot(2, 1, 1)
    plt.plot(y[0, 0:2000])
    plt.subplot(2, 1, 2)
    plt.plot(x[0,0:2000])
    plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Tikhonov_rec/y_vs_x'+str(model_name)+'.png')



    A = np.array(transfer_matrices[0][0])  # Only one torso
    AA, L, LL = pre_m.precompute_matrix(A, atrial_model, order)
    # Classical Tikhonov-based inverse problem approach.
    # x_hat_tikh, lambda_opt_tikh = classical_tikhonov_noiter(A, AA, L, LL, y, 50)
    # x_hat,lambda_opt_list,magnitude_terms_list,error_terms_list,max_lcurve_list = fip.classical_tikhonov_noiter(A, AA, L, LL, y, 500)
    x_hat, lambda_opt, magnitude_term, error_term, maxcurve_index = (
        fip.classical_tikhonov_noiter_global(A, AA, L, LL, y)
    )
    #DF_tikh, sig_k_tikh, phase_tikh = freq_pha.kuklik_DF_phase(x_hat, 500)

    # Classical Tikhonov-based inverse problem metrics
    # RDMSt_tikh, mRDMSt_tikh, stdRDMSt_tikh = metrics.RDMS_calc(x,x_hat)
    # CCt_tikh, mCCt_tikh, stdCCt_tikh = metrics.CC_calc(x,x_hat)
    x_hat = x_hat.T

    x_hat = np.array(x_hat)
    plt.figure(figsize = (20, 7))
    plt.plot(x_hat[:, 0])
    plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Tikhonov_rec/x_hat.png')
    x_array = np.array(x).T
    x_hat =  normalize_array(x_hat, axis_n=0)
    plt.figure(figsize = (20, 7))
    plt.plot(x_hat[:, 0])
    plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Tikhonov_rec/x_hat_norm.png')

    reconstruction_list.append(x_hat)
    x_real_list.append(x_array)

    correlation = corr_spearman_cols(x_hat, x_array)
    corr_list.append(correlation)
    corr_mean = np.mean(correlation)
    corr_std = np.std(correlation)
    corr_list_mean.append(corr_mean)
    corr_list_std.append(corr_std)

    plt.figure(figsize=(20,7))
    plt.plot(x_hat[0:1000,0], label="rec")
    plt.plot(x_array[0:1000,0], label="real")
    plt.legend()
    plt.title(model_name)
    plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Tikhonov_rec/x_hat_final.png')

    #plt.show()

    model_name = all_model_names[model]
    key = f"model{model_name}"

    new_dic[key] = {
        "reconstruction": x_hat.tolist(),  
        "label": x_array.tolist(),  
    }

    rmse_list_node = []
    #except:
    for node in range(0, x_hat.shape[1]):

        # RMSE
        MSE = mean_squared_error(x_hat[:, node], x_array[:, node])
        RMSE = math.sqrt(MSE)
        rmse_list_node.append(RMSE)

    MSE_list.extend([rmse_list_node])

    rmse_array = np.array(MSE_list)


rmse_mean = np.mean(rmse_array, axis=1)
rmse_std = np.std(rmse_array, axis=1)


#print(new_dic)
savemat(experiment_dir + "tikhonov_matlab.mat", new_dic )
corr_dic = {
    "Correlation array": corr_list,
    "corr_list_mean": corr_list_mean,
    "corr_list_std": corr_list_std,
    "RMSE": rmse_mean
}
print("mean", corr_list_mean)
print("std", corr_list_std)
print('MSE mean', np.mean(rmse_mean))

file_name=experiment_dir + "correlation.txt"
with open(file_name, "w") as f:
    for key, value in corr_dic.items():
        f.write(f"{key}: {value}\n")
