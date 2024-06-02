import sys

sys.path.append("../Code")
from tools_tikhonov import load_data
import forward_inverse_problem as fip
import filtering

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
from tools_.tools import corr_spearman_cols
from tools_.tools import array_to_dic_by_models


directory = "/home/profes/miriamgf/tesis/Autoencoders/Data/"
fs = 500

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
print(all_model_names)
print(len(all_model_names), "Models")


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
) = load_data(
    data_type="Flat",
    n_classes=n_classes,
    subsampling=True,
    fs_sub=fs,
    norm=False,
    SR=True,
    SNR=20,
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

for model in range(1, len(all_model_names)):

    try:

        print("Calculando modelo", all_model_names[model])
        pos_model = np.where(np.array(AF_models) == model)
        pos_unique = np.unique(Y_model[pos_model])[0]  # Select only the first torso
        y = np.array(X_1channel[np.where(Y_model == pos_unique)])

        # y = np.array(X_1channel[np.where(Y_model==model)])
        x = np.array(egm_tensor[np.where(Y_model == pos_unique)])
        y = np.vstack(y).T

        A = np.array(transfer_matrices[0][0])  # Only one torso
        AA, L, LL = pre_m.precompute_matrix(A, atrial_model, order)
        # Classical Tikhonov-based inverse problem approach.
        # x_hat_tikh, lambda_opt_tikh = classical_tikhonov_noiter(A, AA, L, LL, y, 50)
        # x_hat,lambda_opt_list,magnitude_terms_list,error_terms_list,max_lcurve_list = fip.classical_tikhonov_noiter(A, AA, L, LL, y, 500)
        x_hat, lambda_opt, magnitude_term, error_term, maxcurve_index = (
            fip.classical_tikhonov_noiter_global(A, AA, L, LL, y)
        )
        DF_tikh, sig_k_tikh, phase_tikh = freq_pha.kuklik_DF_phase(x_hat, 100)

        # Classical Tikhonov-based inverse problem metrics
        # RDMSt_tikh, mRDMSt_tikh, stdRDMSt_tikh = metrics.RDMS_calc(x,x_hat)
        # CCt_tikh, mCCt_tikh, stdCCt_tikh = metrics.CC_calc(x,x_hat)
        x_hat = x_hat.T

        x_hat = np.array(x_hat)
        x_array = np.array(x)

        reconstruction_list.append(x_hat)
        x_real_list.append(x_array)

        correlation = corr_spearman_cols(x_hat, x_array)
        corr_list.append(correlation)
        corr_mean = np.mean(correlation)
        corr_std = np.std(correlation)
        corr_list_mean.append(corr_mean)
        corr_list_std.append(corr_std)

        plt.figure()
        plt.plot(x_hat[0:1000, 0], label="rec")
        plt.plot(x_array[0:1000, 0], label="real")
        plt.legend()
        plt.savefig("scripts/Tikhonov/figures/" + str(model) + str(".png"))
        plt.show()

        model_name = all_model_names[model]
        key = f"model{model_name}"

        new_dic[key] = {
            "reconstruction": x_hat.tolist(),  # Convert rec to a list if necessary
            "label": x_array.tolist(),  # Convert lab to a list if necessary
        }

        model_name

    except:
        print(model, "EXCLUDED")


print(new_dic)
savemat("scripts/Tikhonov/figures/tikhonov_matlab.mat", new_dic)
corr_dic = {
    "Correlation array": corr_list,
    "corr_list_mean": corr_list_mean,
    "corr_list_std": corr_list_std,
}
print("mean", corr_list_mean)
print("std", corr_list_std)
with open("results/correlation.txt", "w") as f:
    for key, value in corr_dic.items():
        f.write(f"{key}: {value}\n")
