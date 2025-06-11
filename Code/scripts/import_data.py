import sys
sys.path.append('../tools_')

#from tools import *
import sys, os
from scipy.io import loadmat
import numpy as np


from tools import sinusoids_generator
from tools import load_egms
from tools import load_egms_df
from tools import ECG_filtering  
from tools import load_transfer
from tools import forward_problem
from tools import addwhitenoise
#from tools import add_white_noise

#CHARGING EGMS DATA 
directory = "../../Data_short/"
#data_short_dir = "/home/alumnos/mburgosc/tfg/egm_reconstruction/Data_short"
data_short_dir = "/home/alumnos/mburgosc/tfg/egm_reconstruction/DATA_USE"
data_short_dir = "/home/alumnos/mburgosc/tfg/egm_reconstruction/Data"


egms_all_models = load_egms_df(data_short_dir)
egms_values = egms_all_models["AF_signal"].tolist()
print("models_names" , egms_all_models["id"].tolist())
models_names = egms_all_models["id"].tolist()
print("Egms models Names LOADED - var: **models_names**")

print(len(egms_values[1][2])) # 2048 samples of time
print("how many models", len(egms_values)) # 64 nodes
print()
print("Egms models LOADED - var: **egms_all_models**, **egms_values** ")

# DE MOMENTO QUITAR FILTRADO Y RUIDO
# noisy_egms = []
# for i in range(len(egms_values)):
#     egms = egms_values[i]
#     egms_noisy, _ = addwhitenoise(egms.T, SNR = 20, fs=500)
#     egms_noisy = egms_noisy.T 
#     print(egms_noisy.shape)
#     noisy_egms.append(egms_noisy)
# print(len(noisy_egms))

#ECG FILTERING AND NORMALIZATION
egms_filtered_norm = []
filtered_model_names = []

for i in range(len(egms_values)):
#for i in range(len(noisy_egms)):
    #ECG FILTERING
    #egms = noisy_egms[i]
    egms = egms_values[i]
    #egms_filt = ECG_filtering(egms.T, 500, order = 2, f_low=3, f_high=60)
    if egms.shape[1] < 2000:
        print(f"Descartando modelo {i} por tener menos de 2000 muestras: {egms.shape[1]}")
        continue  # Saltar este modelo

    filtered_model_names.append(models_names[i])

    print("Original EGM shape:", egms.shape)        # Debe ser (n_nodes, time)
    print("Transposed for filtering:", egms.T.shape)  # Debe ser (time, n_nodes)

    egms = egms.T
    #NORMLIZE EGMS
    high = 1
    low = -1
    mins = np.min(egms, axis=0)
    maxs = np.max(egms, axis=0)

    #mins = np.min(egms_filt, axis=0)
    #maxs = np.max(egms_filt, axis=0)
    rng = maxs - mins

    norm_egms = high - (((high - low) * (maxs - egms)) / rng)
    #norm_egms = high - (((high - low) * (maxs - egms_filt)) / rng)
    
    norm_egms = norm_egms.T #para que queden otra vez (2048 x samples of time)
    egms_filtered_norm.append(norm_egms)
    print(norm_egms.shape)


print("EGMS data normalized - var: **egms_filtered_norm** ")

import random
import matplotlib
import matplotlib.pyplot as plt

node = 100
random_indices = random.sample(range(len(egms_values)), 3)
fig, axs = plt.subplots(3, 1, figsize=(15, 10))

for i, idx in enumerate(random_indices):
    axs[i].plot(egms_values[idx][node])
    axs[i].set_title(f'Original EGM Model {idx}')
    axs[i].set_xlabel('Samples of time')
    axs[i].set_ylabel('Amplitude')

plt.tight_layout()
plt.savefig('original_models_subplot.png')  # Guarda la imagen con 3 subplots
plt.close()


random_ind = random.sample(range(len(egms_filtered_norm)), 3)
fig, axs = plt.subplots(3, 1, figsize=(15, 10))

for i, idx in enumerate(random_ind):
    axs[i].plot(egms_filtered_norm[idx][node, :]) 
    axs[i].set_title(f'Filtered EGM Model {idx}')  
    axs[i].set_xlabel('Samples of time')
    axs[i].set_ylabel('Amplitude')

plt.tight_layout()
plt.savefig('filtered_models_subplot.png')
plt.show()


# LOADING TRANSFER MATRICES AND COMPUTING THE FORWARD PROBLEM
transfer_matrices = load_transfer(ten_leads=False, bsps_set=False)
print(len(transfer_matrices))
#A = transfer_matrices[0]


bsps_all_models = []
bsps_64_all_models = []
bsps_64_all_models_names = []

for i in range(len(egms_filtered_norm)):
    for matrix in transfer_matrices:

        y_f = forward_problem(egms_filtered_norm[i], matrix[0]) # bsps
        bsps_all_models.append(y_f)
        bsps_64 = y_f[matrix[1].ravel(),:]
        bsps_64_all_models.append(bsps_64)
        bsps_64_all_models_names.append(filtered_model_names[i])

print("Transfer matrices LOADED - var: **transfer_matrices** ")
print("BSPS Models LOADED - var: **bsps_all_models** ")
print("BSPS 64 Models LOADED - var: **bsps_64_all_models** ")

print(len(bsps_all_models))
print(bsps_all_models[0].shape)

# EXTRACTING 64 NODES FROM THE BSPS SIGNALS 
# bsps_64_all_models = []
# for i in range(len(bsps_all_models)):
#
#         bsps_64 = bsps_all_models[i][A[1].ravel(),:]
#         bsps_64_all_models.append(bsps_64)

print(bsps_all_models[0].shape)
print(bsps_64_all_models[0].shape)
print(bsps_all_models[1].shape)
print(bsps_64_all_models[1].shape)
print(bsps_64_all_models[11].shape)
print(len(bsps_64_all_models))
print("BSPS 64 Models LOADED - var: **bsps_64_all_models** ")
