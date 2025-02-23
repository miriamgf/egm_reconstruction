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

#CHARGING EGMS DATA 
directory = "../../Data_short/"
data_short_dir = "/home/alumnos/mburgosc/tfg/egm_reconstruction/Data_short"
egms_all_models = load_egms_df(data_short_dir)
egms_values = egms_all_models["AF_signal"].tolist()

print()
print("Egms models LOADED - var: **egms_all_models**, **egms_values** ")

#ECG FILTERING AND NORMALIZATION
egms_filtered_norm = []

for i in range(len(egms_values)):
    #ECG FILTERING
    egms = egms_values[i]
    egms_filt = ECG_filtering(egms.T, 500, order = 2, f_low=3, f_high=60)
    
    #NORMLIZE EGMS
    high = 1
    low = -1

    mins = np.min(egms_filt, axis=0)
    maxs = np.max(egms_filt, axis=0)
    rng = maxs - mins

    norm_egms = high - (((high - low) * (maxs - egms_filt)) / rng)
    
    norm_egms = norm_egms.T #para que queden otra vez (2048 x samples of time)
    egms_filtered_norm.append(norm_egms)

print("EGMS data normalized - var: **egms_filtered_norm** ")

# LOADING TRANSFER MATRICES AND COMPUTING THE FORWARD PROBLEM
transfer_matrices = load_transfer(ten_leads=False, bsps_set=False)
A = transfer_matrices[0]

bsps_all_models = []
for i in range(len(egms_filtered_norm)):

    y_f = forward_problem(egms_filtered_norm[i], A[0]) # bsps
    bsps_all_models.append(y_f)

print("Transfer matrices LOADED - var: **transfer_matrices** ")
print("BSPS Models LOADED - var: **bsps_all_models** ")

# EXTRACTING 64 NODES FROM THE BSPS SIGNALS 
bsps_64_all_models = []
for i in range(len(bsps_all_models)):

    bsps_64 = bsps_all_models[i][A[1].ravel(),:]
    bsps_64_all_models.append(bsps_64)

print("BSPS 64 Models LOADED - var: **bsps_64_all_models** ")
