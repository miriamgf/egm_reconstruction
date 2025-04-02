import tensorflow as tf 
import numpy as np
gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(gpus))

import import_data

egms_all_models = import_data.egms_all_models
egms_values = import_data.egms_values
egms_filtered_norm = import_data.egms_filtered_norm
bsps_all_models = import_data.bsps_all_models
bsps_64_all_models = import_data.bsps_64_all_models
transfer_matrices = import_data.transfer_matrices

import kymatio
from kymatio import Scattering1D


for i in range(len(bsps_all_models)):
    bsps_64 = bsps_all_models[i]


       
min_length = float('inf')  

# We look for the minimum length of the nodes to trimmed all the models
for model in bsps_64_all_models:  
    for node in model:  
        node_length = len(node)  
        min_length = min(min_length, node_length) 

trimmed_models = [
    [node[:min_length] for node in model]  
    for model in bsps_64_all_models
]


bsps_64_mod1 = []
bsps_64_mod2 = []
bsps_64_mod3 = []   
bsps_64_mod4 = []
bsps_64_mod5 = []
bsps_64_mod6 = []
bsps_64_mod7 = []
bsps_64_mod8 = []
bsps_64_mod9 = []
bsps_64_mod10 = []
bsps_64_mod11 = []
bsps_64_mod12 = []

for i in range(len(trimmed_models)):
    if i==0:
        bsps_64_mod1.append(trimmed_models[i])
    elif i==1:
        bsps_64_mod2.append(trimmed_models[i])
    elif i==2:
        bsps_64_mod3.append(trimmed_models[i])
    elif i==3:
        bsps_64_mod4.append(trimmed_models[i])
    elif i==4:
        bsps_64_mod5.append(trimmed_models[i])
    elif i==5:
        bsps_64_mod6.append(trimmed_models[i])
    elif i==6:
        bsps_64_mod7.append(trimmed_models[i])
    elif i==7:
        bsps_64_mod8.append(trimmed_models[i])
    elif i==8:
        bsps_64_mod9.append(trimmed_models[i])
    elif i==9:
        bsps_64_mod10.append(trimmed_models[i])
    elif i==10:
        bsps_64_mod11.append(trimmed_models[i])
    elif i==11:
        bsps_64_mod12.append(trimmed_models[i])

bsps_64_models_trimmed = trimmed_models
print("All models have been trimmed to the minimum length - var: **bsps_64_models_trimmed** ")

J = 4
Q = (16,1)
ws_coeff_all_mods = []  # list to store the wavelet scattering coefficients for all models --> 12 tensors of 64 nodes 

models_name = ['Simulation_01_210209_001_002', 'LA_RSPV_CAF_150115', 'Simulation_01_200428_001_001', 'Simulation_01_201223_001_002', 'Simulation_01_190619_001_004', 'Simulation_01_190502_001_004', 'RA_RAFW_140807', 'RA_RAA_141230', 'Simulation_01_200316_001_  1', 'TwoRotors_181219', 'RA_RAFW_SAF_140730', 'Simulation_01_200212_001_  4']
# Aplicar wavelet scattering a los todos los nodos de los 2 modelos (simple y complex)
for model, name_model in zip(bsps_64_models_trimmed, models_name):
    print(f"\nModel: {name_model}")
    print("Samples of every node:", len(model[0]), "samples")

    model_coeffs = []  # Lista temporal para guardar los coeficientes de los 64 nodos
    
    for j in range(len(model)):
        signal = np.array(model[j])  # Asegurar que es un array de numpy
        len_signal = len(signal)
        
        scattering = Scattering1D(J=J, shape=(len_signal,), Q=Q)
        Sx = scattering(signal)  # Aplicar wavelet scattering
        
        model_coeffs.append(Sx)  # Guardar el coeficiente de este nodo
    
    ws_coeff_all_mods.append(np.stack(model_coeffs))  # Convertir en tensor y añadirlo a la lista principal
    print(f"Shape del tensor de {name_model}: {ws_coeff_all_mods[-1].shape}")

print("Wavelet scattering coefficients for all models (12 tensors) - var: **ws_coeff_all_mods** ")

#self.sinusal_groups
sinusal_groups = {
            "Simulation_01_200316_001_7", "Simulation_01_210209_001_002",
            "Simulation_01_210205_001_002", "Simulation_01_200428_001_001",
            "Simulation_01_200316_001_5", "Simulation_01_200428_001_005",
            "Simulation_01_200212_001_7", "Simulation_01_210119_001_001",
            "Simulation_01_210205_001_003", "Simulation_01_201223_001_002",
            "Simulation_01_200212_001_1", "Simulation_01_200428_001_003",
            "Simulation_01_200316_001_3", "Simulation_01_200316_001_4",
            "Sinusal_150629", "RA_RAA_141216", "Simulation_01_200428_001_002",
            "Simulation_01_210210_001_001", "Simulation_01_210208_001_002",
            "Simulation_01_200212_001_9", "Simulation_01_200316_001_1",
            "Simulation_01_200212_001_6", "Simulation_01_200212_001_5",
            "Simulation_01_210209_001_003", "Simulation_01_200316_001_9",
            "Simulation_01_210209_001_001", "Simulation_01_200428_001_006",
            "Simulation_01_200316_001_5",
        }
 
rotor_simple_groups = {
            "Simulation_01_200212_001_2", "Simulation_01_191001_001_002",
            "Simulation_01_190717_001_001", "Simulation_01_191001_001_007",
            "Simulation_01_200316_001_8", "Simulation_01_200428_001_004",
            "Simulation_01_200212_001_8", "Simulation_01_200212_001_10",
            "Simulation_01_191001_001_001", "Simulation_01_200428_001_007",
            "Simulation_01_200428_001_009", "Simulation_01_200428_001_010",
            "Simulation_01_200316_001_2", "Simulation_01_200316_001_6",
            "Simulation_01_200316_001_10", "Simulation_01_190502_001_003",
            "Simulation_01_190619_001_002", "Simulation_01_191001_001_005",
            "Simulation_01_200212_001_4", "Simulation_01_200212_001_10",
            "Simulation_01_200428_001_008", "Simulation_01_190502_001_005"
        }
 
rotor_complejo_groups = {
            "LA_RSPV_CAF_150115", "LA_RSPV_150113", "LA_PLAW_140612",
            "LA_LSPV_150203", "LA_LSPV_150113", "RA_RAFW_140807",
            "RA_RAA_141230", "TwoRotors_181219", "LA_PLAW_140711_arm",
            "RA_RAFW_SAF_140730", "LA_RIPV_150121", "LA_LIPV_150119"
        }
 
print("List of types of models for labeling - var: **sinusal_groups, rotor_simple_groups, rotor_complejo_groups** ")
