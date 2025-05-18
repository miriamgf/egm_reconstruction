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


    
filtered_models = []  # Lista para almacenar los modelos procesados

for model in bsps_64_all_models:
    new_model = []  # Lista para almacenar los nodos filtrados y recortados
    
    for node in model:
        original_length = len(node)  # Longitud original del nodo

        # Descartar los nodos menores de 2000 samples
        if original_length < 2000:
            continue  # No agregar este nodo a new_model

        # Recortar nodos entre 2001 y 2500 samples a 2000
        if 2001 <= original_length <= 2500:
            node = node[:2000]

        # Recortar nodos entre 4001 y 4501 samples a 4000
        elif 4001 <= original_length <= 4501:
            node = node[:4000]

        # Agregar el nodo procesado a la lista del modelo
        new_model.append(node)

    # Solo agregar el modelo si aún tiene nodos después del filtrado
    if len(new_model) > 0:
        filtered_models.append(new_model)
    else:
        continue



bsps_64_mod1 = []
bsps_64_mod2 = []
bsps_64_mod3 = []   
bsps_64_mod4 = []
bsps_64_mod5 = []
bsps_64_mod6 = []
bsps_64_mod7 = []
bsps_64_mod8 = []
bsps_64_mod9 = []


for i in range(len(filtered_models)):
    if i==0:
        bsps_64_mod1.append(filtered_models[i])
    elif i==1:
        bsps_64_mod2.append(filtered_models[i])
    elif i==2:
        bsps_64_mod3.append(filtered_models[i])
    elif i==3:
        bsps_64_mod4.append(filtered_models[i])
    elif i==4:
        bsps_64_mod5.append(filtered_models[i])
    elif i==5:
        bsps_64_mod6.append(filtered_models[i])
    elif i==6:
        bsps_64_mod7.append(filtered_models[i])
    elif i==7:
        bsps_64_mod8.append(filtered_models[i])
    elif i==8:
        bsps_64_mod9.append(filtered_models[i])


bsps_64_filtered_models = filtered_models
print("All models have been trimmed to the minimum length - var: **bsps_64_filtered_models** ")

J = 4
Q = (16,1)
ws_coeff_all_mods = []  # list to store the wavelet scattering coefficients for all models --> 12 tensors of 64 nodes 

models_name = ['Simulation_01_210209_001_002', 'LA_RSPV_CAF_150115', 'Simulation_01_200428_001_001', 'Simulation_01_201223_001_002', 'Simulation_01_190619_001_004', 'Simulation_01_190502_001_004', 'RA_RAFW_140807', 'RA_RAA_141230', 'Simulation_01_200316_001_  1', 'TwoRotors_181219', 'RA_RAFW_SAF_140730', 'Simulation_01_200212_001_  4']
# Aplicar wavelet scattering a los todos los nodos de los 2 modelos (simple y complex)
for model, name_model in zip(bsps_64_filtered_models, models_name):
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
            "Simulation_01_200316_001_5", "Simulation_01_200316_001_  1"
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
