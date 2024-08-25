import os
import argparse
from tensorflow.keras import backend as K
import mlflow
import tensorflow as tf
import datetime
import time


class TrainConfig_1(object): #Multioutput
    learning_rate_1 = 0.00001
    batch_size_1 = 400
    n_epoch_1 = 100
    use_generator = False 
    parallelism = False

class TrainConfig_2(object):
    learning_rate_2 = 0.00001
    batch_size_2 = 200
    n_epoch_2 = 10


class DataConfig(object):
    # Obtener la URL y el puerto desde las variables de entorno
    mlflow_url = os.environ.get("MLFLOW_URL")
    mlflow_port = os.environ.get("MLFLOW_PORT")
    fs = 500
    fs_sub = 100
    SNR = 20
    n_classes = 3  # this is specific for classification. # 1: Rotor/no rotor ; 2: RA/LA/No rotor (2 classes) ; 3: 7 regions (3 classes) + no rotor (8 classes)
    DF_Mapping = False
    n_nodes_regression = 682 #512, 682, 1024, 2048


