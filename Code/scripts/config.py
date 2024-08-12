import os


class TrainConfig_1(object):
    learning_rate_1 = 0.0001
    batch_size_1 = 400
    n_epoch_1 = 100


class TrainConfig_2(object):
    learning_rate_2 = 0.0001
    batch_size_2 = 400
    n_epoch_2 = 100


class DataConfig(object):
    # Obtener la URL y el puerto desde las variables de entorno
    mlflow_url = os.environ.get("MLFLOW_URL")
    mlflow_port = os.environ.get("MLFLOW_PORT")
    fs = 500
    fs_sub = 200
    SNR = 20
    n_classes = 3  # this is specific for classification. # 1: Rotor/no rotor ; 2: RA/LA/No rotor (2 classes) ; 3: 7 regions (3 classes) + no rotor (8 classes)
    DF_Mapping = False
