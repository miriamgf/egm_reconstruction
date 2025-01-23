import json
import os
import argparse



class ParseHiperparams(object):
    """
    This function loads and parses default params from dictionary and defines search spaces for
    optuna optimization

    """

    def __init__(self):
        self.path = "scripts/src/training/hyperparams/hyperparams.json"

    def get_path(self):
        return self.path

    def parse_default_hyperparams(self):
        with open(self.path, "r") as file:
            data = json.load(file)  # Load the JSON data into a dictionary
        return data

    def parse_optuna_hyperparams(self):
        with open(self.path, "r") as file:
            data = json.load(file)  # Load the JSON data into a dictionary
        optuna_params = data["optuna_parameters"]
        return optuna_params
    
    def load_best_hyperparams(self, custom_path):
        with open(custom_path, "r") as file:
            data = json.load(file)  # Load the JSON data into a dictionary
        return data
    
def str_to_bool(value):
    """Convierte una cadena en booleano"""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', '1'):
        return True
    elif value.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}' instead.")


class TrainConfig_1(object):  # Multioutput

    hyperparams = ParseHiperparams().parse_default_hyperparams()

    learning_rate_1 = hyperparams["learning_rate"]
    batch_size_1 = hyperparams["batch_size"]
    n_epoch_1 = hyperparams["n_epochs"]
    use_generator = hyperparams["use_generator"]
    parallelism = hyperparams["parallelism"]
    inference_pretrained_model = False
    optuna_optimization = hyperparams["optuna_optimization"]


class TrainConfig_2(object):
    learning_rate_2 = 0.00001
    batch_size_2 = 200
    n_epoch_2 = 10


class DataConfig(object):

    hyperparams = ParseHiperparams().parse_default_hyperparams()

    # Obtener la URL y el puerto desde las variables de entorno
    mlflow_url = os.environ.get("MLFLOW_URL")
    mlflow_port = os.environ.get("MLFLOW_PORT")
    fs = hyperparams["fs"]
    fs_sub = hyperparams["fs_sub"]
    SNR = 20
    n_classes = hyperparams[
        "n_classes"
    ]  # this is specific for classification. # 1: Rotor/no rotor ; 2: RA/LA/No rotor (2 classes) ; 3: 7 regions (3 classes) + no rotor (8 classes)
    DF_Mapping = False
    n_nodes_regression = hyperparams["n_nodes_regression"]  # 512, 682, 1024, 2048
