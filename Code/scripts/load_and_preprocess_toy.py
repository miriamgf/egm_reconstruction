# %% Import modules
import sys

sys.path.append("../Code")
sys.path.append("../")
sys.path.append("../tools_")
sys.path.append("../models")
import matplotlib.pyplot as plt
from config import DataConfig, TrainConfig_1

print("importing tools")
import datetime
import os
import time

import mlflow
import tensorflow as tf
from evaluate_function import *
from numpy import *

from tools_.df_mapping import *
from tools_.preprocessing_network import *
from tools_.tools import *

tf.random.set_seed(42)
import datetime
import time

import mlflow
import tensorflow as tf
from tensorflow.keras import backend as K

print("end imports")
# Clear GPU
K.clear_session()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()


# %% Some definitions

# Here it is defined which dataset to use
SNR_em_noise = 1
SNR_white_noise = 20
patches_oclussion = "P1"
experiment_number = 0
unfold_code = 1

experiment_name = (
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_EXP_" + str(experiment_number)
)  # + '_' + str(SNR_white_noise)+ '_' + str(patches_oclussion)+ '_' + str(unfold_code)


root_logdir = "output/logs/"
log_dir = root_logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# data_dir = '/home/profes/miriamgf/tesis/Autoencoders/Data_short/'
data_dir = "../../.../Data_short/"
torsos_dir = "../../../../Labeled_torsos/"
figs_dir = "output/figures/"
models_dir = "output/model/"
dict_var_dir = "output/variables/"
dict_results_dir = "output/results/"
experiment_dir = "output/experiments/experiments_CINC/" + experiment_name + "/"


if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
    print("Directory for experiment", experiment_dir, "created")
else:
    experiment_dir = experiment_dir + "_1"
    print(
        "Existing directory, name changed for avoiding rewriting information to:",
        experiment_dir,
    )

dic_vars = {}
dict_results = {}

# GPU Configuration
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs:", len(physical_devices))
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


start = time.time()

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

# mlflow for tracking experiments: we will
mlflow.set_tracking_uri(uri="http://10.110.100.78:5000")
mlflow.autolog()

# %% Load data

if DataConfig.fs == DataConfig.fs_sub:
    DataConfig.fs = DataConfig.fs_sub

Transfer_model = False  # Transfer learning from sinusoids
sinusoids = False


print("Loading dATA")
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
) = load_data(
    directory=data_dir,
    data_type="1channelTensor",
    n_classes=DataConfig.n_classes,
    subsampling=True,
    fs_sub=DataConfig.fs_sub,
    norm=False,
    SR=True,
    SNR=DataConfig.SNR,
    n_batch=TrainConfig_1.batch_size_1,
    sinusoid=sinusoids,
    SNR_em_noise=SNR_em_noise,
    SNR_white_noise=SNR_em_noise,
    patches_oclussion=patches_oclussion,
    unfold_code=unfold_code,
    inference=False,
)

plt.figure()
plt.plot(X_1channel[0:200, 0, 0], label="bsps")
plt.plot(egm_tensor[0:200, 0], label="egm")
plt.legend()
plt.savefig("output/figures/input_output/before_norm.png")

# Normalize BSPS and EGM
X_1channel = normalize_by_models(X_1channel, Y_model)
egm_tensor = normalize_by_models(egm_tensor, Y_model)
X_1channel = np.nan_to_num(X_1channel, nan=0.0)

plt.figure()
plt.plot(X_1channel[0:200, 0, 0], label="bsps")
plt.plot(egm_tensor[0:200, 0], label="egm")
plt.legend()
plt.savefig("output/figures/input_output/norm.png")


# Train/Test/Val Split
random_split = True
print("Splitting...")
(
    x_train,
    x_test,
    x_val,
    train_models,
    test_models,
    val_models,
    AF_models_train,
    AF_models_test,
    AF_models_val,
    BSPM_train,
    BSPM_test,
    BSPM_val,
) = train_test_val_split_Autoencoder(
    X_1channel,
    AF_models,
    Y_model,
    all_model_names,
    random_split=True,
    train_percentage=0.90,
    test_percentage=0.2,
    deterministic=True,
)


print("TRAIN SHAPE:", x_train.shape, "models:", train_models)
print("TEST SHAPE:", x_test.shape, "models:", test_models)
print("VAL SHAPE:", x_val.shape, "models:", val_models)

x_train, x_test, x_val = preprocessing_autoencoder_input(
    x_train, x_test, x_val, TrainConfig_1.batch_size_1
)

y_train, y_test, y_val = preprocessing_y(
    egm_tensor,
    Y_model,
    AF_models,
    train_models,
    test_models,
    val_models,
    TrainConfig_1.batch_size_1,
    norm=False,
)
