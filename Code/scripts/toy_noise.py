# %%
import sys

sys.path.append("../Code")
import datetime

import tensorflow as tf
from numpy import *

from tools_.df_mapping import *
from tools_.noise_simulation import NoiseSimulation
from tools_.preprocessing_network import *
from tools_.tools import *

tf.random.set_seed(42)

experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

root_logdir = "output/logs/"
log_dir = root_logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = "../../../../Data_short/"
torsos_dir = "../../../../Labeled_torsos/"
figs_dir = "output/figures/"
models_dir = "output/model/"
dict_var_dir = "output/variables/"
dict_results_dir = "output/results/"
experiment_dir = "output/experiments/" + experiment_name + "/"


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
    data_type="1channelTensor",
    n_classes=DataConfig.n_classes,
    subsampling=True,
    fs_sub=DataConfig.fs_sub,
    norm=False,
    SR=True,
    SNR=DataConfig.SNR,
    n_batch=TrainConfig_1.batch_size_1,
    sinusoid=False,
    data_dir=data_dir,
)


Noise_Simulation = NoiseSimulation(
    SNR_em_noise=20, SNR_white_noise=20, oclusion=None, fs=DataConfig.fs_sub
)

em = Noise_Simulation.load_physionet_signals(type_noise="em")

noise_database = Noise_Simulation.configure_noise_database(
    X_1channel, all_model_names, em=True, ma=False, gn=True, noise_augmentation=4
)

noise_database

Noise_Simulation.add_noise(X_1channel, 0, noise_database, distribution_noise_mode=2)
