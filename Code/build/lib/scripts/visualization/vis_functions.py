import scipy.io
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import sys

sys.path.append("../Code")
from tools_.tools import *


def load_signals(type, experiment="20240530-141019", p=None):
    if type == "Tik":
        if p == None:
            p = "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/Tikhonov/tikhonov_matlab.mat"
        dic = scipy.io.loadmat(p)
        fs = 500

    elif type == "DL":
        if p == None:
            p = (
                "/home/profes/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_CINC/"
                + str(experiment)
                + "/reconstructions_by_model_"
                + str(experiment)
                + ".mat"
            )
        dic = scipy.io.loadmat(p)
        fs = get_fs_from_experiment(experiment=experiment)
    return dic, fs


def get_fs_from_experiment(experiment="20240530-141019"):

    p = (
        "/home/profes/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_CINC/"
        + str(experiment)
        + "/"
    )

    # Abrir y leer el archivo hiperparams.txt
    with open(p + "hyperparams.txt", "r") as file:
        lines = file.readlines()

    # Inicializar el valor de fs
    fs_value = None

    # Recorrer las líneas para encontrar el valor de fs
    for line in lines:
        if line.startswith("fs:"):
            # Dividir la línea en dos partes separadas por ':'
            parts = line.split(":")
            # Eliminar espacios en blanco y convertir el valor a entero
            fs_value = int(parts[1].strip())
            break

    return fs_value


# Divide by torso


def select_torso(signal, torso, n_torsos=10):
    """
    This function selects torso to represent from the input signal

    """
    split_signal = np.array_split(signal, n_torsos)
    selected_torso = split_signal[torso]

    return selected_torso


def plot_1D_signals(
    tikhonov_signal,
    tikhonov_label,
    dl_signal,
    dl_label,
    node=0,
    torso=None,
    n_samples=500,
    normalize_tik=False,
    filter_tik_label=False,
):
    """
    This function plots the specified model comparing DL and Tikhonov reconstructions.
    Its is possible to select:
     * the torso desired to view
     * the number of samples desired to view
     * the model desired to view
     * the node of the atria to be represented

    """

    if filter_tik_label:
        tikhonov_label = ECG_filtering(tikhonov_label, fs=500)

    if torso != None:
        dl_signal = select_torso(dl_signal, torso, n_torsos=10)
        dl_label = select_torso(dl_label, torso, n_torsos=10)
        n_samples = len(dl_signal)

    if normalize_tik:
        tikhonov_signal = normalize_array(tikhonov_signal, 1, -1, axis_n=0)
        tikhonov_label = normalize_array(tikhonov_label, 1, -1, axis_n=0)
        add_title = " (normalized)"
    else:
        add_title = ""

    plt.figure(figsize=(18, 7), tight_layout=True)
    plt.subplot(2, 1, 1)
    plt.plot(tikhonov_signal[0:n_samples, node], label="tikhonov reconstruction")
    plt.plot(tikhonov_label[0:n_samples, node], label="tikhonov label", linewidth=0.75)
    plt.ylabel("Voltage")
    plt.xlabel("Samples")
    plt.title("Reconstruction with Tikhonov" + add_title)

    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(dl_signal[0:n_samples, node], label="dl reconstruction")
    plt.plot(dl_label[0:n_samples, node], label="dl label", linewidth=0.75)
    plt.ylabel("Voltage")
    plt.xlabel("Samples")
    plt.legend()
    plt.title("Reconstruction with Deep Learning (normalized)")
    plt.show()


def normalize_array(array, high, low, axis_n=0):
    """
    This functions normalized a 2D 'array' along axis 'axis_n' and between the values 'high' and 'low'

    To normalize a full signal, indicate the index dimension

    """
    mins = np.min(array, axis=axis_n)
    maxs = np.max(array, axis=axis_n)
    rng = maxs - mins
    # if axis_n==1:
    # array=array.T
    norm_array = high - (((high - low) * (maxs - array)) / rng)
    # if axis_n==1:
    # norm_array=norm_array.T
    return norm_array


def plot_correlation_by_nodes(
    range_of_models=None, smoothing=None, type_corr="Pearson", experiment=None
):
    """
    This function plots the correlation of each node between real and reconstructed signals
    in the range of models introduced
    """

    # Compute correlations for each model
    correlations_tk = []
    correlations_dl = []

    for model_i in range_of_models:
        # Load signals of that model
        tikhonov, fs_tk = load_signals("Tik")
        dl, fs_dl = load_signals("DL", experiment=experiment)
        dl_label = dl[model_i]["label"][0][0]
        dl_signal = dl[model_i]["reconstruction"][0][0]
        tikhonov_label = tikhonov[model_i]["label"][0][0]
        tikhonov_signal = tikhonov[model_i]["reconstruction"][0][0]

        if type_corr == "Pearson":

            corr_i_tk = corr_pearson_cols(tikhonov_signal, tikhonov_label)
            corr_i_dl = corr_pearson_cols(dl_signal, dl_label)

        elif type_corr == "Spearman":

            corr_i_tk = corr_spearman_cols(tikhonov_signal, tikhonov_label)
            corr_i_dl = corr_spearman_cols(dl_signal, dl_label)

        if smoothing != None:
            corr_i_tk = moving_average(corr_i_tk, window_size=smoothing)
            corr_i_dl = moving_average(corr_i_dl, window_size=smoothing)

        correlations_tk.append(corr_i_tk)
        correlations_dl.append(corr_i_dl)

    plt.figure(figsize=(18, 7), tight_layout=True)
    plt.subplot(2, 1, 1)
    for model_i in range(0, len(range_of_models)):
        plt.plot(correlations_tk[model_i], label=range_of_models[model_i])
    plt.xlabel("Nodes")
    plt.ylabel(type_corr + "Correlation")
    plt.title("Tikhonov")
    plt.legend()
    plt.subplot(2, 1, 2)
    for model_i in range(0, len(range_of_models)):
        plt.plot(correlations_dl[model_i], label=range_of_models[model_i])
    plt.title("DL")
    plt.xlabel("Nodes")
    plt.ylabel(type_corr + "Correlation")
    plt.legend()
    plt.show()


def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode="valid")


def plot_correlation_by_models(
    range_of_models=None,
    type_corr="Pearson",
    separation=0.3,
    rotation=45,
    experiment=None,
):
    """
    This function plots a boxplot representing the mean correlation between real and reconstructed signals
    for all the nodes of each model in the range of models introduced
    """
    # Compute correlations for each model
    correlations_tk = []
    correlations_dl = []

    for model_i in range_of_models:
        # Load signals of that model
        tikhonov, fs_tk = load_signals("Tik")
        dl, fs_dl = load_signals("DL", experiment=experiment)
        dl_label = dl[model_i]["label"][0][0]
        dl_signal = dl[model_i]["reconstruction"][0][0]
        tikhonov_label = tikhonov[model_i]["label"][0][0]
        tikhonov_signal = tikhonov[model_i]["reconstruction"][0][0]

        if type_corr == "Pearson":

            corr_i_tk = corr_pearson_cols(tikhonov_signal, tikhonov_label)
            corr_i_dl = corr_pearson_cols(dl_signal, dl_label)

        elif type_corr == "Spearman":

            corr_i_tk = corr_spearman_cols(tikhonov_signal, tikhonov_label)
            corr_i_dl = corr_spearman_cols(dl_signal, dl_label)

        correlations_tk.append(corr_i_tk)
        correlations_dl.append(corr_i_dl)

    # Boxplot 2

    data_tk = correlations_tk
    data_dl = correlations_dl
    renamed_models = list(range(1, len(range_of_models)))
    positions_tk = [i * 2 + 1 for i in range(len(data_tk))]
    boxplots_tk = plt.boxplot(
        data_tk,
        positions=positions_tk,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
    )

    positions_dl = [i * 2 + 1 + separation for i in range(len(data_dl))]
    boxplots_dl = plt.boxplot(
        data_dl,
        positions=positions_dl,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen"),
    )

    plt.xticks(
        ticks=[i * 2 + 1 + separation / 2 for i in range(len(renamed_models))],
        labels=renamed_models,
        rotation=rotation,
    )
    plt.xlabel("Model")
    plt.ylabel("Correlation Values")
    plt.title(
        "Correlation of AF Models. Compared performance in Tikhonov and DL reconstructions"
    )
    # Añadir la leyenda manualmente
    plt.legend(
        [boxplots_tk["boxes"][0], boxplots_dl["boxes"][0]],
        ["Tikhonov", "DL"],
        loc="best",
    )
    plt.tight_layout()
    plt.show()
