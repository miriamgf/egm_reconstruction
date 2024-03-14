import matplotlib.pyplot as plt
import pickle
from tools_.preprocessing_network import *
from tools_.plots import *
from scripts.config import TrainConfig_1
from scripts.config import TrainConfig_2
from scripts.config import DataConfig
import random
from scripts.evaluate_function import reshape_tensor
from tools_ import freq_phase_analysis as freq_pha


dict_var_dir = '../output/variables/'
dict_results_dir = '../output/results/'
figs_dir = '../output/figures/'


def DF_mapping(y_test, pred_test, BSPM_test, AF_models_test, norm = True ):

    y_test_flat = reshape_tensor(y_test, n_dim_input=y_test.ndim, n_dim_output=2)
    reconstruction_flat_test = reshape_tensor(pred_test, n_dim_input=pred_test.ndim, n_dim_output=2)

    y_test_subsample = y_test_flat
    estimate_egms_test = reconstruction_flat_test
    estimate_egms_n = normalize_by_models(reconstruction_flat_test, BSPM_test)

    # Normalize (por muestras)
    if norm:
        estimate_egms_n = []
        for model in np.unique(BSPM_test):
            # 1. Normalize Reconstruction
            arr_to_norm_estimate = reconstruction_flat_test[
                np.where((BSPM_test == model))]  # select window of signal belonging to model i
            estimate_egms_norm = normalize_array(arr_to_norm_estimate, 1, -1, 0)  # 0: por muestas
            estimate_egms_n.extend(estimate_egms_norm)  # Add to new norm array
        estimate_egms_n = np.array(estimate_egms_n)

    # DF Mapping
    unique_test_models = np.unique(AF_models_test)
    for model in unique_test_models:
        # model= AF_models_test[1]
        estimation_array = estimate_egms_n[
            np.where((AF_models_test == model))]  # select window of signal belonging to model i
        y_array = y_test_subsample[
            np.where((AF_models_test == model))]  # select window of signal belonging to model i

        DF_rec, sig_k_rec, phase_rec = freq_pha.kuklik_DF_phase(estimation_array.T, fs)
        DF_label, sig_k_label, phase_label = freq_pha.kuklik_DF_phase(y_array.T, fs)

        # Interpolate DF Mapping to 2048 nodes

        sig_k_rec_i = interpolate_fun(sig_k_rec.T, len(sig_k_rec.T), 2048, sig=False)
        sig_k_label_i = interpolate_fun(sig_k_label.T, len(sig_k_label.T), 2048, sig=False)
        phase_rec_i = interpolate_fun(sig_k_rec.T, len(phase_rec.T), 2048, sig=False)
        phase_label_i = interpolate_fun(sig_k_label.T, len(phase_label.T), 2048, sig=False)

        # Save DF Variables: DF Maps, phase maps from reconstruction and label (real)
        dic_DF = {"DF_rec": DF_rec, "sig_k_rec": sig_k_rec_i, "phase_rec": phase_rec_i,
                  "DF_label": DF_label, "sig_k_label": sig_k_label_i, "phase_label": phase_label_i}
        savemat(dict_var_dir + "/DF_Mapping_variables_" + str(model) + ".mat", dic_DF)
