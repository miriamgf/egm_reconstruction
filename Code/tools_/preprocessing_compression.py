from scipy import signal
import numpy as np


def preprocess_compression(
    X_1channel, egm_tensor, AF_models, Y_model, fs_sub, batch_size, downsampling=True
):
    """
    This function process loaded data (BSPS, EGMs and metadata) to perform downsampling and
    truncate the length of the arrays according to the specified batch size

    """

    AF_models = np.array(AF_models)

    if downsampling:
        X_1channel_sub = signal.resample_poly(X_1channel, fs_sub, 500, axis=0)
        egm_tensor_sub = signal.resample_poly(egm_tensor, fs_sub, 500, axis=0)
        AF_models_sub = signal.resample_poly(AF_models, fs_sub, 500, axis=0)
        Y_model_sub = signal.resample_poly(Y_model, fs_sub, 500, axis=0)

    X_1channel_sub = truncate_length_by_batch_size(batch_size, X_1channel_sub)
    egm_tensor_sub = truncate_length_by_batch_size(batch_size, egm_tensor_sub)
    AF_models_sub = truncate_length_by_batch_size(batch_size, AF_models_sub)
    Y_model_sub = truncate_length_by_batch_size(batch_size, Y_model_sub)

    return X_1channel_sub, egm_tensor_sub, list(AF_models_sub), Y_model_sub


def truncate_length_by_batch_size(batch_size, signal_data):

    if signal_data.shape[0] % batch_size != 0:
        trunc_val = np.floor_divide(signal_data.shape[0], batch_size)
        signal_data = signal_data[0 : batch_size * trunc_val, ...]
    return signal_data
