import numpy as np
import h5py
from scipy.io import loadmat
from scipy import signal as sigproc
from scipy import signal
import sys, os
import scipy
from scipy.interpolate import interp1d
from tools_.tools import truncate_length_bsps, interpolate_2D_array


from numpy import reshape

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


directory = "/home/pdi/miriamgf/tesis/Autoencoders/Real_data/"


def load_EGM(
    SR=True,
    subsampling=True,
    fs=977,
    fs_sub=100,
    n_nodes_regression=512,
    batch_size=400,
):

    all_model_names = []
    for subdir, dirs, files in os.walk(directory):
        if subdir != directory:
            model_name = subdir.split("/")[-1]
            if "Sinusal" in model_name and SR == False:
                continue
            else:
                all_model_names.append(model_name)
    # 1) Load EGM
    for model_name in all_model_names:

        egms = load_egms(model_name)
        x = ECG_filtering(egms, fs)
        x = x[:, 3000:]
        if subsampling:
            x_sub = signal.resample_poly(x, fs_sub, fs, axis=1)
            x = x_sub
        x = x.T
        # Interpolate nodes to get 512
        x_original = np.linspace(0, 1, 62)  # Eje original con 62 puntos
        x_new = np.linspace(0, 1, n_nodes_regression)  # Eje nuevo con 512 puntos
        egm_interpolated = np.zeros(
            (x.shape[0], n_nodes_regression)
        )  # Inicializa la matriz interpolada

        for i in range(x.shape[0]):
            f = interp1d(x_original, x[i, :], kind="linear")
            egm_interpolated[i, :] = f(x_new)  # Interpolación en el nuevo eje

        _, _, x = truncate_length_bsps(
            batch_size,
            np.zeros(shape=(egm_interpolated.shape[0], 1, 1)),
            [],
            egm_interpolated.T,
        )

        # x= egm_interpolated

    return x.T


def load_BSPS(subsampling=True, fs=977, fs_sub=100, batch_size=200):

    all_model_names = []
    for subdir, dirs, files in os.walk(directory):
        if subdir != directory:
            model_name = subdir.split("/")[-1]
            all_model_names.append(model_name)

    # 1) Load EGM
    for model_name in all_model_names:
        bsps = load_ecgs(model_name)
        bsps = bsps[:, 3000:]
        bsps_64_filt = ECG_filtering(bsps, fs)

    # RESAMPLING signal to fs= fs_sub
    if subsampling:
        bsps_64 = signal.resample_poly(bsps_64_filt, fs_sub, fs, axis=1)

    else:
        bsps_64 = bsps_64_filt

    tensors_model = get_tensor_model(bsps_64, tensor_type="1channel")
    tensors_model, _, _ = truncate_length_bsps(
        batch_size, tensors_model, [], tensors_model
    )
    tensors_model = interpolate_2D_array(tensors_model)

    return tensors_model


def remove_mean(signal):
    """
    Remove mean from signal

    Parameters:
        signal (array): signal to process

    Returns:
        signotmean: signal with its mean removed
    """
    signotmean = np.zeros(signal.shape)
    for index in range(0, signal.shape[0]):
        signotmean[index, :] = sigproc.detrend(signal[index, :], type="constant")
    return signotmean


def ECG_filtering(signal, fs, order=2, f_low=3, f_high=30):
    """
    Frequency filtering of ECG-EGM.
    SR model: low-pass filtering, 4th-order Butterworth filter.
    FA models: bandpass filtering, 4th-order Butterworth filter.

    Parameters:
        signal (array): signal to process
        fs (int): sampling rate
        f_low (int-float): low cut-off frecuency (default=3Hz)
        f_high (int-float): high cut-off frecuency (default=30Hz)
        model (string): FA model to assess (default: SR)
    Returns:
        proc_ECG_EGM (array): filtered ECG-EGM
    """

    # Remove DC component
    sig_temp = remove_mean(signal)
    # sig_temp = signal

    # Bandpass filtering
    b, a = sigproc.butter(
        order, [f_low / round((fs / 2)), f_high / round((fs / 2))], btype="bandpass"
    )

    proc_ECG_EGM = np.zeros(sig_temp.shape)
    if sig_temp.ndim == 3:
        for i in range(sig_temp.shape[1]):
            for j in range(sig_temp.shape[2]):
                # for index in range(sig_temp.shape[0]):
                proc_ECG_EGM[:, i, j] = sigproc.filtfilt(b, a, sig_temp[:, i, j])
    else:
        for index in range(0, sig_temp.shape[0]):
            proc_ECG_EGM[index, :] = sigproc.filtfilt(b, a, sig_temp[index, :])

    return proc_ECG_EGM


def get_tensor_model(bsps_64, tensor_type="3channel", unfold_code=1):
    """
    Get X (tensor) from one model

    Parameters:
      bsps_64: 64 x n_time matrix bsps for 1 model
      y_model: array of y labels from 1 model

    Return:
      all_tensors: array of all tnesor from 1 model
    """
    all_tensors = np.array([])

    for t in range(0, bsps_64.shape[1]):
        if tensor_type == "3channel":
            tensor_model = get_subtensor_54(bsps_64[:, t], tensor_type)
        else:
            tensor_model = get_subtensor_54(bsps_64[:, t], tensor_type, unfold_code=1)
        if all_tensors.size == 0:
            all_tensors = tensor_model
        else:
            all_tensors = np.concatenate((all_tensors, tensor_model), axis=0)

    return all_tensors


def get_subtensor_54(bsps_54_t, tensor_type="1channel", unfold_code=1):
    """
    Get (6 x 4 x 3) tensor for 1 instance of time.

    Parameters:
    bsps_64_t: 1 instance of time of bsps_64

    Return:
    subtensor: 6 x 4 x 3 matrix
    """

    patches = get_patches_name_54(bsps_54_t)

    R4 = np.mean((patches["A5"], patches["A4"], patches["R3"]))
    R8 = np.mean((patches["D11"], patches["D12"], R4))
    R7 = np.mean((patches["D10"], patches["D11"], patches["R3"]))
    R6 = np.mean((patches["D9"], patches["D10"], patches["R2"]))
    R5 = np.mean((patches["D8"], patches["D9"], patches["R1"]))

    L4 = np.mean((patches["B5"], patches["B4"], patches["L3"]))
    L8 = np.mean((patches["C11"], patches["C12"], L4))
    L7 = np.mean((patches["C10"], patches["C11"], patches["L3"]))
    L6 = np.mean((patches["C9"], patches["C10"], patches["L2"]))
    L5 = np.mean((patches["C8"], patches["D9"], patches["L1"]))

    interp_lat_R4 = np.mean((patches["A6"], patches["A5"], R4))

    interp_lat_R8 = np.mean((patches["D12"], patches["D11"], R8))
    interp_lat_L8 = np.mean((patches["C6"], patches["C5"], L8))
    interp_lat_L4 = np.mean((patches["B6"], patches["B5"], L4))

    interp_lat_R1 = np.mean((patches["A1"], patches["A2"], patches["R1"]))
    interp_lat_R5 = np.mean((patches["D7"], patches["D8"], R5))
    interp_lat_L5 = np.mean((patches["C1"], patches["C2"], L5))
    interp_lat_L1 = np.mean((patches["B1"], patches["B2"], patches["L1"]))

    subtensor = np.array(
        [
            [
                [
                    patches["B6"],
                    patches["B12"],
                    patches["A12"],
                    patches["A6"],
                    interp_lat_R4,
                    interp_lat_R8,
                    patches["D12"],
                    patches["D6"],
                    patches["C12"],
                    patches["C6"],
                    interp_lat_L8,
                    interp_lat_L4,
                    patches["B6"],
                    patches["B12"],
                    patches["A12"],
                    patches["A6"],
                ],
                [
                    patches["B5"],
                    patches["B11"],
                    patches["A11"],
                    patches["A5"],
                    R4,
                    R8,
                    patches["D11"],
                    patches["D5"],
                    patches["C11"],
                    patches["C5"],
                    L8,
                    L4,
                    patches["B5"],
                    patches["B11"],
                    patches["A11"],
                    patches["A5"],
                ],
                [
                    patches["B4"],
                    patches["B10"],
                    patches["A10"],
                    patches["A4"],
                    patches["R3"],
                    R7,
                    patches["D10"],
                    patches["D4"],
                    patches["C10"],
                    patches["C4"],
                    L7,
                    patches["L3"],
                    patches["B4"],
                    patches["B10"],
                    patches["A10"],
                    patches["A4"],
                ],
                [
                    patches["B3"],
                    patches["B9"],
                    patches["A9"],
                    patches["A3"],
                    patches["R2"],
                    R6,
                    patches["D9"],
                    patches["D3"],
                    patches["C9"],
                    patches["C3"],
                    L6,
                    patches["L2"],
                    patches["B3"],
                    patches["B9"],
                    patches["A9"],
                    patches["A3"],
                ],
                [
                    patches["B2"],
                    patches["B8"],
                    patches["A8"],
                    patches["A2"],
                    patches["R1"],
                    R5,
                    patches["D8"],
                    patches["D2"],
                    patches["C8"],
                    patches["C2"],
                    L5,
                    patches["L1"],
                    patches["B2"],
                    patches["B8"],
                    patches["A8"],
                    patches["A2"],
                ],
                [
                    patches["B1"],
                    patches["B7"],
                    patches["A7"],
                    patches["A1"],
                    interp_lat_R1,
                    interp_lat_R5,
                    patches["D7"],
                    patches["D1"],
                    patches["C7"],
                    patches["C1"],
                    interp_lat_L5,
                    interp_lat_L1,
                    patches["B1"],
                    patches["B7"],
                    patches["A7"],
                    patches["A1"],
                ],
            ]
        ]
    )

    subtensor = subtensor.reshape(1, 6, 16)

    return subtensor


def get_patches_name_54(bsps_64):
    """
    Get names of patches in bsps_64

    Parameters:
        bsps_64:
    Return:
        patches: dictionary whit patche name as key and bsps as value.
    """
    patches = {}

    index = 1
    for i in range(0, 12):
        patches["A{0}".format(index)] = bsps_64[i]
        index += 1

    index = 1
    for i in range(12, 24):
        patches["B{0}".format(index)] = bsps_64[i]
        index += 1

    index = 1
    for i in range(24, 36):
        patches["C{0}".format(index)] = bsps_64[i]
        index += 1

    index = 1
    for i in range(36, 48):
        patches["D{0}".format(index)] = bsps_64[i]
        index += 1

    index = 1
    for i in range(48, 51):
        patches["L{0}".format(index)] = bsps_64[i]
        index += 1

    index = 1
    for i in range(51, 54):
        patches["R{0}".format(index)] = bsps_64[i]
        index += 1

    return patches


def load_egms(model_name, sinusoid=False):
    """
    Load electrograms and select 2500 time instants and 2048 nodes

    Parameters:
        model (str): Model to load

    Returns:
        x: Electrograms for the selected model
    """

    try:
        EG = np.transpose(
            np.array((h5py.File(directory + model_name + "/EGMs.mat", "r")).get("x"))
        )
    except:
        EG = scipy.io.loadmat(directory + model_name + "/EGMs.mat").get("x")

    return EG


def load_ecgs(model_name, sinusoid=False):
    """
    Load bsps

    Parameters:
        model (str): Model to load

    Returns:
        x: Electrograms for the selected model
    """

    try:
        EG = np.transpose(
            np.array((h5py.File(directory + model_name + "/ECGs.mat", "r")).get("y"))
        )
    except:
        EG = scipy.io.loadmat(directory + model_name + "/ECGs.mat").get("y")

    return EG


def normalize_by_models(data, Y_model):
    """
    This function normalizes the input tensor between -1 and 1 in each model separately

    """
    # Normalize by models of BSPM

    data_orig = data
    if data.ndim == 3:
        data = reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))

    elif data.ndim == 4:
        data = reshape(
            data, (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3])
        )
    elif data.ndim != 2:
        raise (
            ValueError(
                "Input shape for norm not correct. Nnim should be 2, but is", data.ndim
            )
        )

    data_n = []

    for model in np.unique(Y_model):
        arr_to_norm = data[
            np.where((Y_model == model))
        ]  # select window of signal belonging to model i
        norm_model = normalize_array(arr_to_norm, 1, -1)
        data_n.extend(norm_model)  # Add to new norm array

    data_n = np.array(data_n)

    if data_orig.ndim == 3:
        data_n = data_n.reshape(
            data_orig.shape[0], data_orig.shape[1], data_orig.shape[2]
        )

    elif data_orig.ndim == 4:
        data_n = data_n.reshape(
            data_orig.shape[0],
            data_orig.shape[1],
            data_orig.shape[2],
            data_orig.shape[3],
        )

    return data_n


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
