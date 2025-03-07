import numpy as np
import scipy.signal as sigproc
import cv2
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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

def remove_mean(signal):
    """
    Remove mean from signal

    Parameters:
        signal (array): signal to process

    Returns:
        signotmean: signal with its mean removed
    """
    signal=signal.T

    signotmean = np.zeros(signal.shape)

    for index in range(0, signal.shape[0]):
        signotmean[index, :] = sigproc.detrend(signal[index, :], type="constant")
    return signotmean.T

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

    sig_temp = remove_mean(signal)

    #sig_temp = signal

    # Bandpass filtering
    b, a = sigproc.butter(
        order,
        [f_low / round((fs / 2)), f_high / round((fs / 2))],
        btype="bandpass",
    )

    proc_ECG_EGM = np.zeros(sig_temp.shape)
    if sig_temp.ndim == 3:
        for i in range(sig_temp.shape[1]):
            for j in range(sig_temp.shape[2]):
                # for index in range(sig_temp.shape[0]):
                proc_ECG_EGM[:, i, j] = sigproc.filtfilt(b, a, sig_temp[:, i, j])
    else:
        for node in range(0, sig_temp.shape[1]):
            proc_ECG_EGM[:, node] = sigproc.filtfilt(b, a, sig_temp[:, node])

    return proc_ECG_EGM

def bspm_to_images(bspms, tank_el_position):
    '''
    This function reshapes 1D bspms into 2D and interpolates to fit the input shape
    
    '''

    matrix_2D=[[ 145, 146, 155, 156, 165, 166, 129, 130, 139, 140, 181, 182], 
                [ 147, 147, 157, 157, 167, 167, 131, 131, 141, 141, 183, 183],
                [ 148, 149, 158, 159, 168, 169, 132, 133, 142, 143, 184, 185],
                [ 150, 151, 160, 161, 170, 171, 134, 135, 144, 177, 186, 187],
                [ 152, 152, 162, 162, 172, 172, 136, 136, 178, 178, 188, 188],
                [ 153, 154, 163, 164, 173, 174, 137, 138, 179, 180, 189, 190]]

    bspms_reshaped=bspms.reshape(bspms.shape[0], 12, 5)
    bspms_interpol_array=[]
    for element in range(bspms_reshaped.shape[0]):
        signal=bspms_reshaped[element, :, :]
        element_interp = cv2.resize(signal, (12, 32), interpolation=cv2.INTER_CUBIC)

        '''
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(bspms_reshaped[element, :, :], cmap='gray')
        plt.title('Original BSPM')
        plt.subplot(1, 2, 2)
        plt.imshow(element_interp, cmap='gray')
        plt.title('interpolated BSPM')
        plt.savefig(f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/interpolated_bspm_{element}.png")
        plt.close()
        print('saved image at ', f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/interpolated_bspm_{element}.png")
        '''

        bspms_interpol_array.append(element_interp)
    bspms_reshaped=np.array(bspms_interpol_array)
    bspms_reshaped = np.transpose(bspms_reshaped, (0, 2, 1))

        

    '''
    bspms_reshaped
    for instant_i in range(bspms.shape[0]):
        bspms_instant=bspms[instant_i, :]
        id_to_signal = dict(zip(tank_el_position.flatten(), bspms_instant))
        mapped_signal_matrix = np.vectorize(lambda x: id_to_signal.get(x, np.nan))(matrix_2D)
        bspms_reshaped.append(mapped_signal_matrix)
    bspms_reshaped=np.array(bspms_reshaped)
    '''
    return bspms_reshaped

def correlation_by_node(array1, array2):
                """
                Calcula la correlación de Spearman entre las columnas de dos arrays.

                Args:
                    array1: un array de numpy de dimensión (n,m)
                    array2: otro array de numpy de dimensión (n,m)

                Returns:
                    Un array de numpy de dimensión (m,) que contiene la correlación de Spearman
                    de las columnas de array1 y array2.
                """

                # Verificar si ambos arrays tienen las mismas dimensiones
                assert (
                    array1.shape == array2.shape
                ), "Los arrays deben tener las mismas dimensiones."

                # Calcular la correlación de Spearman de las columnas de ambos arrays
                n_cols = array1.shape[1]
                print('Computing correlation in :', n_cols, 'nodes')
                corr = np.zeros(n_cols)
                for i in range(n_cols):
                    corr[i], _ = spearmanr(array1[:, i], array2[:, i]) # or pearsonr

                return corr

def reshape_tensor(tensor, n_dim_input, n_dim_output):
    """
    Reshapes the tensors used during pipeline, considering that the first two dimensions are (#n batches, batch size).
    In the case of n_dim_input = 5, the last dimension is the number of channels.

    Parameters
    ----------
    tensor: tensor to reshape
    n_dim_input: input shape
    n_dim_output: desired output shape

    Returns
    -------

    """

    try:

        # case of autoencoder output: first two dimensions
        if n_dim_input == 5 and n_dim_output == 2:
            reshaped_tensor = np.reshape(
                tensor,
                (
                    tensor.shape[0] * tensor.shape[1],
                    tensor.shape[2] * tensor.shape[3] * tensor.shape[4],
                ),
            )
            return reshaped_tensor

        # case of regression output
        elif n_dim_input == 3 and n_dim_output == 2:
            reshaped_tensor = np.reshape(
                tensor, (tensor.shape[0] * tensor.shape[1], tensor.shape[2])
            )
            return reshaped_tensor

        elif n_dim_input == 5 and n_dim_output == 4:
            reshaped_tensor = np.reshape(
                tensor,
                (
                    tensor.shape[0] * tensor.shape[1],
                    tensor.shape[2],
                    tensor.shape[3],
                    tensor.shape[4],
                ),
            )
            return reshaped_tensor
        elif n_dim_input == 4 and n_dim_output == 5:
            reshaped_tensor = np.reshape(
                tensor,
                (tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3], 1),
            )

            return reshaped_tensor

        elif n_dim_input == 4 and n_dim_output == 2:
            reshaped_tensor = np.reshape(
                tensor,
                (tensor.shape[0], tensor.shape[1] * tensor.shape[2] * tensor.shape[3]),
            )

            return (reshaped_tensor,)

    except:
        raise (
            ValueError(
                "Input shape - Output shape combination is not implemented: check reshape_tensor documentation"
            )
        )

def batch_generation(data, batch_size, type_data):
    '''
    This function operates batch generation prior to inference
    
    
    '''
    rows = data.shape[0]
    n_batch=batch_size
    divisible_rows = (rows // n_batch) * n_batch

    if type_data== "bspm":
        # Batch generation
        data_in_batches = np.reshape(
                data,
                (
                    int(len(data) / n_batch),
                    n_batch,
                    data.shape[1],
                    data.shape[2],
                    1,
                ),
            )
        

    
    elif type_data=="egms":

        data_in_batches = np.reshape(
                data,
                (
                    int(len(data) / n_batch),
                    n_batch,
                    data.shape[1],
                    
                    1,
                ),
            )
    
    #Remove Nans
    data_in_batches = np.nan_to_num(
        data_in_batches, nan=0.0
    )  # Nans generated during noise addition

    return data_in_batches
