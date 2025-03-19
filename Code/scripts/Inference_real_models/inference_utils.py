import numpy as np
import scipy.signal as sigproc
import cv2
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def normalize_array(array, high, low, axis_n=0):
    """
    Normalizes a 2D array along the specified axis to be within the given range [low, high].

    Parameters:
        array (ndarray): Input 2D array.
        high (float): Upper bound of the normalization range.
        low (float): Lower bound of the normalization range.
        axis_n (int, optional): Axis along which to normalize (default is 0).

    Returns:
        ndarray: Normalized array.
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
    Removes the mean from each row (node) of a 2D signal.

    Parameters:
        signal (ndarray): 2D array where each row is a signal from a node.

    Returns:
        ndarray: Signal with mean removed for each row.
    """
    signal=signal.T

    signotmean = np.zeros(signal.shape)

    for node in range(0, signal.shape[0]):
        signotmean[node, :] = sigproc.detrend(signal[node, :], type="constant")
    return signotmean.T

def ECG_filtering(signal, fs, order=2, f_low=3, f_high=30):
    """
    Applies bandpass filtering to an ECG signal using a Butterworth filter.

    Parameters:
        signal (ndarray): Input signal.
        fs (int): Sampling frequency.
        order (int, optional): Filter order (default is 2).
        f_low (float, optional): Low cut-off frequency (default is 3 Hz).
        f_high (float, optional): High cut-off frequency (default is 30 Hz).

    Returns:
        ndarray: Filtered signal.
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

def map_custom_electrodes(bspms, flip_cols=True):
    """
    Reshapes and interpolates 1D BSPM signals into 2D images.

    Parameters:
        bspms (ndarray): BSPM signals.
        tank_el_position (ndarray): Electrode positions.
        from_12_5 (bool, optional): Whether the input is reshaped from (12,5) format (default is False).

    Returns:
        ndarray: Reshaped and interpolated BSPM images.
    """

    matrix_2D= [[ 145, 146, 155, 156, 165, 166, 129, 130, 139, 140, 181, 182], 
                [ 147, 147, 157, 157, 167, 167, 131, 131, 141, 141, 183, 183],
                [ 148, 149, 158, 159, 168, 169, 132, 133, 142, 143, 184, 185],
                [ 150, 151, 160, 161, 170, 171, 134, 135, 144, 177, 186, 187],
                [ 152, 152, 162, 162, 172, 172, 136, 136, 178, 178, 188, 188],
                [ 153, 154, 163, 164, 173, 174, 137, 138, 179, 180, 189, 190]]
    
    unique_vals = sorted(set(sum(matrix_2D, [])))

    # Crear el diccionario con mapeo consecutivo
    mapping = {value: idx + 1 for idx, value in enumerate(unique_vals)}

    # Mostrar el resultado
    print(mapping)
    
    f_a=np.array([[129, 130],
                 [131, 131],
                 [132, 133],
                 [134, 135],
                 [136, 136],
                 [137, 138]])    
    
    f_b=np.array([[139, 140], 
                  [141, 141], 
                  [142, 143],
                  [144, 177],
                  [178, 178],
                  [179, 180]])
    f_c=np.array([[181,182], 
                [183,183],
                [184,185],
                [186,187],
                [188,188],
                [189, 190]])
    f_d=np.array( [[145, 146],
                [147, 147],
                [148, 149],
                [150, 151],
                [152, 152],
                [153, 154]])
    
    f_e=np.array([[155, 156],
                [157, 157],
                [158, 159], 
                [160, 161], 
                [162, 162], 
                [163, 164]])
    
    f_f = np.array([[165, 166], 
                     [167, 167], 
                     [168, 169], 
                     [170, 171],
                     [172, 172],
                     [173, 174]])
    
 
    #Flip columns
    if flip_cols:
        f_a[:, [0, 1]]=f_a[:, [1, 0]]
        f_b[:, [0, 1]]=f_b[:, [1, 0]]
        f_c[:, [0, 1]]=f_c[:, [1, 0]]
        f_d[:, [0, 1]]=f_d[:, [1, 0]]
        f_e[:, [0, 1]]=f_e[:, [1, 0]]
        f_f[:, [0, 1]]=f_f[:, [1, 0]]
        print(f_a)

    
    conc=np.concatenate((f_e, f_d, f_c, f_b, f_a, f_f, f_e, f_d), axis=1)
    
    conc_mapped = np.vectorize(mapping.get)(conc)
    
    print(conc_mapped)

    new_order = np.array(conc_mapped).flatten() - 1  # Ajustamos los índices
    valid_indices = [idx for idx in new_order if 0 <= idx < 60]

    # Reordenamos las columnas del array signals
    signals_reordered = bspms[:, valid_indices]
    signals_reordered_reshape = np.reshape(signals_reordered, (signals_reordered.shape[0], 
                                                               conc_mapped.shape[0], 
                                                               conc_mapped.shape[1]) )

    return signals_reordered_reshape

    

def bspm_to_images(bspms, tank_el_position, from_12_5=False, custom_layout=True):
    """
    Reshapes and interpolates 1D BSPM signals into 2D images.

    Parameters:
        bspms (ndarray): BSPM signals.
        tank_el_position (ndarray): Electrode positions.
        from_12_5 (bool, optional): Whether the input is reshaped from (12,5) format (default is False).

    Returns:
        ndarray: Reshaped and interpolated BSPM images.
    """

    if custom_layout:
        bspms=map_custom_electrodes(bspms)
    
    bspms_before_interpol=bspms


    if from_12_5:
        bspms=bspms.reshape(bspms.shape[0], 12, 5)
    bspms_interpol_array=[]
    for element in range(bspms.shape[0]):
        
        signal=bspms[element, :, :]
        element_interp = cv2.resize(signal, (32, 12), interpolation=cv2.INTER_CUBIC)

        if element==0:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(bspms[element, :, :], cmap='gray')
            plt.title('Original BSPM')
            plt.subplot(1, 2, 2)
            plt.imshow(element_interp, cmap='gray')
            plt.title('interpolated BSPM')
            plt.savefig(f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/interpolated_bspm_{element}_bicubic.png")
            plt.close()
            print('saved image at ', f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/interpolated_bspm_{element}_bicubic.png")
            

        bspms_interpol_array.append(element_interp)

    bspms_reshaped=np.array(bspms_interpol_array)
    bspms_reshaped = np.transpose(bspms_reshaped, (0, 1, 2))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(bspms[element, :, :], cmap='gray')
    plt.title('Original BSPM')
    plt.subplot(1, 2, 2)
    plt.imshow(element_interp, cmap='gray')
    plt.title('interpolated BSPM')
    plt.savefig(f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/interpolated_bspm_{element}_bicubic.png")
    plt.close()
    print('saved image at ', f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/interpolated_bspm_{element}_bicubic.png")
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(bspms_reshaped[:, 0, 0])
    plt.title('Original BSPM')
    plt.subplot(1, 2, 2)
    plt.plot(bspms_before_interpol[:, 0, 0])
    plt.title('interpolated BSPM')
    plt.savefig(f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/interpolated_bspm_1d_{element}_bicubic.png")
    plt.close()
    print('saved image at ', f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/inference_heartlab/interpolated_bspm_1d_{element}_bicubic.png")
    



    return bspms_reshaped, bspms_before_interpol

def correlation_by_node(array1, array2):
    """
    Computes Spearman correlation for each node (column) between two arrays.

    Parameters:
        array1 (ndarray): First array (n, m).
        array2 (ndarray): Second array (n, m).

    Returns:
        ndarray: Array of correlation values for each node (column).
    """

    # Check identical dimensions
    assert (
        array1.shape == array2.shape
    ), "Arrays must have the identical dimensions"

    # Compute correlation node-wise
    n_cols = array1.shape[1]
    print('Computing correlation in :', n_cols, 'nodes')
    corr = np.zeros(n_cols)
    for i in range(n_cols):
        corr[i], _ = spearmanr(array1[:, i], array2[:, i]) # or pearsonr

    return corr

def reshape_tensor(tensor, n_dim_input, n_dim_output):
    """
    Reshapes a tensor based on specified input and output dimensions.

    Parameters:
        tensor (ndarray): Input tensor.
        n_dim_input (int): Original number of dimensions.
        n_dim_output (int): Desired number of dimensions.

    Returns:
        ndarray: Reshaped tensor.
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
    """
    Generates batches for inference based on data type.

    Parameters:
        data (ndarray): Input data.
        batch_size (int): Batch size.
        type_data (str): Type of data ('bspm' or 'egms').

    Returns:
        ndarray: Batches of data.
    """

    n_batch=batch_size

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
    

    return data_in_batches
