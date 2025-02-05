import numpy as np
from scipy import signal as sigproc
import time
from scipy.signal import butter, filtfilt, welch, coherence
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import os




def bandpass_filter(signal, fs, lowcut, highcut):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype="band")  # Orden 4 del filtro
    return filtfilt(b, a, signal)


def normalize_array(array, high=1, low=-1,axis_n=0):
    """
    Normalizes a 2D array along the specified axis to be between -1 and 1.

    Parameters:
    - array: 2D numpy array to be normalized.
    - axis_n: Axis along which normalization is applied (default is 0).

    Returns:
    - norm_array: A numpy array with normalized values between -1 and 1 along the specified axis.
    """
    # Compute the minimum and maximum along the specified axis
    mins = np.min(array, axis=axis_n, keepdims=True)
    maxs = np.max(array, axis=axis_n, keepdims=True)
    rng = maxs - mins  # Range of the values

    # Handle division by zero
    rng[rng == 0] = 1  # Avoid division by zero by setting range to 1

    # Perform normalization to range [-1, 1]
    norm_array = -1 + 2 * (array - mins) / rng
    
    return norm_array

def postprocess_prediction(data, cutoff_DC=0.5):
    
    data_centered = remove_mean(data, cutoff_DC) #only detrend
    data_norm=normalize_array(data_centered, high=1, low=-1, axis_n=0)
    return data_norm


def remove_mean(signal, cutoff=0.5):
    """
    Remove mean from signal

    Parameters:
        signal (array): signal to process

    Returns:
        signotmean: signal with its mean removed
    """
    signotmean = np.zeros(signal.shape)
    for index in range(0, signal.shape[0]):

        detrended=sigproc.detrend(signal[index, :], type="constant")

        #Remove DC component near 0 Hz
        cutoff = cutoff
        order = 4 
        b, a = sigproc.butter(order, cutoff / (fs / 2), btype='high', analog=False)
        centered = sigproc.filtfilt(b, a, detrended)

        #additional detrend
        signotmean[index, :] = centered

    return signotmean

def compute_spectral_coherence(prediction, y_label, fs, ROI_freq=[0.5,30], nperseg_val=256, plot=False, custom_path=0):
        '''
        This function computes the spectral coherence between two signals and returns the mean coherence in the ROI_freq range
        

        Args:
            prediction: array with the predicted EGM signal
            y_label: array with the real EGM signal
            fs: sampling frequency
            ROI_freq: list with the lower and upper limit of the frequency range to compute the coherence
            nperseg_val: number of samples per segment to compute the coherence
        Returns:
            Cxy_roi_mean: mean coherence in the ROI_freq range
        
        '''

        print('Computing spectral coherence...')


        time_start=time.time()

        coh_list = []
        for channel in range(0,y_label.shape[1]):
        
            y_label_filtered = bandpass_filter(y_label[:, channel], fs, ROI_freq[0], ROI_freq[1])
            y_pred_filtered = bandpass_filter(prediction[:, channel], fs, ROI_freq[0], ROI_freq[1])

            #normalize
            y_label_filtered=normalize_array(y_label_filtered, high=1, low=-1, axis_n=0)
            y_pred_filtered=normalize_array(y_pred_filtered, high=1, low=-1, axis_n=0)


            # Alinear las señales
            #lag = np.argmax(np.correlate(y_label_filtered, y_pred_filtered, mode="full")) - len(y_label_filtered)
            #y_pred_aligned = np.roll(y_pred_filtered, lag)

            # Calcular los periodogramas de Welch para ambas señales
            f1, Pxx1 = welch(y_pred_filtered, fs, nperseg=nperseg_val)#, noverlap=nperseg_val//2)  # Señal 1
            f2, Pxx2 = welch(y_label_filtered, fs, nperseg=nperseg_val)#, noverlap=nperseg_val//2)  # Señal 2

            f_coh, Cxy = coherence(y_pred_filtered, y_label_filtered, fs=fs, nperseg=nperseg_val)#, noverlap=nperseg_val//2)

            # Suavizar coherencia para reducir ruido
            Cxy_smoothed = uniform_filter1d(Cxy, size=5)

            # Calcular promedio de coherencia en el ROI
            ROI_indices = (f_coh >= ROI_freq[0]) & (f_coh <= ROI_freq[1])
            coherence_mean_ROI = np.mean(Cxy_smoothed[ROI_indices])
            coh_list.append(coherence_mean_ROI)
        
        best_channel=np.argmax(coh_list)
        worst_channel=np.argmin(coh_list)

        channels_to_plot=[best_channel, worst_channel]
        if plot:
            for channel in channels_to_plot:
                if channel == best_channel:
                    id='best'
                else:
                    id='worst'
                y_label_filtered = bandpass_filter(y_label[:, channel], fs, ROI_freq[0], ROI_freq[1])
                y_pred_filtered = bandpass_filter(prediction[:, channel], fs, ROI_freq[0], ROI_freq[1])

                #normalize
                y_label_filtered=normalize_array(y_label_filtered, high=1, low=-1, axis_n=0)
                y_pred_filtered=normalize_array(y_pred_filtered, high=1, low=-1, axis_n=0)

                # Alinear las señales
                lag = np.argmax(np.correlate(y_label_filtered, y_pred_filtered, mode="full")) - len(y_label_filtered)
                y_pred_aligned = np.roll(y_pred_filtered, lag)


                # Calcular los periodogramas de Welch para ambas señales
                f1, Pxx1 = welch(y_pred_filtered, fs, nperseg=nperseg_val)#, noverlap=nperseg_val//2)  
                f2, Pxx2 = welch(y_label_filtered, fs, nperseg=nperseg_val)#, noverlap=nperseg_val//2)  

                f_coh, Cxy = coherence(y_pred_aligned, y_label_filtered, fs=fs, nperseg=nperseg_val)#, noverlap=nperseg_val)

                # Suavizar coherencia para reducir ruido
                Cxy_smoothed = uniform_filter1d(Cxy, size=5)

                plt.figure(figsize=(24, 12), tight_layout=True)
                plt.plot(f_coh, Cxy, label="Coherence (Original)")
                plt.plot(f_coh, Cxy_smoothed, label="Coherence (Smoothed)")
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("Coherence")
                plt.title("Coherence with Butterworth Filtering")
                plt.xlim([0, 40])
                plt.ylim([0, 1])
                plt.grid()
                path_to_save=f"{custom_path}/coh_coherence_{id}.png"
                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
                plt.savefig(path_to_save)
                print('Saved in ', path_to_save)
                plt.close()


                plt.figure(figsize=(24, 12),tight_layout=True)
                plt.plot(f1, Pxx1, label="Prediction")
                plt.plot(f2, Pxx2, label="Ground truth")
                plt.xlim([0, 40])
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("Power spectral density")
                plt.title("Power spectral density")
                plt.legend()
                plt.grid()
                path_to_save=f"{custom_path}/coh_welch_{id}.png"
                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
                plt.savefig(path_to_save)
                print('Saved in ', path_to_save)
                plt.close()

                plt.figure(figsize=(24, 12),tight_layout=True)
                plt.plot(y_pred_aligned, label="Prediction")
                plt.plot(y_label_filtered, label="Ground truth")
                plt.ylabel("Amplitude (normalized)")
                plt.xlabel("Samples")
                plt.title("Time domain")
                plt.legend()
                plt.grid()
                path_to_save=f"{custom_path}/coh_time_{id}.png"
                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
                plt.savefig(path_to_save)
                print('Saved in ', path_to_save)
                plt.close()
             
        return coh_list


#####################################################################################################################
#####################################################################################################################

#load signals
fs=100

prediction = np.load("/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/evaluation/toy/saved_signals_toy/prediction_patient_1.npy")
y_label = np.load("/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/evaluation/toy/saved_signals_toy/label_patient_1.npy")

# Posprocess prediction
prediction_post=postprocess_prediction(prediction, cutoff_DC=1.5)

# EGM filtering
y_label=normalize_array(y_label, high=1, low=-1, axis_n=1)
y_label = remove_mean(y_label, cutoff=1.5) 

#define path to save 
custom_path= "/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/evaluation/toy/coherence"
coh_list=compute_spectral_coherence(prediction_post, y_label, fs, ROI_freq=[1.5,10], nperseg_val=256, plot=True, custom_path=custom_path)
print('Mean coherence:', np.mean(coh_list))




