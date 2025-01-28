import sys
sys.path.append("../Code")
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr, spearmanr



def downsampling(array, fs_sub):
    downsampling_factor = int(500 / fs_sub)
    array_sub = array[::downsampling_factor]
    return array_sub

def normalize_array(array, high=1, low=-1, axis_n=0):
    mins = np.min(array, axis=axis_n)
    maxs = np.max(array, axis=axis_n)
    rng = maxs - mins
    if axis_n == 1:
        array = array.T
    norm_array = high - (((high - low) * (maxs - array)) / rng)
    if axis_n == 1:
        norm_array = norm_array.T
    return norm_array



def compare_r_peaks(real_peaks, recon_peaks, signal_true, signal_pred,tolerance_samples):
    """
    Compare the R-peak detection between the real signal and the reconstructed signal.

    Parameters
    ----------
    real_peaks : array-like
        Indices of the R-peaks in the real signal (sorted).
    recon_peaks : array-like
        Indices of the R-peaks in the reconstructed signal (sorted).
    tolerance_samples : int
        Tolerance in the number of samples to consider two peaks as coincident.

    Returns
    -------
    metrics : dict
        Dictionary containing:
            - TP (True Positives)
            - FP (False Positives)
            - FN (False Negatives)
            - Sensitivity
            - Precision
            - F1-score
            - MeanTimeError: average time error (in samples) between matched peaks
            - StdTimeError: standard deviation of the time error (in samples) between matched peaks
    """
    

    matched_peaks = []       # Picos emparejados: [(pos_real, val_real, pos_recon, val_recon)]
    unmatched_real = []      # Picos reales no emparejados (FN): [(pos_real, val_real)]
    unmatched_recon = []     # Picos reconstruidos no emparejados (FP): [(pos_recon, val_recon)]
    matched_peaks_r = []     # Picos emparejados: [(pos_real)]

    recon_peaks_remaining = list(recon_peaks)

    #list corr
    error_list=[]
    TP, FP, FN = 0, 0, 0

    # Iterar sobre cada pico real
    for r_peak in real_peaks:
        min_diff = float("inf")
        best_candidate = None

    
        for c_peak in recon_peaks_remaining:
            diff = abs(r_peak - c_peak)
            if diff < min_diff:
                min_diff = diff
                best_candidate = c_peak
        
        # Si hay coincidencia dentro del umbral
        if best_candidate is not None and min_diff <= tolerance_samples:
            TP += 1
        
            peaks_true_values = signal_true[r_peak]
            peaks_recon_values = signal_pred[r_peak]
            error = np.sqrt(abs(peaks_true_values - peaks_recon_values)) #RMSE de la diferencia de amplitud entre el pico real y el predicho
            error_list.append(error)             

            matched_peaks.append(best_candidate)
            matched_peaks_r.append(r_peak)
            recon_peaks_remaining.remove(best_candidate)  


        else:
            FN += 1
            unmatched_real.append((r_peak, signal_true[r_peak]))

    # Los picos reconstruidos restantes son falsos positivos
    for c_peak in recon_peaks_remaining:
        unmatched_recon.append((c_peak, signal_pred[c_peak]))

    FP = len(unmatched_recon)

    error = np.mean(error_list)
    
    #  metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1_score = (2 * sensitivity * precision) / (sensitivity + precision) if (sensitivity + precision) > 0 else 0.0
    
    
    metrics = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Sensitivity": sensitivity,
        "Precision": precision,
        "F1-score": f1_score,
        "Error": error
    }
    
    return metrics, matched_peaks, matched_peaks_r




def deflexion_detection(array, fs, amplitude_threshold=0.25, prominence_value=0.1):
    print('Deflexion detection')
    # Detectar deflexiones
    peaks_list = []
    for lead in range(0,array.shape[1]):
        lead_i= array[:, lead]
        #ecg_cleaned = nk.ecg_clean(lead_i, sampling_rate=fs, method="neurokit")
        #signals, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method='elgendi2010', correct_artifacts=True)
        distance_in_samples = int(0.1 * fs)  # = 200, si fs=1000
        

        peaks=scipy.signal.find_peaks(lead_i, distance=distance_in_samples, prominence=prominence_value)
        peaks=peaks[0]
        peaks_list.append(peaks)

    print('Distance in samples:', distance_in_samples)

    return peaks_list

from scipy.signal import butter, filtfilt

# Función para aplicar filtro Butterworth pasa-banda
def bandpass_filter(signal, fs, lowcut, highcut):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype="band")  # Orden 4 del filtro
    return filtfilt(b, a, signal)



