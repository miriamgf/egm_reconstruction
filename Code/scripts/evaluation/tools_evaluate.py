import sys
sys.path.append("../Code")
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr, spearmanr
import pywt
from scipy.signal import butter, filtfilt, welch, csd


def downsampling(array, fs_sub):
    downsampling_factor = int(500 / fs_sub)
    array_sub = array[::downsampling_factor]
    return array_sub

import numpy as np
'''
def normalize_array(array, high=1, low=-1, axis_n=0):
    """
    Normaliza un array entre `low` y `high` a lo largo del eje especificado.

    Parameters:
        array (numpy.ndarray): El array a normalizar.
        high (float): Valor máximo deseado después de la normalización.
        low (float): Valor mínimo deseado después de la normalización.
        axis (int): Eje sobre el cual normalizar (0 para filas, 1 para columnas).

    Returns:
        numpy.ndarray: Array normalizado en el rango [low, high].
    """
    # Calcular mínimo y máximo a lo largo del eje especificado, manteniendo la forma correcta
    mins = np.min(array, axis=axis_n, keepdims=True)
    maxs = np.max(array, axis=axis_n, keepdims=True)
    
    # Evitar división por cero si todos los valores son iguales
    rng = maxs - mins
    rng[rng == 0] = 1  # Evita divisiones por cero en dimensiones constantes

    # Aplicar la normalización
    norm_array = low + (array - mins) * (high - low) / rng

    return norm_array
'''
    





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

def compute_HR_from_RR_dist(peaks, fs):
    # Calcular intervalos RR en segundos
    RR_intervals = np.diff(peaks) / fs

    # Calcular el ritmo cardíaco en BPM
    HR = 60 / np.mean(RR_intervals)
    return HR

def wavelet_filter(signal, wav='db4', level=4, threshold_mult=0.8):
    filtered_signal = []

    for lead in range(signal.shape[1]):  # Iterar por cada canal (columna)
        # Aplicar la Transformada Wavelet Discreta (DWT)
        coeffs = pywt.wavedec(signal[:, lead], wav, level=level)

        # Seleccionar los coeficientes de detalle del nivel 2 (índice 2)
        if len(coeffs) > 2:  # Verificar que existe el nivel 2
            detail_coeffs = np.array(coeffs[2])  # Copia segura del array

            # Aplicar un umbral para eliminar ruido
            threshold = np.std(detail_coeffs) * threshold_mult
            detail_coeffs[np.abs(detail_coeffs) < threshold] = 0  # Eliminación de ruido

            # Crear una copia de los coeficientes originales para la reconstrucción
            filtered_coeffs = coeffs[:]  # Copia completa de los coeficientes
            filtered_coeffs[2] = detail_coeffs  # Sustituimos solo el nivel 2 de detalle

            # Reconstrucción de la señal con los coeficientes filtrados
            filtered_lead = pywt.waverec(filtered_coeffs, wav)

            # Ajustar la longitud de la señal filtrada si es necesario
            if len(filtered_lead) > signal.shape[0]:
                filtered_lead = filtered_lead[:signal.shape[0]]
            elif len(filtered_lead) < signal.shape[0]:
                filtered_lead = np.pad(filtered_lead, (0, signal.shape[0] - len(filtered_lead)), 'constant')

        else:
            filtered_lead = signal[:, lead]  # Si no hay suficiente nivel de descomposición, dejar igual

        filtered_signal.append(filtered_lead)  # Guardar la señal filtrada del lead

    return np.array(filtered_signal).T  # Transponer para mantener la forma original


def deflexion_detection(array, fs, amplitude_threshold=0.25, prominence_value=0.1):
    print('Deflexion detection')
    # Detectar deflexiones
    peaks_list = []
    for lead in range(0,array.shape[1]):
        lead_i= array[:, lead]
        distance_in_samples = int(0.15 * fs)  #FA worst case scenario (más corto)--> 150 ms 
        peaks=scipy.signal.find_peaks(lead_i,distance=distance_in_samples, prominence=prominence_value)
        peaks=peaks[0]
        peaks_list.append(peaks)

    return peaks_list

def custom_coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
              nfft=None, detrend='constant', axis=-1):
    '''
    Function backbone from scipy.coherence

    Customized to normalize nfft
    
    '''
    freqs, Pxx = welch(x, fs=fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft, detrend=detrend,
                       axis=axis)
    _, Pyy = welch(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                   nfft=nfft, detrend=detrend, axis=axis)
    _, Pxy = csd(x, y, fs=fs, window=window, nperseg=nperseg,
                 noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    
    # Normalizar espectros 
    Pxx /= np.mean(Pxx)  # Normalización por energía total
    Pyy /= np.mean(Pyy)
    Pxy /= np.mean(np.abs(Pxy))
    
    Cxy = np.abs(Pxy)**2 / (Pxx * Pyy)

    return freqs, Cxy

# Función para aplicar filtro Butterworth pasa-banda
def bandpass_filter(signal, fs, lowcut, highcut):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype="band")  # Orden 4 del filtro
    return filtfilt(b, a, signal)



