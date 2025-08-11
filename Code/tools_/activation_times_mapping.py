import numpy as np

def compute_activation_times(signal_matrix, sampling_rate_hz):
    """
    Calcula el mapa de tiempos de activación (Activation Time Map) a partir de señales unipolares.

    Parámetros:
    - signal_matrix: ndarray de forma (n_nodos, n_muestras)
    - sampling_rate_hz: frecuencia de muestreo (Hz), e.g. 500

    Retorna:
    - activation_times: ndarray de forma (n_nodos, 1) en milisegundos
    """
    # Derivada temporal en el eje de las muestras
    dv_dt = np.gradient(signal_matrix, axis=1)

    # Índice del mínimo (máxima pendiente negativa) para cada nodo
    activation_indices = np.argmin(dv_dt, axis=1)

    # Tiempo en milisegundos por muestra
    time_per_sample = 1000.0 / sampling_rate_hz

    # Convertir a tiempo y cambiar forma a (n_nodos, 1)
    activation_times = (activation_indices * time_per_sample).reshape(-1, 1)

    activation_times= activation_times.ravel()
    activation_times = list(activation_times) 

    return activation_times

from scipy.signal import find_peaks

def extract_latido(signal_matrix, fs, latido_index=0, window_ms=300):
    """
    Extrae una ventana temporal centrada en el pico R del latido deseado.
    
    Parámetros:
    - signal_matrix: ndarray (n_nodos, n_muestras)
    - fs: frecuencia de muestreo en Hz
    - latido_index: índice del latido que quieres extraer (0 para el primero)
    - window_ms: duración de la ventana en milisegundos (ej. 300 ms)
    
    Retorna:
    - signal_segment: ndarray (n_nodos, muestras de ventana)
    """
    # Usamos el promedio espacial para detectar latidos
    avg_signal = signal_matrix.mean(axis=0)

    # Encontrar picos R (negativos si es unipolar)
    peaks, _ = find_peaks(-avg_signal, distance=fs*0.6)  # mínimo 600 ms entre latidos

    if latido_index >= len(peaks):
        raise ValueError(f"Solo se encontraron {len(peaks)} latidos, pero pediste el índice {latido_index}")

    center = peaks[latido_index]
    half_window = int((window_ms / 1000) * fs // 2)
    start = max(center - half_window, 0)
    end = min(center + half_window, signal_matrix.shape[1])

    return signal_matrix[:, start:end]
