# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:22:24 2018

@author: Miguel Ángel
"""

import numpy as np
from scipy import signal as sigproc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


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


def detrendSpline(EGM_ECG, fs, l_w=0.25):
    """input signal: Signal to be detrended
          fs: sampling frequency
          l_w: window length in secs (1 sec by default)
    output ecg_detrend: ecg without the base line
           s_pp: trend
    """

    EGM_ECG_detrend = np.zeros(EGM_ECG.shape)
    trend = np.zeros(EGM_ECG.shape)

    for i in range(EGM_ECG.shape[0]):
        signal = EGM_ECG[i, :]
        L_s = len(signal)
        t = np.arange(0, L_s) * 1.0 / fs

        numSeg = np.floor(t[-1] / l_w)

        s_m = np.zeros(int(numSeg))
        t_m = np.zeros(int(numSeg))

        for k in range(int(numSeg)):
            # for over each window and compute the median
            ind_seg = (t >= (k) * l_w) & (t <= (k + 1) * l_w)
            t_aux = t[ind_seg]
            t_m[k] = t_aux[int(len(t_aux) / 2)]
            s_m[k] = np.median(signal[ind_seg])

        # fit the spline to the median points
        # Add first and last value in
        t_m = np.concatenate(([0], t_m, [t[-1]]))
        s_m = np.concatenate(([signal[0]], s_m, [signal[-1]]))
        cp = interp1d(t_m, s_m, kind="cubic")
        trend[i, :] = cp(t)
        EGM_ECG_detrend[i, :] = signal - trend[i, :]

    return EGM_ECG_detrend, trend


def ECG_filtering_real_data(signal_orig, fs, filt_order=6, f_cut=30):
    """
    Low-pass frequency filtering of ECG-EGM (real data).

    Parameters:
        signal (array): signal to process
        fs (int): sampling rate
        filt-order (int): filt order (default=6)
        f_cut (int-float): cut-off frecuency (default=30Hz)
    Returns:
        proc_ECG_EGM (array): filtered ECG-EGM
    """

    signal, _ = detrendSpline(signal_orig, fs, l_w=0.25)

    b, a = sigproc.butter(filt_order, f_cut, "lp", fs=fs)
    proc_ECG_EGM = sigproc.filtfilt(b, a, signal, axis=1)

    return proc_ECG_EGM


def ECG_filtering(signal, fs, f_low=3, f_high=30, model="SR"):
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

    if model == "SR":
        # LPF Filtering
        b, a = sigproc.butter(4, f_high / round((fs / 2)), btype="low", fs=fs)
        proc_ECG_EGM = np.zeros(sig_temp.shape)

        for index in range(0, sig_temp.shape[0]):
            proc_ECG_EGM[index, :] = sigproc.filtfilt(b, a, sig_temp[index, :])
    else:
        # Bandpass filtering
        b, a = sigproc.butter(
            4,
            [f_low / round((fs / 2)), f_high / round((fs / 2))],
            btype="bandpass",
            fs=fs,
        )
        proc_ECG_EGM = np.zeros(sig_temp.shape)

        for index in range(0, sig_temp.shape[0]):
            proc_ECG_EGM[index, :] = sigproc.filtfilt(b, a, sig_temp[index, :])

    return proc_ECG_EGM


def addwhitenoise(signal, fs=50, SNR=20, model="AF", seed="N"):
    """
    Add gaussian white noise. We assume constant noise power in all electrodes.

    Parameters:
        signal (array): signal to process
        SNR (int): mean SNR value (in dB)
        fs (float): sampling rate. Default: 500Hz (for computerized models)
        model (string): check if the model to filter is an AF or SR model (default='AF')
    Returns:
        filtered_ECG_EGM (array): ECG-EGM with additive noise (filtered)
        noisy_ECG_EGM (array): ECG-EGM with additive noise
        Cn (array): Noise covariance matrix
    """

    # Generate the seed for reproducibility of simulations (we generate the same random numbers)

    if seed == "Y":
        np.random.seed(0)

    PowerInSigdB = 10 * np.log10(np.mean(np.power(np.abs(signal), 2)))

    sigma = np.sqrt(np.power(10, (PowerInSigdB - SNR) / 10))
    if signal.ndim == 2:
        noise = sigma * (np.random.randn(signal.shape[0], signal.shape[1]))
    else: 
        noise = sigma * (np.random.randn(signal.shape[0], signal.shape[1], signal.shape[2] ))

    noisy_EGM = noise + signal
    
    plt.figure()
    plt.plot(noise[:, 0, 0], label = 'noise')
    plt.plot(signal[:, 0, 0], label = 'signal')
    plt.plot(noisy_EGM[:, 0, 0], label='noisy')
    plt.legend()
    plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Noise_module/white_noise.png')



    Cn = np.power(sigma, 2) * np.eye(noisy_EGM.shape[0])

    # filtered_EGM = ECG_filtering(noisy_EGM, fs, model = model) Lo saco de aquí porque lcoge la función ECG_filtering de este script

    return noisy_EGM, Cn
