# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:22:24 2018

@author: Miguel Ángel
"""

import numpy as np
from scipy import signal as sigproc

def remove_mean(signal):
    """
    Remove mean from signal
 
    Parameters:
        signal (array): signal to process
 
    Returns:
        signotmean: signal with its mean removed
    """
    signotmean=np.zeros(signal.shape)
    for index in range(0, signal.shape[0]):
        signotmean[index,:]=sigproc.detrend(signal[index,:],type='constant')
    return signotmean


def ECG_filtering(signal, fs, f_low=3, f_high=30, model='SR'):
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
    sig_temp=remove_mean(signal)
    
    if model=='SR':       
        #LPF Filtering
        b, a = sigproc.butter(6, f_high/round((fs/2)), btype='low')
        proc_ECG_EGM=np.zeros(sig_temp.shape)

        for index in range(0, sig_temp.shape[0]):
            proc_ECG_EGM[index,:]=sigproc.filtfilt(b,a,sig_temp[index,:])
    else:
        #Bandpass filtering
        b, a = sigproc.butter(6, [f_low/round((fs/2)), f_high/round((fs/2))], btype='bandpass')
        proc_ECG_EGM=np.zeros(sig_temp.shape)

        for index in range(0, sig_temp.shape[0]):
            proc_ECG_EGM[index,:]=sigproc.filtfilt(b,a,sig_temp[index,:])

    return proc_ECG_EGM
            
def addwhitenoise(signal, SNR=20, seed='Y'):
    """
    Add gaussian white noise. We assume constant noise power in all electrodes.
    
    Parameters:
        signal (array): signal to process
        SNR (int): mean SNR value (in dB)
    Returns:
        noisy_ECG_EGM (array): ECG-EGM with additive noise
        Cn (array): Noise covariance matrix
    """
    
    #Generate the seed for reproducibility of simulations (we generate the same random numbers)
    if seed=='Y':
        np.random.seed(0)
    
    PowerInSigdB = 10*np.log10(np.mean(np.power(np.abs(signal),2)))
    
    sigma=np.sqrt(np.power(10,(PowerInSigdB-SNR)/10))
    noise=sigma*(np.random.randn(signal.shape[0],signal.shape[1]))
    
    noisy_ECG_EGM=noise+signal
    
    Cn=np.power(sigma,2)*np.eye(noisy_ECG_EGM.shape[0])
    return noisy_ECG_EGM,Cn