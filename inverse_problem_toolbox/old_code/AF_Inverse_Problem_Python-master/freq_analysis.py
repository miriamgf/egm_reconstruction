#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:52:31 2019

Spectral analysis functions for AF analysis

@author: obarquero
"""
import numpy as np
import scipy as sc
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt

def botterom_smith_df(egm,fs,f1 = 40,f2=250,f3 = 20,plot_flag = True):
    """
    Compute Botterom and Smith preprocessing of AF EGMs signals. Note that this
    preprocessing makes only sense when working on bipolar EGMs, otherwise the
    preprocessing might hurt more than help to estimate DF.
            
    Parameters
    ----------
    egm : numpy array (n_samples, 1)
        Electrogram signal. It can be in physical or normalized units.
    fs : int 
        Sampling Frequency in Hz. If it is smaller than 500 Hz, the signal is resample
        up to 1 Khz
    f1 : float
        Low frequency cut of the band-pass filter.
    f2 : float
        Up frequency cut of the band-pass filter.
    f3 : float
        frequency cut of the low-pas filter.
    plot_flag : Boolean (default, True)
        Allows to plot information about the preprocessing
            
    Returns
    -------
    z_egm : numpy array (n_samples, 1) 
        Preprocessed EGM, normalized units.
    P_z : numpy array (n_samples, 1) 
        Power Spectral Density estimation of the preprocessed EGM. [units^2/Hz]
    f : numpy array (n_samples, 1)
        Frequency vector
    df : float
        Dominant frequency estimation using Botterom-Smith preprocessing.
        
    References
    ----------
    
    [Bott-Smith]: Gregory W. Botteron and Joseph M. Smith. A Technique for 
    Measurement of the Extent of Spatial Organization of Atrial Activation 
    During Atrial Fibrillation in the Intact Human Heart. IEEE TBME, vol 42, num 6
    1995
    """
    
    #normalize egm
    egm = egm/np.max(np.abs(egm))
    
   # if fs < 1e3:
        #resample to 1 KHz
    #    new_fs = 1e3 #1 KHz
     #   egm = signal.resample_poly(egm,new_fs,fs)
        #change fs
     #   fs = new_fs
    
    eps = 2.2204e-16   
    t = np.arange(len(egm))/fs
    
    
    # 1. Remove base line
    
    egm_d,egm_trend = detrendSpline(egm,fs)
    
    if plot_flag:
        plt.figure()
        plt.plot(t,egm,label='Original EGM')
        plt.plot(t,egm_trend,'r:',linewidth = 2,)
        plt.plot(t,egm_d,label='EGM dentrended')
        plt.legend()
    
    # 2. Band-pass filtering between 40-250 Hz
   # f1 = 40 # flow 40 Hz
   # f2 = 250 #250 Hz. If the sampling frequency is 500Hz, then may be it is a good idea to reinterpolato to a 1KHz
    
        
    b,a = signal.butter(3,[f1, f2], 'bp',fs = fs)
    #TO DO: consider the case that fs = 500 or smaller
    #if fs <= 500:
    #    b,a = signal.butter(5,f1,'hp',fs=fs)        
    #elif fs> 500:
     #   b,a = signal.butter(5,[f1, f2], 'bp',fs = fs)
    
    #plot w,h
    w,h = signal.freqz(b,a,fs = fs)
    
    if plot_flag:
        plt.figure()
        plt.plot(w,np.abs(h))
        plt.xlabel('Freq (Hz)')
        plt.ylabel('Band-Pass Filter')
    
    #filter the signal
    egm_bp = signal.filtfilt(b,a,egm_d)
    
    if plot_flag:
        plt.figure()
        plt.plot(t,egm,label = 'Original normalized EGM')
        plt.plot(t,egm_bp, label = 'Band-Pass filtered EGM')
        plt.legend()
    
    # 3. Rectification of the signal
    
    egm_abs = np.abs(egm_bp)
    
    # 4. Low pas filtering
    
    f1 = 20 #Hz
    b1, a1 = signal.butter(3,f1,'lp', fs = fs)
    
    #plot
    w,h = signal.freqz(b1,a1,fs = fs)
    
    if plot_flag:
        plt.figure()
        plt.plot(w,np.abs(h))
        plt.xlabel('Freq (Hz)')
        plt.ylabel('Low-pass filter')
    
    z_egm = signal.filtfilt(b1,a1,egm_abs)
    
    f,Pz = signal.welch(z_egm,fs = fs, nfft = 2024)
    
    #5. Df estimation
    
    #TO DO

    return z_egm, Pz, f


    
    
##########################################
#   Base line removal
#########################################

def detrendSpline(signal,fs,l_w = 0.25):
    """ input signal: Signal to be detrended
              fs: sampliing frequency
              l_w: window length in secs (1 sec by default)
        output ecg_detrend: ecg without the base line
               s_pp: trend
    """
    L_s = len(signal)
    t = np.arange(0,L_s)*1./fs
    
    numSeg = np.floor(t[-1]/l_w)

    s_m = np.zeros(int(numSeg))
    t_m = np.zeros(int(numSeg))
    
    for k in range(int(numSeg)):
        #for over each window and compute the median
        ind_seg = (t >= (k)*l_w) & (t <= (k+1)*l_w)
        t_aux = t[ind_seg]
        t_m[k] = t_aux[int(len(t_aux)/2)]
        s_m[k] = np.median(signal[ind_seg]);
   
    #fit the spline to the median points
    #Add first and last value in
    t_m = np.concatenate(([0],t_m,[t[-1]]))
    s_m = np.concatenate(([signal[0]],s_m,[signal[-1]]))
    cp = interp1d(t_m,s_m,kind = 'cubic')
    trend = cp(t)
    signal_detrend = signal - trend;
    return signal_detrend,trend
    

# load data
    


#botterom_smith_df(egms[0,:],500)