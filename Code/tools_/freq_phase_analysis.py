# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:54:43 2018

@author: Miguel Ángel
"""

import numpy as np
from scipy import signal as sigproc
from scipy import stats
import matplotlib.pyplot as plt


def freq_response_plot(b, a, fs):
    """
    Plot frequency response of a filter

    Parameters:
        b, a (array): coefficients of the filter
        fs (int): sampling frequency
    """

    w, h = sigproc.freqz(b, a)
    w = w * fs / (2 * np.pi);
    fig = plt.figure()
    ax1 = plt.subplot(121)
    plt.plot(w, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [Hz]')
    plt.axis('tight')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.xlim((0, 60))

    ax3 = plt.subplot(122)
    plt.plot(w, abs(h), 'b')
    plt.ylabel('Amplitude', color='b')
    plt.xlabel('Frequency [Hz]')
    plt.axis('tight')
    ax4 = ax3.twinx()
    angles = np.unwrap(np.angle(h))
    ax4.plot(w, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.xlim((0, 60))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.suptitle('Digital filter frequency response')
    plt.show()


def kuklik_DF_phase(signals, fs):
    """
    Kuklik signal processing for DF and phase calculation.

    Parameters:
        signals (array): signals to process
        fs (int): sampling frequency
    Returns:
        DFs (array): dominant frequency for each channel
        signals_k (array): sine recompositions for phase computation (Kuklik et al)
        instant_phase (array): instant phase
    """

    # Define initial parameters
    lowf = 1.5
    highf = 25
    period_min = 130e-3
    period_max = 280e-3

    # Define sizes
    siglen = np.shape(signals)[1]
    n_signals = np.shape(signals)[0]

    sig = np.zeros((n_signals, siglen))
    rsig = np.zeros((n_signals, siglen))

    # Filter signals
    b, a = sigproc.butter(4, [lowf / round((fs / 2)), highf / round((fs / 2))], btype='bandpass')
    for i in range(0, n_signals):
        sig[i, :] = sigproc.filtfilt(b, a, signals[i, :])

    # Save array of dV/dt --> first derivative
    diffv = np.diff(sig)

    # Determine PD (Welch Periodogram)
    pd = np.empty((n_signals, 1));
    pd[:] = np.nan;
    tvec = np.arange(1 / fs, siglen / fs, 1 / fs)  # Time at each sample in seconds
    seglength = 2 * fs;
    n_overlap = fs;
    nFFT = np.power(200, 2);  # Zero pad FFT to get finer resolution (interpolates spectrum)

    for i in range(0, n_signals):
        # Find largest continuous chunk of signal for each channel
        isig = sig[i, :];
        sigtimes = np.where(~np.isnan(isig))[0];  # Non-saturated times
        if np.size(sigtimes) > 0:  # Make sure there is some signal here
            nvals = np.insert(np.cumsum(np.diff(sigtimes) != 1), 0, 0)  # Mark consecutive regions
            sigseg = isig[sigtimes[nvals == stats.mode(nvals)[0]]]  # Find longest consecutive region

            # Find DF of this chunk
            if np.size(sigseg) >= seglength:  # Ensure we hace enough signal
                f, pxx = sigproc.welch(sigseg, nperseg=seglength, noverlap=n_overlap, nfft=nFFT, fs=fs)
                invf = np.divide(1, f);
                pxx = pxx[np.logical_and(invf >= period_min, invf <= period_max)];
                invf = np.delete(invf, np.where(np.logical_or(invf < period_min, invf > period_max)))
                pdval = invf[np.where(pxx == np.max(pxx))[0]];
                mind = np.argmin(np.abs(tvec - pdval))
                pd[i] = mind  # Save PD in units of samples, NOT seconds.
            else:
                # If the lenght of the signal is not long enough, channel discarded.
                print('Not Enough Signal to Take FFT! Channel', i, 'Discarded.');
                sig[i, :] = np.nan;

    # Compute sine recomposition per Kuklik et al (2015 paper). MUUUUCH SLOWER THAN MATLAB: OPTIMIZE
    for i in range(0, n_signals):
        print('Kuklik signal recomposition. Node:', i);
        for t in range(0, siglen - 1):
            if ~np.isnan(pd[i]):
                halfpd = np.round(pd[i] / 2)
                dvdt = diffv[i, t];
                tt = np.arange(-halfpd, halfpd + 1);
                swave = np.sin(2 * np.pi * tt / pd[i]) * np.abs(dvdt) * ((1 - np.sign(dvdt)) / 2)
                twave = t + tt;
                swave = np.delete(swave, np.where(np.logical_or(twave < 0, twave > siglen - 1)))
                twave = np.delete(twave, np.where(twave < 0));
                twave = np.delete(twave, np.where(twave > siglen - 1))
                rsig[i, np.int64(twave)] = rsig[i, np.int64(twave)] + swave
            else:
                rsig[i, :] = np.nan

    # Compute instantaneous phase using Hilbert transform
    instant_phase = np.zeros((n_signals, siglen))
    for i in range(0, n_signals):
        instant_phase[i, :] = -np.arctan2(np.imag(sigproc.hilbert(rsig[i, :])), rsig[i, :])

    DFs = np.divide(fs, pd)
    signals_k = rsig;

    return DFs, signals_k, instant_phase


def botterom_smith_analysis(signals, fs, f1=40, f2=250, f3=20):
    """
    Botterom-Smith signal processing for DF and phase calculation.

    Parameters:
        signals (array): signals to process
        fs (int): sampling frequency
    Returns:
        z_egms : numpy array (n_samples, 1)
            Preprocessed EGM, normalized units.
        P_z : numpy array (n_samples, 1)
            Power Spectral Density estimation of the preprocessed EGM. [units^2/Hz]
        f : numpy array (n_samples, 1)
            Frequency vector
        df : float
            Dominant frequency estimation using Botterom-Smith preprocessing.
    """
    # Convert signals array to list
    signals_list = signals.tolist();

    df = []
    z_egms = []
    P_z = []
    for signal in signals_list:
        z_egm_item, P_z_item, f = fq_bs.botterom_smith_df(signal, fs, f1, f2, f3, plot_flag=False)
        z_egms.append(z_egm_item)
        P_z.append(P_z_item)

        #        idx_df = np.argmax(P_z_item[f<10])
        idx_df = np.argmax(P_z_item)
        df_aux = f[idx_df]
        df.append(df_aux)

    df = np.asarray(df)
    z_egms = np.asarray(z_egms)
    P_z = np.asarray(P_z)

    return z_egms, P_z, f, df


def df_estimation_peak(input_signals, fs):
    """
    Botterom-Smith signal processing for DF and phase calculation.

    Parameters:
        signals (array): signals to process
        fs (int): sampling frequency
    Returns:
        z_egms : numpy array (n_samples, 1)
            Preprocessed EGM, normalized units.
        P_z : numpy array (n_samples, 1)
            Power Spectral Density estimation of the preprocessed EGM. [units^2/Hz]
        f : numpy array (n_samples, 1)
            Frequency vector
        df : float
            Dominant frequency estimation using Botterom-Smith preprocessing.
    """
    # signals = np.zeros(input_signals.shape)
    #
    # # Detrend signals
    # for k in range(0,signals.shape[0]):
    #     signals[k,:],_=fq_bs.detrendSpline(input_signals[k,:],fs)
    #
    # # Filter signals (FPA,fc = 2Hz)
    b, a = sigproc.butter(4, 3, 'hp', fs=fs)
    signals = sigproc.filtfilt(b, a, input_signals, axis=1)

    # signals=input_signals

    # Compute Welch's Periodogram.
    # We take 1/4 seg for each segment, NFFT=2^13, and we don't take into account
    # the start and end segments of each signal.
    # f, pxx = sigproc.welch(signals[:,100:-500],fs=fs,nfft=2**13,nperseg=2048,noverlap=None,axis=1)
    f, pxx = sigproc.welch(signals[:, int(1 * fs):int(5 * fs)], fs=fs, nfft=2 ** 13, nperseg=2048, noverlap=None,
                           axis=1)

    df = []
    for i in range(0, pxx.shape[0]):
        pxx_item = pxx[i, :]
        # idx_4 = (f >= 3.8) & (f<=10)
        idx_4 = f <= 10
        idx_df = np.argmax(pxx_item[idx_4])
        f_df = f[idx_4]
        df_aux = f_df[idx_df]
        df.append(df_aux)

    df = np.asarray(df)

    return f, pxx, df, signals