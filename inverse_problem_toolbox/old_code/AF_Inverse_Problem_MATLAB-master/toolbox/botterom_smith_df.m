function [z_egm, P_z, f, df] = botterom_smith_df(egm,fs,f1,f2,f3)

%% Compute Botterom and Smith preprocessing of AF EGMs signals.
% Note that this preprocessing makes only sense when working on bipolar 
% EGMs, otherwise the preprocessing might hurt more than help to estimate 
% DF.

%% Parameters
% - egm : numpy array (n_samples, 1) Electrogram signal. It can be in physical or normalized units.
% - fs : int Sampling Frequency in Hz. If it is smaller than 500 Hz, the
% signal is resampled up to 1 Khz.
% - f1 : float Low frequency cut of the band-pass filter.
% - f2 : float Up frequency cut of the band-pass filter.
% - f3 : float frequency cut of the low-pass filter.
% - plot_flag : Boolean (default, True) Allows to plot information about the preprocessing

%%  Returns
% - z_egm : numpy array (n_samples, 1) Preprocessed EGM, normalized units.
% - P_z : numpy array (n_samples, 1) Power Spectral Density estimation of the preprocessed EGM. [units^2/Hz]
% - f : numpy array (n_samples, 1) Frequency vector 
% - df : float Dominant frequency estimation using Botterom-Smith preprocessing.

%%  References
% [Bott-Smith]: Gregory W. Botteron and Joseph M. Smith. A Technique for
% Measurement of the Extent of Spatial Organization of Atrial Activation
% During Atrial Fibrillation in the Intact Human Heart. 
% IEEE TBME, vol 42, num 6 1995

%% Initialize parameters (time vectors and number of segments)
L_s = length(egm);
dur_signal = L_s/fs;
t=0:1/fs:dur_signal-1/fs;

%% Normalize EGM and remove baseline
egm = egm/max(abs(egm));
[egm_d,~] = detrendSpline(egm,fs);

%% Band-pass filtering between f1 and f2 Hz.
[b,a] = butter(3,[f1,f2]*1/(fs/2),'bandpass');
egm_bp = filtfilt(b,a,egm_d);

%% Rectification of the signal
egm_abs = abs(egm_bp);

%% Low-pass filtering
[b1,a1] = butter(3,f3*1/(fs/2),'low');
z_egm = filtfilt(b1,a1,egm_abs);

%% DF estimation
[P_z,f] = pwelch(z_egm,[],[],2048,fs);
Pz_FA=P_z(f>=3 & f<10);
f_FA=f(f>=3 & f<10);
[~,I]=max(Pz_FA);
df=f_FA(I);