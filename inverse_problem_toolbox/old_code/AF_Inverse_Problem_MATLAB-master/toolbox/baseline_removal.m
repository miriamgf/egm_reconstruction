function signal_no_baseline = baseline_removal(signal,fs)

%% Signal size.
nL = size(signal,1);
nT = size(signal,2);

dur_signal = nT/fs;
time_fs = linspace(0,dur_signal,nT);

%% Decimation rate.
fs_decim = 12.5;
r=round(fs/fs_decim);

%% LPF and decimation
lpf_baseline = designfilt('lowpassiir', 'FilterOrder', 10,...
    'HalfPowerFrequency', 2, 'SampleRate', fs_decim);
for i=1:nL
    signal_dec = decimate(signal(i,:),r);
    signal_dec = filtfilt(lpf_baseline,signal_dec);
    time_fs_dec = linspace(0,dur_signal,size(signal_dec,2));
    signal_baseline(i,:) = interp1(time_fs_dec,signal_dec,time_fs);
end

%% Baseline removal.
signal_no_baseline = signal-signal_baseline;

end