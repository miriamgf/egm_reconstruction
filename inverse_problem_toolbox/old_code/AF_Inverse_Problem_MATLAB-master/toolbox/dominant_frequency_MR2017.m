function xDF = dominant_frequency_MR2017 (EGM, fs)

%% Preparamos el parpool para paralelizar.
if isempty(gcp('nocreate'))
    parpool;
end

%% Paralelizo con bloques de 100 en 100 y evitar memory overflow.
index_step=1:100:size(EGM,1);
if isempty(find(index_step==size(EGM,1), 1))
    index_step(end+1)=size(EGM,1);
end

%% Assessment:
[N,~] = size(EGM);
xDF = zeros(N,1);

%% Baseline removal
signal_no_baseline = baseline_removal(EGM,fs);

%% LPF
lpf_DF = designfilt('lowpassiir', 'FilterOrder', 10,...
    'HalfPowerFrequency', 10, 'SampleRate', fs);
signal_DF = filtfilt(lpf_DF,signal_no_baseline);

%% DF estimation
window_DF = round(size(signal_DF,2)/8);
n_overlap = round(0.8*window_DF);

for i=1:length(index_step)-1
    parfor j=index_step(i):index_step(i+1)
        fprintf('DF (MR2017). Node %d \n',j);
        [Pff,f] = pwelch(signal_DF(j,:),window_DF,n_overlap,65536,fs);
        [~,idx] = max(Pff);
        xDF(j) = f(idx);
    end
end