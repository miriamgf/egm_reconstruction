function Data_filtered = filter_signal(Lf, Hf, DATA, Fs)
    % This function filters the input signal DATA using a series of filters:
    % 1. High-pass Butterworth filter to remove baseline wandering
    % 2. Low-pass Butterworth filter to remove high-frequency noise
    % 3. Notch filters to remove powerline noise and harmonics
    
    % Parameters
    support_length = 10000; % Length of support zeros added to avoid edge effects
    Q = 80; % Quality factor for notch filters 80
    corr = 1.3; % Correction factor for quality
    frequency = 60; % Powerline frequency
    ast = 120; % Stopband attenuation in dB 120
    order = 6; % Number of harmonics to filter
    
    % Notch filters to remove powerline harmonics
    Data = apply_notch_filters(DATA, Q, corr, frequency, ast, Fs, order);

    % Pre-processing
    Data = [repmat(Data(1), 1, support_length), Data, repmat(Data(end), 1, support_length)];
    
    % High-pass filter to remove baseline wandering
    Data = apply_filter(Data, Lf, Fs, 'high', 20);
    
    % Low-pass filter to remove high-frequency noise
    Data = apply_filter(Data, Hf, Fs, 'low', 10);

    % Remove support zeros and output
    Data_filtered = Data(support_length + 1 : end - support_length);

end


function filtered_data = apply_filter(data, cutoff, Fs, type, order)
    % This function applies a Butterworth filter to the data.
    [z, p, k] = butter(order, cutoff / (Fs / 2), type);
    [sos, g] = zp2sos(z, p, k);
    filtered_data = filtfilt(sos, g, data);
end


function filtered_data = apply_notch_filters(data, Q, corr, frequency, ast, Fs, order)
    % This function applies a series of notch filters to remove harmonics.
    filtered_data = data;
    for i = 1:order
        h1 = fdesign.notch('N,F0,Q,Ast', 6, frequency * i, Q * i * corr, ast, Fs);
        d = design(h1);
        filtered_data = filtfilt(d.sosMatrix, d.ScaleValues, filtered_data);
    end
end

















