%% Extraction and filtering of electrical data
clear; clc;


%% Reads the Open Ephys files
% 
% % fulfilename is the path to the structure.oebin file contained in one of
% % the experiments
% fullfilename = "/media/obarquero/Datos/GitHub/egm_reconstruction/inverse_problem_toolbox/inverse_problem_playground/code_angelica/Lambda optimization/01 - data/structure.oebin"; % Put the .oebin path
% 
% % Channels to save
% channels = [1:192];
% 
% DATA = load_open_ephys_binary(fullfilename, 'continuous', 1); % Load the data file
% DATA.Data = DATA.Data(channels, :); % Select desired channels
% EVENTS = load_open_ephys_binary(fullfilename, 'events', 1); % Load the TTL file
% 
% rectime = length(DATA.Data(1, :))/DATA.Header.sample_rate; %s

%% Reads data from .mat
DATA = load("../01 - data/sinus_raw.mat").D_EL;

%% Comment this section if conversion to uV is not wanted
bitVolts = 0.1949999928474426; % Conversion factor
DATA.Data = DATA.Data*bitVolts; % Converts the data to uV


%% Convert TTL to appropiate format
%OJKI: I guess all this information is already in the .mat file.

% Takes only every other timestamp to record the timing of each rising edge
%TTL = EVENTS.Timestamps(1:2:end);
TTL = DATA.Timestamps(1:2:end);

if isempty(TTL)==false
    optictime(1) = (TTL(1) - DATA.Timestamps(1)); %s
    optictime(2) = optictime(1) + 10; %s
end

% Checks if the TTL variable is empty
if isempty(TTL)
    % Prints an alert message if TTL is empty
    disp('Warning, the TTL is empty!');
else
    % Prints a confirmation message if TTL is not empty
    disp('The TTL is not empty.');
end

%% Tank average subtraction
tank_signals = DATA.Data([129:174,177:190], :);
mean_tank = mean(tank_signals);
DATA.Data([129:174,177:190], :) = tank_signals - mean_tank;



%% Comparation Plot

Fs = 4000;

% Default time
if isempty(TTL) % Sem TTL
    ti = 1;
    tf = rectime;
    tline1 = ti;
    tline2 = tf;
else
    ti = optictime(1);
    tf = optictime(2);
    tline1 = (TTL(1) - DATA.Timestamps(1)); % s
    tline2 = tline1 + 10; % s
end

% Specific Time
ti = 2;
tf = 4;
tline1 = ti;
tline2 = tf;

time = linspace(ti, tf, (tf - ti) * Fs + 1);

% Electrodes selection
el1 = 6;  % RA
el2 = 70; % LA
el3 = 26; % V
el4 = 180; % Tank
Index = {'RA', 'LA', 'V','TANK'};

% Filter range (Butterworth)
f_low = 0.5;
f_high = 250;

% Filter Configuration (Wavelet)
waveletType = {'db4'}; % Wavelet type
numLevels = 10; % Number of decomposition levels
reconstructionLevelsSets = {[7, 8, 9, 10]}; % Levels to be used on reconstruction

% Create a figure
figure;
sgtitle('EXP XX - Folder XX - REC XX');

% Plotting using filter_signal function (Butterworth)
for i = 1:4
    subplot(5, 2, 2*i - 1); % Adjusted index for Butterworth
    filteredSignal = filter_signal(f_low, f_high, DATA.Data(eval(['el' num2str(i)]), :), Fs);
    plot(time, filteredSignal(ti * Fs:tf * Fs));
    title(['Eletrodo: ' num2str(eval(['el' num2str(i)])) ' (' Index{i} ') - Butterworth']);
    xlabel("Time (s)");
    ylabel('Amplitude ($\mu$V)', 'Interpreter', 'latex');
    xline(tline1, 'red');
    xline(tline2, 'red');
end

figure();
% Plotting using wavelet_filter function (Wavelet)
for i = 1:4
    subplot(5, 2, 2*i); % Adjusted index for Wavelet
    filteredSignal = wavelet_filter(DATA.Data(eval(['el' num2str(i)]), :), Fs, 1, waveletType, numLevels, reconstructionLevelsSets);
    plot(time, filteredSignal(:, ti * Fs:tf * Fs));
    title(['Eletrodo: ' num2str(eval(['el' num2str(i)])) ' (' Index{i} ') - Wavelet']);
    xlabel("Time (s)");
    ylabel('Amplitude ($\mu$V)', 'Interpreter', 'latex');
    xline(tline1, 'red');
    xline(tline2, 'red');
end

%% Save file to .mat
% Filename to save
FileName = 'Exx_Fxx_Rxx';

% Raw file export
D_EL = struct(); % Initialize structure
D_EL.Data = DATA.Data; 
D_EL.Timestamps = DATA.Timestamps;
D_EL.Header = DATA.Header;
D_EL.Header.channels = channels;
D_EL.TTL = TTL;
D_EL.opticalin = TTL(1) - DATA.Timestamps(1);
save(['electric_data_', FileName, '_raw'], 'D_EL'); 

clear D_EL;

% Defining electrode arrays
el_butter = [1:80, 81:192]; % for Butterworth
el_wavelet = []; % for Wavelet
el_toZero = [33:64, 81:128, 175:176, 191:192]; % to set to zero

% Filter range (Butterworth)
f_low_butter = 0.5;
f_high_butter = 80;
% Filter Configuration (Wavelet)
waveletType = {'db4'}; % Wavelet type
numLevels = 10; % Number of decomposition levels
reconstructionLevelsSets = {[7, 8, 9, 10]}; % Levels to be used on reconstruction

% Initialize filtered data variable
filtered_data = zeros(size(DATA.Data)); % Ensure the size matches the input data

% Filter electrodes with Butterworth
for i = 1:length(el_butter)
    filteredSignal = filter_signal(f_low_butter, f_high_butter, DATA.Data(el_butter(i), :), Fs);
    filtered_data(el_butter(i), :) = filteredSignal;
end

% Filter electrodes with Wavelet
for i = 1:length(el_wavelet)
    filteredSignal = wavelet_filter(DATA.Data(el_wavelet(i), :), Fs, 1, waveletType, numLevels, reconstructionLevelsSets);
    filtered_data(el_wavelet(i), :) = filteredSignal(1:size(filteredSignal,2)-1);
end

% Set specified electrodes to zero
for i = 1:length(el_toZero)
    filtered_data(el_toZero(i), :) = zeros(1, size(filtered_data, 2));
end

% Save filtered data
D_EL = struct(); % Initialize structure
D_EL.Data = filtered_data; 
D_EL.Timestamps = DATA.Timestamps;
D_EL.Header = DATA.Header;
D_EL.Header.channels = channels;
D_EL.TTL = TTL;
D_EL.opticalin = TTL(1) - DATA.Timestamps(1);

% Adding filtering information to the Header
D_EL.Header.FilterParameters = struct();

if ~isempty(el_butter) && all(el_butter > 0) && all(el_butter < 193)
    if all(el_butter > 0) && all(el_butter < 81)
        filter_butter = ["MEA", f_low_butter, f_high_butter];
        filter_wavelet = ["Tank", reconstructionLevelsSets];
        D_EL.Header.FilterParameters.filter_butter = filter_butter;
        D_EL.Header.FilterParameters.filter_wavelet = filter_wavelet;
    else
        if all(el_butter > 80) && all(el_butter < 193)
            filter_butter = ["Tank", f_low_butter, f_high_butter];
            filter_wavelet = ["MEA", reconstructionLevelsSets];
            D_EL.Header.FilterParameters.filter_butter = filter_butter;
            D_EL.Header.FilterParameters.filter_wavelet = filter_wavelet;
        else
            filter_butter = ["MEA", "Tank", f_low_butter, f_high_butter];
            D_EL.Header.FilterParameters.filter_butter = filter_butter;
        end
    end
else
    filter_wavelet = ["MEA", "Tank", reconstructionLevelsSets];
    D_EL.Header.FilterParameters.filter_wavelet = filter_wavelet;
end

% Save the filtered data
save(['electric_data_', FileName, '_filtered'], 'D_EL', '-v7.3');

% Clear unwanted variables
clear bitVolts channels DATA EVENTS D_EL FileName TTL fullfilename
