%% Reads the Open Ephys files
% fulfilename is the path to the structure.oebin file contained in one of
% the experiments
fullfilename = "E:\HEartLab\Dados exp\electrical_mapping_20230524_E8\01\no_filter\2023-05-24_12-44-17\Record Node 101\experiment1\recording7\structure.oebin";

% Channels to save
channels = [1:10, 15]; % This exemple saves channels 1 to 10 and 15

% Filename to save
FileName = 'ElectricExemple';

DATA = load_open_ephys_binary(fullfilename, 'continuous', 1); % Load the data file
DATA.Data = DATA.Data(channels, :); % Select desired channels
EVENTS = load_open_ephys_binary(fullfilename, 'events', 1); % Load the TTL file

%% Comment this section if conversion to uV is not wanted
bitVolts = 0.1949999928474426; % Conversion factor
DATA.Data = DATA.Data*bitVolts; % Converts the data to uV

%% Convert TTL to appropiate format
% Takes only every other timestamp to record the timing of each rising edge
TTL = EVENTS.Timestamps(1:2:end);

%% Save file to .mat
expData.Data = DATA.Data; 
expData.Timestamps = DATA.Timestamps;
expData.Header = DATA.Header;
expData.Header.channels = channels;
expData.TTL = TTL;

% To add another information to the header
% expData.Header.animal = 'rabbit';
% expData.Header.mass = '3,35 Kg'; 

save(FileName, '-struct', 'expData');

% Clear undesired variables 
clear bitVolts channels DATA EVENTS expData FileName TTL fullfilename

