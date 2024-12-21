% Loads simuation parameters
% It also loads data set depending on simulation options.
%
% This routine by Vctor Surez Gutirrez (victor.suarez.gutierrez@urjc.es)
%
%
%% Paramters defined in this script:
% fs: sampling frecuency.
% f_low: low cut-off frecuency (default= 11 Hz).
% f_high: high cut-off frecuency (default= 13 Hz).
% params: paramters for each algorithm
% Mtransfer: transfer matrix.
% x: epicardial potentials [NxT]
% lambda_search_values: search grid for reg. param
% Nsamples: number of time instants


%% Common variables:
fs          = 500;       % Sampling frecuency.
Results     = [];        % Initialize.

params1 = 10.^(-4:-0.01:-6);
params2 = 10.^(-0.01:-0.01:-3);
params3 = 10.^(-0.1:-0.1:-12);

%% Low and high cutoff frequencies for bandpass filtering
if ~isempty(strfind(model,'Sinusal'))
    f_low = 0;
else
    f_low = 3;
end
f_high = 30;

%% Load algorithm's parameters
path_pattern = ['/' model '/' model '_Tikhonov_' reg_param_method '_SNR' num2str(SNR)];

if ischar(known_nodes_constrained)
    save_path = [path_pattern sprintf('_%s_%s',SNR_name_cons,known_nodes_constrained)];
else
    save_path = [path_pattern sprintf('_%s_%d_nodes',SNR_name_cons,known_nodes_constrained)];
end 

%% Load geometry, transfer matrix and signals:
load([data_path 'geometry.mat']); 
load([data_path 'transfer.mat']);
load([model_path model '.mat']);

%% Remove samples during initial and end transients
EG = eval(model);
Nsamples_max = 2500;
if length(EG)>2500
    samples = length(EG(1,:))-Nsamples_max-2000+2:length(EG(1,:))-2000;
else
    samples = 2:length(EG(1,:));
end

x = EG(:,samples);    % [NxT]; N: numbers of nodes (triangles vertexs) in epicardium.
clear EG;