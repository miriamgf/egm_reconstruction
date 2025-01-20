%% Loading Files

% Define file paths
signal_file = "../../01 - data/electric_data_Exx_Fxx_Rxx_filtered.mat";
electrodes_idx_file = '../../01 - data/eletrodos_LR.mat';
heart_geo_file = "../../01 - data/../../01 - data/heart_geometry_20000_exp14.mat";
tank_geo_file = "../../01 - data/LR_tank.mat";
mtransfer_file = "../../01 - data/MTransfer_exp14_LR_20000.mat";

%% Reading Files

% Extracting tank signals
signal_data = load(signal_file);
%signal_data = signal_data.(subsref(fieldnames(signal_data), substruct('{}', {1})));
signal_data = signal_data.(subsref(fieldnames(signal_data), substruct('{}', {1})));
signal = signal_data.Data([129:174, 177:190], :); % Keeping only 60 electrodes

% Electrodes
electrodes_data = load(electrodes_idx_file);
electrodes = electrodes_data.(subsref(fieldnames(electrodes_data), substruct('{}', {1})));

% Tank geometry
%tank_data = load(tank_geo_file);
%tank_geo = tank_data.(subsref(fieldnames(tank_data), substruct('{}', {1})));

% Heart geometry
%heart_data = load(heart_geo_file);
%heart_geo = heart_data.(subsref(fieldnames(heart_data), substruct('{}', {1})));

% Transfer Matrix
mtransfer_data = load(mtransfer_file);
MTransfer = mtransfer_data.(subsref(fieldnames(mtransfer_data), substruct('{}', {1})));

% Clear unnecessary variables
clear heart_geo_file tank_geo_file mtransfer_file signal_file electrodes_idx_file heart_data tank_data mtransfer_data electrodes_data;

%% Convert to mV (if needed)
% The signals are exported in microvolts

% Uncomment if conversion is needed
% signal = 0.00019 * signal;
% signal = signal / 1000;

%% Define Estimation Parameters

lambda = 10.^(-0.5:-0.5:-12.5); % Regularization parameter. It will be optmized inside of the regularization code.
order = 0; % Order of regularization (0, 1, or 2)
reg_param_method = 'i'; % global (g) or by instant (i) calculation
compute_params = 1; % 1 if reg_params need to be calculated
model = 'SAF'; % Model type
fs = 4000; % Sampling frequency
SNR = 100; % Signal-to-noise ratio; it will only be used in the l-curve; set it to any number different of 40.

%% Interpolate ECG Signal

y = signal;
idx = electrodes';

[lap, edge] = mesh_laplacian(tank_geo.vertices, tank_geo.faces);
[int, keepindex, repindex] = mesh_laplacian_interp_current(lap, idx);

% Interpolate and keep measured data
interp_signal = int * y;
for i = 1:60
    interp_signal(idx(1, i), :) = y(i, :);
end

%% Define Transfer Matrix

A = MTransfer;

%% Estimation Calculation

% Set estimation time window (in seconds)
est_start = 2;
est_end = 2.1;

% Convert time to samples
est_start_sample = est_start * fs + 1;
est_end_sample = est_end * fs;

% Precompute matrices for the regularization method
[AA, L, LL] = precompute_matrices(A, order, heart_geo);

% Regularization method (comment/uncomment as needed)

% Tikhonov method
%[x_hat, lambda_opt] = tikhonov(A, L, AA, LL, interp_signal(:, est_start_sample:est_end_sample), lambda, SNR, order, reg_param_method, compute_params);

% TSVD method
%[x_hat, k] = tsvd(A, L, interp_signal(:, est_start_sample:est_end_sample), lambda, SNR, order, compute_params);

% DSVD
[x_hat, lambda_opt] = dsvd (A, interp_signal(:, est_start_sample:est_end_sample), lambda, SNR, compute_params);

%% Plot All Signals

figure();
plot(x_hat');
title('ECGi Signals', 'FontSize', 15);
ylabel('Amplitude (\muV)', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Samples', 'FontSize', 15);

%% Plot Vertices in Time

data_est = x_hat; 

% Define vertices to plot
v1 = 1;
v2 = 1000;
v3 = 2900; 
v4 = 7000;

% Define time window for plot (in seconds)
t_start = 1;
t_end = 2;

% Convert time to samples
t_start_sample = t_start * fs + 1;
t_end_sample = t_end * fs;

% Convert to time array
time = linspace(t_start, t_end, length(t_start_sample:t_end_sample));
tline1 = 684 / 4000 + 1;

figure();
subplot(4, 1, 1);
plot(time, data_est(v1, t_start_sample:t_end_sample), 'LineWidth', 1, 'Color', 'black');
title(['Vertex ', num2str(v1)], 'FontSize', 15);
ylabel('Amplitude ($\mu$V)', 'Interpreter', 'latex', 'FontSize', 14);

subplot(4, 1, 2);
plot(time, data_est(v2, t_start_sample:t_end_sample), 'LineWidth', 1, 'Color', 'black');
title(['Vertex ', num2str(v2)], 'FontSize', 15);
ylabel('Amplitude ($\mu$V)', 'Interpreter', 'latex', 'FontSize', 14);

subplot(4, 1, 3);
plot(time, data_est(v3, t_start_sample:t_end_sample), 'LineWidth', 1, 'Color', 'black');
title(['Vertex ', num2str(v3)], 'FontSize', 15);
ylabel('Amplitude ($\mu$V)', 'Interpreter', 'latex', 'FontSize', 14);

subplot(4, 1, 4);
plot(time, data_est(v4, t_start_sample:t_end_sample), 'LineWidth', 1, 'Color', 'black');
title(['Vertex ', num2str(v4)], 'FontSize', 15);
ylabel('Amplitude ($\mu$V)', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Time (s)', 'FontSize', 15);

%% Plot Vertices and Tank signals in Time

% Define tank electrodes to plot
tank_el1 = 130;
tank_el2 = 145;
v3 = 2900; 
v4 = 7000;

% Define time window for plot (in seconds)
t_start = 1;
t_end = 2;

% Convert time to samples
t_start_sample = t_start * fs + 1;
t_end_sample = t_end * fs;

% Convert to time array
time = linspace(t_start, t_end, length(t_start_sample:t_end_sample));
tline1 = 684 / 4000 + 1;


figure();
subplot(4, 1, 1);
%for tank signals, it is necessary to take into account the estimation
%start sample
plot(time, signal_data.EL(tank_el1, (est_start_sample + t_start_sample) : (est_start_sample + t_end_sample)), 'LineWidth', 1, 'Color', 'black');
title(['Tank electrode ', num2str(tank_el1)], 'FontSize', 15);
ylabel('Amplitude ($\mu$V)', 'Interpreter', 'latex', 'FontSize', 14);

subplot(4, 1, 2);
plot(time, signal_data.EL(tank_el2,  (est_start_sample + t_start_sample) : (est_start_sample + t_end_sample)), 'LineWidth', 1, 'Color', 'black');
title(['Tank electrode ', num2str(tank_el2)], 'FontSize', 15);
ylabel('Amplitude ($\mu$V)', 'Interpreter', 'latex', 'FontSize', 14);

subplot(4, 1, 3);
plot(time, data_est(v3, t_start_sample:t_end_sample), 'LineWidth', 1, 'Color', 'black');
title(['Vertex ', num2str(v3)], 'FontSize', 15);
ylabel('Amplitude ($\mu$V)', 'Interpreter', 'latex', 'FontSize', 14);

subplot(4, 1, 4);
plot(time, data_est(v4, t_start_sample:t_end_sample), 'LineWidth', 1, 'Color', 'black');
title(['Vertex ', num2str(v4)], 'FontSize', 15);
ylabel('Amplitude ($\mu$V)', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Time (s)', 'FontSize', 15);

%% Plotting 1 Map

% Define time for plotting (seconds)
inst1 = 3;

% Conversion to samples
inst1_sample = inst1 * fs + 1;

% Organizing geometry
faces = heart_geo.faces;
x = heart_geo.vertices(:,1);
y = heart_geo.vertices(:,2);
z = heart_geo.vertices(:,3);

% Organizing signal
sinal1 = x_hat(:, inst1);

% Plot
figure();
trisurf(faces, x, y, z, sinal1, 'facecolor', 'interp', 'LineStyle', 'none');
grid off; axis off;
title(['Instant ', num2str(inst1), 's  Sample', num2str(inst1_sample)]);
colormap('jet');

% Add colorbar with label
c = colorbar;
c.Label.String = 'Amplitude (\muV)';

%% Plotting 3 Maps

% Define instants for plotting (in seconds)
inst1 = 0.2;
inst2 = 0.3;
inst3 = 0.4;

% Convert to samples
inst1_sample = inst1 * fs + 1;
inst2_sample = inst2 * fs + 1;
inst3_sample = inst3 * fs + 1;

% Organize geometry
faces = heart_geo.faces;
x = heart_geo.vertices(:, 1);
y = heart_geo.vertices(:, 2);
z = heart_geo.vertices(:, 3);

% Organize signal
signal_map1 = x_hat(:, inst1_sample);
signal_map2 = x_hat(:, inst2_sample);
signal_map3 = x_hat(:, inst3_sample);

% Create the tiled layout
tlo = tiledlayout(1, 3);

% Create the subplots and store axes handles
ax1 = nexttile(tlo);
trisurf(faces, x, y, z, signal_map1, 'facecolor', 'interp', 'LineStyle', 'none');
grid off; axis off;
title(['Instant ', num2str(inst1), 's   Sample ', num2str(inst1_sample)]);

ax2 = nexttile(tlo);
trisurf(faces, x, y, z, signal_map2, 'facecolor', 'interp', 'LineStyle', 'none');
grid off; axis off;
title(['Instant ', num2str(inst2), 's   Sample ', num2str(inst2_sample)]);

ax3 = nexttile(tlo);
trisurf(faces, x, y, z, signal_map3, 'facecolor', 'interp', 'LineStyle', 'none');
grid off; axis off;
title(['Instant ', num2str(inst3), 's   Sample ', num2str(inst3_sample)]);

% Comment from here if it's not necessary the same colorbar for all plots
% Set the colormap for the entire figure
colormap(jet);
% 
% % Adjust color limits to be the same for all plots
min_value = min([sinal1(:); sinal2(:); sinal3(:)]);
max_value = max([sinal1(:); sinal2(:); sinal3(:)]);
set([ax1, ax2, ax3], 'CLim', [min_value max_value]);

% % Add a single colorbar for the entire figure
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Amplitude (\muV)';

%% Saving

% Check the regularization method
if exist('lambda_opt', 'var')
    reg_method = 'Tikhonov';
else
    % If not Tikhonov, it was probably TSVD
    reg_method = 'TSVD';
end

% Filename to save
FileName = 'EXX_FXX_RXX';

% Get the current directory
current_dir = pwd;

% Determine the parent directory
[parent_dir, ~, ~] = fileparts(current_dir);

% Create a timestamp
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

% Define the path to save the file in the parent directory
output_file = fullfile(parent_dir, ['estimated_', FileName, '_', timestamp, '.mat']);

% Prepare the data structure
estimated = struct(); % Initialize structure
estimated.Data = x_hat; 
estimated.Time = [init_time, final_time];
estimated.Regularization = reg_method;
estimated.Order = order;
estimated.Heart_geometry = heart_geo;
estimated.Filtering = 'Butterworth(0.5-250Hz)';
estimated.Sync = 'Yes';

% Save the file to the specified path
save(output_file, 'estimated');

fprintf('File saved to: %s\n', output_file);
