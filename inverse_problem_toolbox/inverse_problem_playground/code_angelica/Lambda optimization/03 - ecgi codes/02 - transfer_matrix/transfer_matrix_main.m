%% Data Loading

% Define file paths
heart_geo_file = 'C:\Users\HeartLAB\Documents\Documents\Conferences\CinC 2024\Version1\ECGi\Dados\heart_geometry_exp14.mat';
tank_geo_file = 'C:\Users\HeartLAB\Documents\Documents\Conferences\CinC 2024\Version1\ECGi\Dados\tank_geometry.mat';

% Load tank geometry data
tank_data = load(tank_geo_file);
tank_geo = tank_data.(subsref(fieldnames(tank_data), substruct('{}', {1})));

% Load heart geometry data
heart_data = load(heart_geo_file);

% if you are taking the geometry direclty from the 'projections' file, be sure the geometry is centered and not upside down
% heart_geo = heart_data.geometry_20000;

% if you corrected and saved the geometry separately
heart_geo = heart_data.heart_geo;

% Clear unnecessary variables
clear heart_geo_file tank_geo_file heart_data tank_data;

%% Define Conductivity

% Conductivities for blood, tissues, and air, respectively (in S/m)
% Uncomment the appropriate line for the desired conductivities
conductivities = [0.3 0.2 0] * 0.01; % Old
% conductivities = [0 0.2 0] * 0.01; % New

%% Organize Geometries

% Store geometry data in a struct array
Model(1).vertices = heart_geo.vertices;
Model(1).faces = heart_geo.faces;
Model(2).vertices = tank_geo.vertices;
Model(2).faces = tank_geo.faces;

%% Create Transfer Matrix

disp('Calculating Transfer Matrix [Direct Problem]...');

% Calculate the transfer matrix using the BEM method
[MTransfer] = BEM_TransfMat(Model, conductivities);

%% Saving

% Define file path for saving the transfer matrix
transfer_matrix_file = 'transfer_matrix.mat';

% Save the transfer matrix
save(transfer_matrix_file, 'MTransfer');

disp(['Transfer matrix saved to: ', transfer_matrix_file]);