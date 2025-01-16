%% Data Loading

% Define file paths
heart_geo_file = 'C:\Users\HeartLAB\Documents\Documents\Conferences\CinC 2024\Version1\ECGi\Dados\projected_signals_exp14.mat';
tank_geo_file = 'C:\Users\HeartLAB\Documents\Documents\Conferences\CinC 2024\Version1\ECGi\Dados\tank_geometry.mat';

% Extracting and reading data
tank_data = load(tank_geo_file);
tank_geo = tank_data.(subsref(fieldnames(tank_data), substruct('{}', {1})));

heart_data = load(heart_geo_file);
heart_geo = heart_data.geometry_20000;

% Clear unnecessary variables
clear heart_geo_file tank_geo_file heart_data tank_data;

%% Plot Initial Geometries
% Plot both tank and heart geometries to check initial alignment

figure(1);
trisurf(heart_geo.faces, heart_geo.vertices(:,1), heart_geo.vertices(:,2), heart_geo.vertices(:,3), ...
    'FaceAlpha', 0.2, 'FaceColor', 'y');
hold on;
trisurf(tank_geo.faces, tank_geo.vertices(:,1), tank_geo.vertices(:,2), tank_geo.vertices(:,3), ...
    'FaceAlpha', 0.2, 'FaceColor', 'g');
title('Initial Geometries');
hold off;

%% Correct Upside Down Geometry
% Check if the heart geometry is upside down and correct it if necessary

heart_geo.vertices(:,3) = -heart_geo.vertices(:,3);

%% Correction of Face Indices
% Some heart geometries might contain zero indices. If you can plot both
% geometries without any error, this part is not necessary

for j = 1:3
    for i = 1:length(heart_geo.faces)
        heart_geo.faces(i,j) = heart_geo.faces(i,j) + 1;
    end
end

%% Shifting Heart Geometry to Center of Tank
% Shift heart vertices to center the heart geometry within the tank

vertices_new = zeros(size(heart_geo.vertices));

for dim = 1:3
    if dim == 3
        dc = mean(heart_geo.vertices(:,dim)) - mean(tank_geo.vertices(:,dim));
    else
        dc = mean(heart_geo.vertices(:,dim));
    end

    % Shifting vertices
    for vertice_index = 1:length(heart_geo.vertices)
        vertices_new(vertice_index, dim) = heart_geo.vertices(vertice_index, dim) - dc;
    end
end

% Update heart geometry with shifted vertices
heart_geo = struct('vertices', vertices_new, 'faces', heart_geo.faces);

% Clear temporary variables
clear dim vertice_index dc vertices_new;

%% Plot Corrected Geometries
% Plot both tank and heart geometries after correction

figure(2);
trisurf(heart_geo.faces, heart_geo.vertices(:,1), heart_geo.vertices(:,2), heart_geo.vertices(:,3), ...
    'FaceAlpha', 0.2, 'FaceColor', 'y');
hold on;
trisurf(tank_geo.faces, tank_geo.vertices(:,1), tank_geo.vertices(:,2), tank_geo.vertices(:,3), ...
    'FaceAlpha', 0.2, 'FaceColor', 'g');
title('Corrected Geometries');
hold off;

%% Saving Corrected Geometry
% Define the path to save the corrected structure
adjusted_geo = 'heart_geometry_exp14.mat';

% Save the corrected heart geometry
save(adjusted_geo, 'heart_geo');

disp('Corrected heart geometry saved.');
