function [x_interp] = interpolacion_nodos (x_inicial,interp_nodes)

%% Script prueba interpolación laplaciana.
clc, close all;

load data/filled_geometry.mat;

%% Cargo caras y vértices del modelo.
faces=atrial_model.faces;
vertices=atrial_model.vertices;

%% Cargo la señal que quiero interpolar.

epic_signals=x_inicial;

% known_nodes_constrained=n_nodos;
% dD=zeros(1,2048);
% Delta_x_epi=floor(2048/known_nodes_constrained);
% dD(1:Delta_x_epi:end) = 1;
% 
% interp_nodes=find(dD);

%% Calculo el laplaciano de la malla triangular y la matriz de interpolación.
[lap,~] = mesh_laplacian(vertices,faces);

[int, keepindex, repindex] = mesh_laplacian_interp(lap, interp_nodes);

%% Realizo la interpolación con los nodos seleccionados.
Vknown=epic_signals(interp_nodes,:);

if isempty(repindex),
  Vint = int * Vknown;
else
  Vint = int * Vknown(keepindex);
end

x_interp=Vint;

end