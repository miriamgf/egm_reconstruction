function Main(model,known_nodes_constrained,SNR,SNR_cons)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This code implements the experiments for the paper:
%   "Electrocardiographic imaging including intracardiac
%   information to achieve accurate global mapping during
%   atrial fibrillation"
%
% Submitted to Frontiers in Physiology.
%
%               Miguel Ángel Cámara Vázquez & Óscar Barquero Pérez &
%               Felipe Alonso-Atienza & Carlos Figuera Pozuelo
%
%               Sept. 2018
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all, clc;

addpath('toolbox');

%% Options

if nargin==5
    SNR_name_cons=sprintf('SNR_cons%d',SNR_cons);
else
    SNR_name_cons='SNR_consMax';
end

compute_params   = 1;     % {1, 0} If 1, computes de regularization
% parameter and save to parameters.mat file. If
% 0, it loads de regularization parameter from
% parameters.mat file (stored in models folders).

reg_param_method = 'g';  % {'i','g'} Method for computing regularization
% parameter: 'i': recompute for each instant;
% 'g': constant parameter (unique for all time
% instants. Use '' (empty string) for other
% methods.

%% Paths
data_path    = [pwd '/data/'];
model_path   = [pwd '/models/'];
results_path = [pwd '/results/'];

%% Load main simulation parameters
load_parameters;

%% Pre-process potentials and compute forward problem
[x, y, Cn, A] = preprocess (x, MTransfer, model, SNR, fs, f_low, f_high);

if nargin==5
    [x_cons, Cn_x] = addwhitenoise(x, SNR_cons);
else
    x_cons=x;
end

%% Pre-compute DF, phase map and driver for true potentials and A'A, L, L'L and D
% For Tikh-g0
[AA, L_tikhg0, LL_tikhg0] = precompute_matrices (A, 0, atrial_model);
% For Cons-g1
[~, L_consg1, LL_consg1] = precompute_matrices (A, 1, atrial_model);
% For Cons-g2
[~, L_consg2, LL_consg2] = precompute_matrices (A, 2, atrial_model);

if ischar(known_nodes_constrained)
    load([data_path 'Constellation_nodes.mat']);
    clear known_nodes_constrained;
    known_nodes_constrained=constellation_nodes;
end
dD=zeros(1,2048);
if length(known_nodes_constrained)>1
    dD(known_nodes_constrained) = 1;
    D = diag(dD);
elseif length(known_nodes_constrained)==1 && known_nodes_constrained~=0
    Delta_x_epi=floor(2048/known_nodes_constrained);
    dD(1:Delta_x_epi:end) = 1;
    D = diag(dD);
else
    D=dD;
end
Constrained_nodes=find(dD);

%% Ground Truth characteristics
% Dominant Frequency (BS).
[xDF_BS,x_BS] = df_BS(x,fs);
% xDF_BS = importdata([results_path model '/df_TwoRotors_GT_BS.txt']);
% x_BS = importdata([results_path model '/egm_TwoRotors_GT_BS.txt']);
mxDF_BS = mean(xDF_BS);
stdxDF_BS = std(xDF_BS);

% Dominant Frequency (Classical method).
xDF_Classical = dominant_frequency (x, fs);
mxDF_Classical = mean(xDF_Classical);
stdxDF_Classical = std(xDF_Classical);

% Correction Classical_BS
if contains(model,'LSPV')
    [xDF_BS_Corrected,~] = BS_correction_LSPV(x,xDF_Classical,xDF_BS);
    mxDF_BS_Corrected = mean(xDF_BS_Corrected);
    stdxDF_BS_Corrected = std(xDF_BS_Corrected);
elseif contains(model,'TwoRotors')
    [xDF_BS_Corrected,~] = BS_correction_TR(x,xDF_Classical,xDF_BS);
    mxDF_BS_Corrected = mean(xDF_BS_Corrected);
    stdxDF_BS_Corrected = std(xDF_BS_Corrected);
else
    [xDF_BS_Corrected,~] = BS_correction(xDF_Classical,xDF_BS);
    mxDF_BS_Corrected = mean(xDF_BS_Corrected);
    stdxDF_BS_Corrected = std(xDF_BS_Corrected);
end

% Phase (xDF Classical)
xphase_classical = instantphase (x, xDF_Classical, model, fs,1);

% Phase (xDF BS)
xphase_BS = instantphase (x, xDF_BS, model, fs,1);

% Phase (xDF BS_Corrected)
xphase_BS_Corrected = instantphase (x, xDF_BS_Corrected, model, fs,1);

% Phase (noHDF)
xphase_noHDF = instantphase (x, [], model, fs,0);

%% Drivers (BS_Corrected)
if isempty(strfind(model,'Sinusal'))
    % Drivers (Classical)
    %     try
    %         [xdriver_classical, xmodes_classical, xSMF_classical] = driver_location (load([data_path 'filled_geometry.mat']) , x, xDF_Classical, fs,1);
    %     catch ME
    %         disp('Error detectando rotores (xDF Classical)');
    %         xdriver_classical = []; xmodes_classical = []; xSMF_classical = [];
    %     end
    try
        [xdriver_classical_noHDF, xmodes_classical_noHDF, xSMF_classical_noHDF] = driver_location (load([data_path 'filled_geometry.mat']) , x, xDF_Classical, fs,0);
    catch ME
        disp('Error detectando rotores (xDF Classical, noHDF)');
        xdriver_classical_noHDF = []; xmodes_classical_noHDF = []; xSMF_classical_noHDF = [];
    end
    
    % Drivers (BS)
    %     try
    %         [xdriver_BS, xmodes_BS, xSMF_BS] = driver_location (load([data_path 'filled_geometry.mat']) , x, xDF_BS, fs,1);
    %     catch ME
    %         disp('Error detectando rotores (xDF BS)');
    %         xdriver_BS = []; xmodes_BS = []; xSMF_BS = [];
    %     end
    try
        [xdriver_BS_noHDF, xmodes_BS_noHDF, xSMF_BS_noHDF] = driver_location (load([data_path 'filled_geometry.mat']) , x, xDF_BS, fs,0);
    catch ME
        disp('Error detectando rotores (xDF BS, noHDF)');
        xdriver_BS_noHDF = []; xmodes_BS_noHDF = []; xSMF_BS_noHDF = [];
    end
    
    % Drivers (BS_Corrected)
    %     try
    %         [xdriver_BS_Corrected, xmodes_BS_Corrected, xSMF_BS_Corrected] = driver_location (load([data_path 'filled_geometry.mat']) , x, xDF_BS_Corrected, fs,1);
    %     catch ME
    %         disp('Error detectando rotores (xDF BS_Corrected)');
    %         xdriver_BS_Corrected = []; xmodes_BS_Corrected = []; xSMF_BS_Corrected = [];
    %     end
    try
        [xdriver_BS_Corrected_noHDF, xmodes_BS_Corrected_noHDF, xSMF_BS_Corrected_noHDF] = driver_location (load([data_path 'filled_geometry.mat']) , x, xDF_BS_Corrected, fs,0);
    catch ME
        disp('Error detectando rotores (xDF BS_Corrected)');
        xdriver_BS_Corrected_noHDF = []; xmodes_BS_Corrected_noHDF = []; xSMF_BS_Corrected_noHDF = [];
    end
end

% SampEn
% xsampen = sampen_egms(x,fs);
% mxsampen = mean(xsampen);
% stdxsampen = std(xsampen);

% Organization Indexes
% [xRI, xOI, metrics_xRI_xOI] = RI_OI_EGMs (x, fs, xDF);

%% Epicardial potentials
% BS_Corrected Tikhonov (g0)
[x_hat_tikh,lambda_opt_tikh] = constrained_tikhonov (A, AA, L_tikhg0, LL_tikhg0, y, x, params1, 1);
% Constrained Tikhonov (g1)
[x_hat_consg1,lambda_opt_consg1] = constrained_tikhonov (A, AA, L_consg1, LL_consg1, y, x, params1, [], params2, D, x_cons, 1);
% Constrained Tikhonov (g2)
[x_hat_consg2,lambda_opt_consg2] = constrained_tikhonov (A, AA, L_consg2, LL_consg2, y, x, params1, [], params2, D, x_cons, 1);

% if contains(model,'LSPV')
%     x_hat_consg1 = smoothdata(x_hat_consg1);
%     x_hat_consg2 = smoothdata(x_hat_consg2);
% end

% With interpolation
x_interp = interpolacion_nodos (x,Constrained_nodes);

%% Metrics Tikhonov Time-DF-Phase (without constrained nodes).
% Node restriction
selected_nodes=1:2048;
selected_nodes(Constrained_nodes)=[];
selected_nodes(selected_nodes>2039)=[];

% Time metrics
timemetrics_interp = timemetrics_mod (x(1:2039,:)', x_interp(1:2039,:)',selected_nodes);
timemetrics_tikhg0 = timemetrics_mod (x(1:2039,:)', x_hat_tikh(1:2039,:)',selected_nodes);
timemetrics_consg1 = timemetrics_mod (x(1:2039,:)', x_hat_consg1(1:2039,:)',selected_nodes);
timemetrics_consg2 = timemetrics_mod (x(1:2039,:)', x_hat_consg2(1:2039,:)',selected_nodes);

% Dominant frequency and metrics (BS)
[DF_interp_BS, metrics_DF_interp_BS,xhat_interp_BS]=DF_metrics(x_interp, fs, 'BS', xDF_BS, []);
[DF_tikh_BS, metrics_DF_tikh_BS,xhat_tikh_BS]=DF_metrics(x_hat_tikh, fs, 'BS', xDF_BS, []);
[DF_consg1_BS, metrics_DF_consg1_BS,xhat_consg1_BS]=DF_metrics(x_hat_consg1, fs, 'BS', xDF_BS, []);
[DF_consg2_BS, metrics_DF_consg2_BS,xhat_consg2_BS]=DF_metrics(x_hat_consg2, fs, 'BS', xDF_BS, []);

% DF_interp_BS = importdata([results_path model '/df_TwoRotors_interp_BS.txt']);
% xhat_interp_BS = importdata([results_path model '/egm_TwoRotors_interp_BS.txt']);
% [metrics_DF_interp_BS] = DF_metrics_BS_datapython(DF_interp_BS,xDF_BS);
% 
% DF_tikh_BS = importdata([results_path model '/df_TwoRotors_tikh_BS.txt']);
% xhat_tikh_BS = importdata([results_path model '/egm_TwoRotors_tikh_BS.txt']);
% [metrics_DF_tikh_BS] = DF_metrics_BS_datapython(DF_tikh_BS,xDF_BS);
% 
% DF_consg1_BS = importdata([results_path model '/df_TwoRotors_consg1_BS.txt']);
% xhat_consg1_BS = importdata([results_path model '/egm_TwoRotors_consg1_BS.txt']);
% [metrics_DF_consg1_BS] = DF_metrics_BS_datapython(DF_consg1_BS,xDF_BS);
% 
% DF_consg2_BS = importdata([results_path model '/df_TwoRotors_consg2_BS.txt']);
% xhat_consg2_BS = importdata([results_path model '/egm_TwoRotors_consg2_BS.txt']);
% [metrics_DF_consg2_BS] = DF_metrics_BS_datapython(DF_consg2_BS,xDF_BS);

% Dominant frequency and metrics (Classical)
[DF_interp_classical, metrics_DF_interp_classical, ~]=DF_metrics(x_interp, fs, 'Classical', xDF_Classical, []);
[DF_tikh_classical, metrics_DF_tikh_classical, ~]=DF_metrics(x_hat_tikh, fs, 'Classical', xDF_Classical, []);
[DF_consg1_classical, metrics_DF_consg1_classical, ~]=DF_metrics(x_hat_consg1, fs, 'Classical', xDF_Classical, []);
[DF_consg2_classical, metrics_DF_consg2_classical, ~]=DF_metrics(x_hat_consg2, fs, 'Classical', xDF_Classical, []);

% Dominant frequency and metrics (BS_Corrected)
if contains(model,'LSPV')
    [DF_interp_BS_Corrected,metrics_DF_interp_BS_Corrected] = BS_correction_LSPV(x_interp,DF_interp_classical,DF_interp_BS,xDF_BS_Corrected);
    [DF_tikh_BS_Corrected,metrics_DF_tikh_BS_Corrected] = BS_correction_LSPV(x_hat_tikh,DF_tikh_classical,DF_tikh_BS,xDF_BS_Corrected);
    [DF_consg1_BS_Corrected,metrics_DF_consg1_BS_Corrected] = BS_correction_LSPV(x_hat_consg1,DF_consg1_classical,DF_consg2_BS,xDF_BS_Corrected);
    [DF_consg2_BS_Corrected,metrics_DF_consg2_BS_Corrected] = BS_correction_LSPV(x_hat_consg2,DF_consg1_classical,DF_consg2_BS,xDF_BS_Corrected);
elseif contains(model,'TwoRotors')
    [DF_interp_BS_Corrected,metrics_DF_interp_BS_Corrected] = BS_correction_TR(x_interp,DF_interp_classical,DF_interp_BS,xDF_BS_Corrected);
    [DF_tikh_BS_Corrected,metrics_DF_tikh_BS_Corrected] = BS_correction_TR(x_hat_tikh,DF_tikh_classical,DF_tikh_BS,xDF_BS_Corrected);
    [DF_consg1_BS_Corrected,metrics_DF_consg1_BS_Corrected] = BS_correction_TR(x_hat_consg1,DF_consg1_classical,DF_consg2_BS,xDF_BS_Corrected);
    [DF_consg2_BS_Corrected,metrics_DF_consg2_BS_Corrected] = BS_correction_TR(x_hat_consg2,DF_consg1_classical,DF_consg2_BS,xDF_BS_Corrected);
else
    [DF_interp_BS_Corrected, metrics_DF_interp_BS_Corrected]= BS_correction(DF_interp_classical,DF_interp_BS,xDF_BS_Corrected);
    [DF_tikh_BS_Corrected, metrics_DF_tikh_BS_Corrected]= BS_correction(DF_tikh_classical,DF_tikh_BS,xDF_BS_Corrected);
    [DF_consg1_BS_Corrected, metrics_DF_consg1_BS_Corrected]= BS_correction(DF_consg1_classical,DF_consg1_BS,xDF_BS_Corrected);
    [DF_consg2_BS_Corrected, metrics_DF_consg2_BS_Corrected]= BS_correction(DF_consg2_classical,DF_consg2_BS,xDF_BS_Corrected);
end

% DTW
[DTW_interp, metrics_DTW_interp]=DTW_egms(x,x_interp,selected_nodes);
[DTW_tikhg0, metrics_DTW_tikhg0]=DTW_egms(x,x_hat_tikh,selected_nodes);
[DTW_consg1, metrics_DTW_consg1]=DTW_egms(x,x_hat_consg1,selected_nodes);
[DTW_consg2, metrics_DTW_consg2]=DTW_egms(x,x_hat_consg2,selected_nodes);

% Phase maps and metrics (Classical)
[phase_interp_classical, phase_metrics_interp_classical]=phase_metrics(x_interp,DF_interp_classical,model,fs,xphase_classical, selected_nodes,1);
[phase_tikh_classical, phase_metrics_tikh_classical]=phase_metrics(x_hat_tikh,DF_tikh_classical,model,fs,xphase_classical, selected_nodes,1);
[phase_consg1_classical, phase_metrics_consg1_classical]=phase_metrics(x_hat_consg1,DF_consg1_classical,model,fs,xphase_classical, selected_nodes,1);
[phase_consg2_classical, phase_metrics_consg2_classical]=phase_metrics(x_hat_consg2,DF_consg2_classical,model,fs,xphase_classical, selected_nodes,1);

% Phase maps and metrics (BS)
[phase_interp_BS, phase_metrics_interp_BS]=phase_metrics(x_interp,DF_interp_BS,model,fs,xphase_BS, selected_nodes,1);
[phase_tikh_BS, phase_metrics_tikh_BS]=phase_metrics(x_hat_tikh,DF_tikh_BS,model,fs,xphase_BS, selected_nodes,1);
[phase_consg1_BS, phase_metrics_consg1_BS]=phase_metrics(x_hat_consg1,DF_consg1_BS,model,fs,xphase_BS, selected_nodes,1);
[phase_consg2_BS, phase_metrics_consg2_BS]=phase_metrics(x_hat_consg2,DF_consg2_BS,model,fs,xphase_BS, selected_nodes,1);

% Phase maps and metrics (BS_Corrected)
[phase_interp_BS_Corrected, phase_metrics_interp_BS_Corrected]=phase_metrics(x_interp,DF_interp_BS_Corrected,model,fs,xphase_BS_Corrected, selected_nodes,1);
[phase_tikh_BS_Corrected, phase_metrics_tikh_BS_Corrected]=phase_metrics(x_hat_tikh,DF_tikh_BS_Corrected,model,fs,xphase_BS_Corrected, selected_nodes,1);
[phase_consg1_BS_Corrected, phase_metrics_consg1_BS_Corrected]=phase_metrics(x_hat_consg1,DF_consg1_BS_Corrected,model,fs,xphase_BS_Corrected, selected_nodes,1);
[phase_consg2_BS_Corrected, phase_metrics_consg2_BS_Corrected]=phase_metrics(x_hat_consg2,DF_consg2_BS_Corrected,model,fs,xphase_BS_Corrected, selected_nodes,1);

% Phase maps and metrics (noHDF)
[phase_interp_noHDF, phase_metrics_interp_noHDF]=phase_metrics(x_interp,[],model,fs,xphase_noHDF, selected_nodes,0);
[phase_tikh_noHDF, phase_metrics_tikh_noHDF]=phase_metrics(x_hat_tikh,[],model,fs,xphase_noHDF, selected_nodes,0);
[phase_consg1_noHDF, phase_metrics_consg1_noHDF]=phase_metrics(x_hat_consg1,[],model,fs,xphase_noHDF, selected_nodes,0);
[phase_consg2_noHDF, phase_metrics_consg2_noHDF]=phase_metrics(x_hat_consg2,[],model,fs,xphase_noHDF, selected_nodes,0);

% SampEn
% [sampen_interp,samp_metrics_interp] = sampen_metrics(x_interp,fs);
% [sampen_tikh,samp_metrics_tikh] = sampen_metrics(x_hat_tikh,fs);
% [sampen_consg1,samp_metrics_consg1] = sampen_metrics(x_hat_consg1,fs);
% [sampen_consg2,samp_metrics_consg2] = sampen_metrics(x_hat_consg2,fs);

% Organization Indexes
% [RI_interp, OI_interp, metrics_RI_OI_interp] = RI_OI_EGMs (x_interp, fs, DF_interp);
% [RI_tikh, OI_tikh, metrics_RI_OI_tikh] = RI_OI_EGMs (x_hat_tikh, fs, DF_tikh);
% [RI_consg1, OI_consg1, metrics_RI_OI_consg1] = RI_OI_EGMs (x_hat_consg1, fs, DF_consg1);
% [RI_consg2, OI_consg2, metrics_RI_OI_consg2] = RI_OI_EGMs (x_hat_consg2, fs, DF_consg2);

% Table with metrics
table_metrics{1,1}=model;
table_metrics{1,2}='Interpolation';
table_metrics{1,3}='Tikh-g0';
table_metrics{1,4}='Cons-g1';
table_metrics{1,5}='Cons-g2';
table_metrics{2,1}='RDMS';
table_metrics{2,2}=sprintf('%.2f±%.2f',timemetrics_interp.MRDMS,timemetrics_interp.stdRDMS);
table_metrics{2,3}=sprintf('%.2f±%.2f',timemetrics_tikhg0.MRDMS,timemetrics_tikhg0.stdRDMS);
table_metrics{2,4}=sprintf('%.2f±%.2f',timemetrics_consg1.MRDMS,timemetrics_consg1.stdRDMS);
table_metrics{2,5}=sprintf('%.2f±%.2f',timemetrics_consg2.MRDMS,timemetrics_consg2.stdRDMS);
table_metrics{3,1}='CC';
table_metrics{3,2}=sprintf('%.2f±%.2f',timemetrics_interp.MCC,timemetrics_interp.stdCC);
table_metrics{3,3}=sprintf('%.2f±%.2f',timemetrics_tikhg0.MCC,timemetrics_tikhg0.stdCC);
table_metrics{3,4}=sprintf('%.2f±%.2f',timemetrics_consg1.MCC,timemetrics_consg1.stdCC);
table_metrics{3,5}=sprintf('%.2f±%.2f',timemetrics_consg2.MCC,timemetrics_consg2.stdCC);
table_metrics{4,1}='DTW';
table_metrics{4,2}=sprintf('%.2f±%.2f',metrics_DTW_interp.mDTW,metrics_DTW_interp.stdDTW);
table_metrics{4,3}=sprintf('%.2f±%.2f',metrics_DTW_tikhg0.mDTW,metrics_DTW_tikhg0.stdDTW);
table_metrics{4,4}=sprintf('%.2f±%.2f',metrics_DTW_consg1.mDTW,metrics_DTW_consg1.stdDTW);
table_metrics{4,5}=sprintf('%.2f±%.2f',metrics_DTW_consg2.mDTW,metrics_DTW_consg2.stdDTW);
table_metrics{5,1}='Phase RDMS (Classical)';
table_metrics{5,2}=sprintf('%.2f±%.2f',phase_metrics_interp_classical.MRDMSt_phase,phase_metrics_interp_classical.std_MRDMSt_phase);
table_metrics{5,3}=sprintf('%.2f±%.2f',phase_metrics_tikh_classical.MRDMSt_phase,phase_metrics_tikh_classical.std_MRDMSt_phase);
table_metrics{5,4}=sprintf('%.2f±%.2f',phase_metrics_consg1_classical.MRDMSt_phase,phase_metrics_consg1_classical.std_MRDMSt_phase);
table_metrics{5,5}=sprintf('%.2f±%.2f',phase_metrics_consg2_classical.MRDMSt_phase,phase_metrics_consg2_classical.std_MRDMSt_phase);
table_metrics{6,1}='Phase CC (Classical)';
table_metrics{6,2}=sprintf('%.2f±%.2f',phase_metrics_interp_classical.MCCt_phase,phase_metrics_interp_classical.std_MCCt_phase);
table_metrics{6,3}=sprintf('%.2f±%.2f',phase_metrics_tikh_classical.MCCt_phase,phase_metrics_tikh_classical.std_MCCt_phase);
table_metrics{6,4}=sprintf('%.2f±%.2f',phase_metrics_consg1_classical.MCCt_phase,phase_metrics_consg1_classical.std_MCCt_phase);
table_metrics{6,5}=sprintf('%.2f±%.2f',phase_metrics_consg2_classical.MCCt_phase,phase_metrics_consg2_classical.std_MCCt_phase);
table_metrics{7,1}='Phase RDMS (BS)';
table_metrics{7,2}=sprintf('%.2f±%.2f',phase_metrics_interp_BS.MRDMSt_phase,phase_metrics_interp_BS.std_MRDMSt_phase);
table_metrics{7,3}=sprintf('%.2f±%.2f',phase_metrics_tikh_BS.MRDMSt_phase,phase_metrics_tikh_BS.std_MRDMSt_phase);
table_metrics{7,4}=sprintf('%.2f±%.2f',phase_metrics_consg1_BS.MRDMSt_phase,phase_metrics_consg1_BS.std_MRDMSt_phase);
table_metrics{7,5}=sprintf('%.2f±%.2f',phase_metrics_consg2_BS.MRDMSt_phase,phase_metrics_consg2_BS.std_MRDMSt_phase);
table_metrics{8,1}='Phase CC (BS)';
table_metrics{8,2}=sprintf('%.2f±%.2f',phase_metrics_interp_BS.MCCt_phase,phase_metrics_interp_BS.std_MCCt_phase);
table_metrics{8,3}=sprintf('%.2f±%.2f',phase_metrics_tikh_BS.MCCt_phase,phase_metrics_tikh_BS.std_MCCt_phase);
table_metrics{8,4}=sprintf('%.2f±%.2f',phase_metrics_consg1_BS.MCCt_phase,phase_metrics_consg1_BS.std_MCCt_phase);
table_metrics{8,5}=sprintf('%.2f±%.2f',phase_metrics_consg2_BS.MCCt_phase,phase_metrics_consg2_BS.std_MCCt_phase);
table_metrics{9,1}='Phase RDMS (BS Corrected)';
table_metrics{9,2}=sprintf('%.2f±%.2f',phase_metrics_interp_BS_Corrected.MRDMSt_phase,phase_metrics_interp_BS_Corrected.std_MRDMSt_phase);
table_metrics{9,3}=sprintf('%.2f±%.2f',phase_metrics_tikh_BS_Corrected.MRDMSt_phase,phase_metrics_tikh_BS_Corrected.std_MRDMSt_phase);
table_metrics{9,4}=sprintf('%.2f±%.2f',phase_metrics_consg1_BS_Corrected.MRDMSt_phase,phase_metrics_consg1_BS_Corrected.std_MRDMSt_phase);
table_metrics{9,5}=sprintf('%.2f±%.2f',phase_metrics_consg2_BS_Corrected.MRDMSt_phase,phase_metrics_consg2_BS_Corrected.std_MRDMSt_phase);
table_metrics{10,1}='Phase CC (BS Corrected)';
table_metrics{10,2}=sprintf('%.2f±%.2f',phase_metrics_interp_BS_Corrected.MCCt_phase,phase_metrics_interp_BS_Corrected.std_MCCt_phase);
table_metrics{10,3}=sprintf('%.2f±%.2f',phase_metrics_tikh_BS_Corrected.MCCt_phase,phase_metrics_tikh_BS_Corrected.std_MCCt_phase);
table_metrics{10,4}=sprintf('%.2f±%.2f',phase_metrics_consg1_BS_Corrected.MCCt_phase,phase_metrics_consg1_BS_Corrected.std_MCCt_phase);
table_metrics{10,5}=sprintf('%.2f±%.2f',phase_metrics_consg2_BS_Corrected.MCCt_phase,phase_metrics_consg2_BS_Corrected.std_MCCt_phase);
table_metrics{11,1}='Phase RDMS (noHDF)';
table_metrics{11,2}=sprintf('%.2f±%.2f',phase_metrics_interp_noHDF.MRDMSt_phase,phase_metrics_interp_noHDF.std_MRDMSt_phase);
table_metrics{11,3}=sprintf('%.2f±%.2f',phase_metrics_tikh_noHDF.MRDMSt_phase,phase_metrics_tikh_noHDF.std_MRDMSt_phase);
table_metrics{11,4}=sprintf('%.2f±%.2f',phase_metrics_consg1_noHDF.MRDMSt_phase,phase_metrics_consg1_noHDF.std_MRDMSt_phase);
table_metrics{11,5}=sprintf('%.2f±%.2f',phase_metrics_consg2_noHDF.MRDMSt_phase,phase_metrics_consg2_noHDF.std_MRDMSt_phase);
table_metrics{12,1}='Phase CC (noHDF)';
table_metrics{12,2}=sprintf('%.2f±%.2f',phase_metrics_interp_noHDF.MCCt_phase,phase_metrics_interp_noHDF.std_MCCt_phase);
table_metrics{12,3}=sprintf('%.2f±%.2f',phase_metrics_tikh_noHDF.MCCt_phase,phase_metrics_tikh_noHDF.std_MCCt_phase);
table_metrics{12,4}=sprintf('%.2f±%.2f',phase_metrics_consg1_noHDF.MCCt_phase,phase_metrics_consg1_noHDF.std_MCCt_phase);
table_metrics{12,5}=sprintf('%.2f±%.2f',phase_metrics_consg2_noHDF.MCCt_phase,phase_metrics_consg2_noHDF.std_MCCt_phase);

% Table with EGM characteristics
table_characteristics{1,1}=model;
table_characteristics{1,2}='Ground Truth';
table_characteristics{1,3}='Interpolation';
table_characteristics{1,4}='Tikh-g0';
table_characteristics{1,5}='Cons-g1';
table_characteristics{1,6}='Cons-g2';
table_characteristics{2,1}='DF (Classical)';
table_characteristics{2,2}=sprintf('%.2f±%.2f',mxDF_Classical,stdxDF_Classical);
table_characteristics{2,3}=sprintf('%.2f±%.2f',metrics_DF_interp_classical.mDF,metrics_DF_interp_classical.stdDF);
table_characteristics{2,4}=sprintf('%.2f±%.2f',metrics_DF_tikh_classical.mDF,metrics_DF_tikh_classical.stdDF);
table_characteristics{2,5}=sprintf('%.2f±%.2f',metrics_DF_consg1_classical.mDF,metrics_DF_consg1_classical.stdDF);
table_characteristics{2,6}=sprintf('%.2f±%.2f',metrics_DF_consg2_classical.mDF,metrics_DF_consg2_classical.stdDF);
table_characteristics{3,1}='DF (BS)';
table_characteristics{3,2}=sprintf('%.2f±%.2f',mxDF_BS,stdxDF_BS);
table_characteristics{3,3}=sprintf('%.2f±%.2f',metrics_DF_interp_BS.mDF,metrics_DF_interp_BS.stdDF);
table_characteristics{3,4}=sprintf('%.2f±%.2f',metrics_DF_tikh_BS.mDF,metrics_DF_tikh_BS.stdDF);
table_characteristics{3,5}=sprintf('%.2f±%.2f',metrics_DF_consg1_BS.mDF,metrics_DF_consg1_BS.stdDF);
table_characteristics{3,6}=sprintf('%.2f±%.2f',metrics_DF_consg2_BS.mDF,metrics_DF_consg2_BS.stdDF);
table_characteristics{4,1}='DF (BS Corrected)';
table_characteristics{4,2}=sprintf('%.2f±%.2f',mxDF_BS_Corrected,stdxDF_BS_Corrected);
table_characteristics{4,3}=sprintf('%.2f±%.2f',metrics_DF_interp_BS_Corrected.mDF,metrics_DF_interp_BS_Corrected.stdDF);
table_characteristics{4,4}=sprintf('%.2f±%.2f',metrics_DF_tikh_BS_Corrected.mDF,metrics_DF_tikh_BS_Corrected.stdDF);
table_characteristics{4,5}=sprintf('%.2f±%.2f',metrics_DF_consg1_BS_Corrected.mDF,metrics_DF_consg1_BS_Corrected.stdDF);
table_characteristics{4,6}=sprintf('%.2f±%.2f',metrics_DF_consg2_BS_Corrected.mDF,metrics_DF_consg2_BS_Corrected.stdDF);
% table_characteristics{3,1}='SampEn';
% table_characteristics{3,2}=sprintf('%.2f±%.2f',mxsampen,stdxsampen);
% table_characteristics{3,3}=sprintf('%.2f±%.2f',samp_metrics_interp.mSampEn,samp_metrics_interp.stdSampEn);
% table_characteristics{3,4}=sprintf('%.2f±%.2f',samp_metrics_tikh.mSampEn,samp_metrics_tikh.stdSampEn);
% table_characteristics{3,5}=sprintf('%.2f±%.2f',samp_metrics_consg1.mSampEn,samp_metrics_consg1.stdSampEn);
% table_characteristics{3,6}=sprintf('%.2f±%.2f',samp_metrics_consg2.mSampEn,samp_metrics_consg2.stdSampEn);
% table_characteristics{4,1}='OI';
% table_characteristics{4,2}=sprintf('%.2f±%.2f',metrics_xRI_xOI.mOI,metrics_xRI_xOI.stdOI);
% table_characteristics{4,3}=sprintf('%.2f±%.2f',metrics_RI_OI_interp.mOI,metrics_RI_OI_interp.stdOI);
% table_characteristics{4,4}=sprintf('%.2f±%.2f',metrics_RI_OI_tikh.mOI,metrics_RI_OI_tikh.stdOI);
% table_characteristics{4,5}=sprintf('%.2f±%.2f',metrics_RI_OI_consg1.mOI,metrics_RI_OI_consg1.stdOI);
% table_characteristics{4,6}=sprintf('%.2f±%.2f',metrics_RI_OI_consg2.mOI,metrics_RI_OI_consg2.stdOI);
% table_characteristics{5,1}='RI';
% table_characteristics{5,2}=sprintf('%.2f±%.2f',metrics_xRI_xOI.mRI,metrics_xRI_xOI.stdRI);
% table_characteristics{5,3}=sprintf('%.2f±%.2f',metrics_RI_OI_interp.mRI,metrics_RI_OI_interp.stdRI);
% table_characteristics{5,4}=sprintf('%.2f±%.2f',metrics_RI_OI_tikh.mRI,metrics_RI_OI_tikh.stdRI);
% table_characteristics{5,5}=sprintf('%.2f±%.2f',metrics_RI_OI_consg1.mRI,metrics_RI_OI_consg1.stdRI);
% table_characteristics{5,6}=sprintf('%.2f±%.2f',metrics_RI_OI_consg2.mRI,metrics_RI_OI_consg2.stdRI);

%% Statistical analysis interp vs Tikh-based (without constrained nodes)
% RDMS
p_value_RDMS_g0 = statistical_analysis (timemetrics_interp.RDMSt,timemetrics_tikhg0.RDMSt,selected_nodes);
p_value_RDMS_g1 = statistical_analysis (timemetrics_interp.RDMSt,timemetrics_consg1.RDMSt,selected_nodes);
p_value_RDMS_g2 = statistical_analysis (timemetrics_interp.RDMSt,timemetrics_consg2.RDMSt,selected_nodes);

% CC
p_value_CC_g0 = statistical_analysis (timemetrics_interp.CCt,timemetrics_tikhg0.CCt,selected_nodes);
p_value_CC_g1 = statistical_analysis (timemetrics_interp.CCt,timemetrics_consg1.CCt,selected_nodes);
p_value_CC_g2 = statistical_analysis (timemetrics_interp.CCt,timemetrics_consg2.CCt,selected_nodes);

% RDMSt_phase (Classical)
p_value_phase_RDMS_g0_classical = statistical_analysis (phase_metrics_interp_classical.RDMSt_phase,phase_metrics_tikh_classical.RDMSt_phase,selected_nodes);
p_value_phase_RDMS_g1_classical = statistical_analysis (phase_metrics_interp_classical.RDMSt_phase,phase_metrics_consg1_classical.RDMSt_phase,selected_nodes);
p_value_phase_RDMS_g2_classical = statistical_analysis (phase_metrics_interp_classical.RDMSt_phase,phase_metrics_consg2_classical.RDMSt_phase,selected_nodes);

% CCt_phase (Classical)
p_value_phase_CC_g0_classical = statistical_analysis (phase_metrics_interp_classical.CCt_phase,phase_metrics_tikh_classical.CCt_phase,selected_nodes);
p_value_phase_CC_g1_classical = statistical_analysis (phase_metrics_interp_classical.CCt_phase,phase_metrics_consg1_classical.CCt_phase,selected_nodes);
p_value_phase_CC_g2_classical = statistical_analysis (phase_metrics_interp_classical.CCt_phase,phase_metrics_consg2_classical.CCt_phase,selected_nodes);

% RDMSt_phase (BS)
p_value_phase_RDMS_g0_BS = statistical_analysis (phase_metrics_interp_BS.RDMSt_phase,phase_metrics_tikh_BS.RDMSt_phase,selected_nodes);
p_value_phase_RDMS_g1_BS = statistical_analysis (phase_metrics_interp_BS.RDMSt_phase,phase_metrics_consg1_BS.RDMSt_phase,selected_nodes);
p_value_phase_RDMS_g2_BS = statistical_analysis (phase_metrics_interp_BS.RDMSt_phase,phase_metrics_consg2_BS.RDMSt_phase,selected_nodes);

% CCt_phase (BS)
p_value_phase_CC_g0_BS = statistical_analysis (phase_metrics_interp_BS.CCt_phase,phase_metrics_tikh_BS.CCt_phase,selected_nodes);
p_value_phase_CC_g1_BS = statistical_analysis (phase_metrics_interp_BS.CCt_phase,phase_metrics_consg1_BS.CCt_phase,selected_nodes);
p_value_phase_CC_g2_BS = statistical_analysis (phase_metrics_interp_BS.CCt_phase,phase_metrics_consg2_BS.CCt_phase,selected_nodes);

% RDMSt_phase (BS_Corrected)
p_value_phase_RDMS_g0_BS_Corrected = statistical_analysis (phase_metrics_interp_BS_Corrected.RDMSt_phase,phase_metrics_tikh_BS_Corrected.RDMSt_phase,selected_nodes);
p_value_phase_RDMS_g1_BS_Corrected = statistical_analysis (phase_metrics_interp_BS_Corrected.RDMSt_phase,phase_metrics_consg1_BS_Corrected.RDMSt_phase,selected_nodes);
p_value_phase_RDMS_g2_BS_Corrected = statistical_analysis (phase_metrics_interp_BS_Corrected.RDMSt_phase,phase_metrics_consg2_BS_Corrected.RDMSt_phase,selected_nodes);

% CCt_phase (BS_Corrected)
p_value_phase_CC_g0_BS_Corrected = statistical_analysis (phase_metrics_interp_BS_Corrected.CCt_phase,phase_metrics_tikh_BS_Corrected.CCt_phase,selected_nodes);
p_value_phase_CC_g1_BS_Corrected = statistical_analysis (phase_metrics_interp_BS_Corrected.CCt_phase,phase_metrics_consg1_BS_Corrected.CCt_phase,selected_nodes);
p_value_phase_CC_g2_BS_Corrected = statistical_analysis (phase_metrics_interp_BS_Corrected.CCt_phase,phase_metrics_consg2_BS_Corrected.CCt_phase,selected_nodes);

% RDMSt_phase (noHDF)
p_value_phase_RDMS_g0_noHDF = statistical_analysis (phase_metrics_interp_noHDF.RDMSt_phase,phase_metrics_tikh_noHDF.RDMSt_phase,selected_nodes);
p_value_phase_RDMS_g1_noHDF = statistical_analysis (phase_metrics_interp_noHDF.RDMSt_phase,phase_metrics_consg1_noHDF.RDMSt_phase,selected_nodes);
p_value_phase_RDMS_g2_noHDF = statistical_analysis (phase_metrics_interp_noHDF.RDMSt_phase,phase_metrics_consg2_noHDF.RDMSt_phase,selected_nodes);

% CCt_phase (noHDF)
p_value_phase_CC_g0_noHDF = statistical_analysis (phase_metrics_interp_noHDF.CCt_phase,phase_metrics_tikh_noHDF.CCt_phase,selected_nodes);
p_value_phase_CC_g1_noHDF = statistical_analysis (phase_metrics_interp_noHDF.CCt_phase,phase_metrics_consg1_noHDF.CCt_phase,selected_nodes);
p_value_phase_CC_g2_noHDF = statistical_analysis (phase_metrics_interp_noHDF.CCt_phase,phase_metrics_consg2_noHDF.CCt_phase,selected_nodes);

% % DTW
p_value_DTW_g0 = statistical_analysis (DTW_interp,DTW_tikhg0,selected_nodes);
p_value_DTW_g1 = statistical_analysis (DTW_interp,DTW_consg1,selected_nodes);
p_value_DTW_g2 = statistical_analysis (DTW_interp,DTW_consg2,selected_nodes);

% DF (Classical)
p_value_DF_interp_classical = statistical_analysis (xDF_Classical,DF_interp_classical,[]);
p_value_DF_g0_classical = statistical_analysis (xDF_Classical,DF_tikh_classical,[]);
p_value_DF_g1_classical = statistical_analysis (xDF_Classical,DF_consg1_classical,[]);
p_value_DF_g2_classical = statistical_analysis (xDF_Classical,DF_consg2_classical,[]);

% DF (BS)
p_value_DF_interp_BS = statistical_analysis (xDF_BS,DF_interp_BS,[]);
p_value_DF_g0_BS = statistical_analysis (xDF_BS,DF_tikh_BS,[]);
p_value_DF_g1_BS = statistical_analysis (xDF_BS,DF_consg1_BS,[]);
p_value_DF_g2_BS = statistical_analysis (xDF_BS,DF_consg2_BS,[]);

% DF (BS_Corrected)
p_value_DF_interp_BS_Corrected = statistical_analysis (xDF_BS_Corrected,DF_interp_BS_Corrected,[]);
p_value_DF_g0_BS_Corrected = statistical_analysis (xDF_BS_Corrected,DF_tikh_BS_Corrected,[]);
p_value_DF_g1_BS_Corrected = statistical_analysis (xDF_BS_Corrected,DF_consg1_BS_Corrected,[]);
p_value_DF_g2_BS_Corrected = statistical_analysis (xDF_BS_Corrected,DF_consg2_BS_Corrected,[]);

% % SampEn
% p_value_sampen_interp = statistical_analysis (xsampen,sampen_interp,[]);
% p_value_sampen_g0 = statistical_analysis (xsampen,sampen_tikh,[]);
% p_value_sampen_g1 = statistical_analysis (xsampen,sampen_consg1,[]);
% p_value_sampen_g2 = statistical_analysis (xsampen,sampen_consg2,[]);
%
% % OI
% p_value_OI_interp = statistical_analysis (xOI,OI_interp,[]);
% p_value_OI_g0 = statistical_analysis (xOI,OI_tikh,[]);
% p_value_OI_g1 = statistical_analysis (xOI,OI_consg1,[]);
% p_value_OI_g2 = statistical_analysis (xOI,OI_consg2,[]);
%
% % RI
% p_value_RI_interp = statistical_analysis (xRI,RI_interp,[]);
% p_value_RI_g0 = statistical_analysis (xRI,RI_tikh,[]);
% p_value_RI_g1 = statistical_analysis (xRI,RI_consg1,[]);
% p_value_RI_g2 = statistical_analysis (xRI,RI_consg2,[]);

% Table with p-values (metrics)
table_pvalues_metrics{1,1}=model;
table_pvalues_metrics{1,2}='Interp_Tikh-g0';
table_pvalues_metrics{1,3}='Interp_Cons-g1';
table_pvalues_metrics{1,4}='Interp_Cons-g2';
table_pvalues_metrics{2,1}='RDMS';
table_pvalues_metrics{2,2}=p_value_RDMS_g0;
table_pvalues_metrics{2,3}=p_value_RDMS_g1;
table_pvalues_metrics{2,4}=p_value_RDMS_g2;
table_pvalues_metrics{3,1}='CC';
table_pvalues_metrics{3,2}=p_value_CC_g0;
table_pvalues_metrics{3,3}=p_value_CC_g1;
table_pvalues_metrics{3,4}=p_value_CC_g2;
table_pvalues_metrics{4,1}='DTW';
table_pvalues_metrics{4,2}=p_value_DTW_g0;
table_pvalues_metrics{4,3}=p_value_DTW_g1;
table_pvalues_metrics{4,4}=p_value_DTW_g2;
table_pvalues_metrics{5,1}='RDMS phase (Classical)';
table_pvalues_metrics{5,2}=p_value_phase_RDMS_g0_classical;
table_pvalues_metrics{5,3}=p_value_phase_RDMS_g1_classical;
table_pvalues_metrics{5,4}=p_value_phase_RDMS_g2_classical;
table_pvalues_metrics{6,1}='CC phase (Classical)';
table_pvalues_metrics{6,2}=p_value_phase_CC_g0_classical;
table_pvalues_metrics{6,3}=p_value_phase_CC_g1_classical;
table_pvalues_metrics{6,4}=p_value_phase_CC_g2_classical;
table_pvalues_metrics{7,1}='RDMS phase (BS)';
table_pvalues_metrics{7,2}=p_value_phase_RDMS_g0_BS;
table_pvalues_metrics{7,3}=p_value_phase_RDMS_g1_BS;
table_pvalues_metrics{7,4}=p_value_phase_RDMS_g2_BS;
table_pvalues_metrics{8,1}='CC phase (BS)';
table_pvalues_metrics{8,2}=p_value_phase_CC_g0_BS;
table_pvalues_metrics{8,3}=p_value_phase_CC_g1_BS;
table_pvalues_metrics{8,4}=p_value_phase_CC_g2_BS;
table_pvalues_metrics{9,1}='RDMS phase (BS Corrected)';
table_pvalues_metrics{9,2}=p_value_phase_RDMS_g0_BS_Corrected;
table_pvalues_metrics{9,3}=p_value_phase_RDMS_g1_BS_Corrected;
table_pvalues_metrics{9,4}=p_value_phase_RDMS_g2_BS_Corrected;
table_pvalues_metrics{10,1}='CC phase (BS Corrected)';
table_pvalues_metrics{10,2}=p_value_phase_CC_g0_BS_Corrected;
table_pvalues_metrics{10,3}=p_value_phase_CC_g1_BS_Corrected;
table_pvalues_metrics{10,4}=p_value_phase_CC_g2_BS_Corrected;
table_pvalues_metrics{11,1}='RDMS phase (noHDF)';
table_pvalues_metrics{11,2}=p_value_phase_RDMS_g0_noHDF;
table_pvalues_metrics{11,3}=p_value_phase_RDMS_g1_noHDF;
table_pvalues_metrics{11,4}=p_value_phase_RDMS_g2_noHDF;
table_pvalues_metrics{12,1}='CC phase (noHDF)';
table_pvalues_metrics{12,2}=p_value_phase_CC_g0_noHDF;
table_pvalues_metrics{12,3}=p_value_phase_CC_g1_noHDF;
table_pvalues_metrics{12,4}=p_value_phase_CC_g2_noHDF;

% Table with p-values (characteristics)
table_pvalues_characteristics{1,1}=model;
table_pvalues_characteristics{1,2}='GroundTruth_Interp';
table_pvalues_characteristics{1,3}='GroundTruth_Tikh-g0';
table_pvalues_characteristics{1,4}='GroundTruth_Cons-g1';
table_pvalues_characteristics{1,5}='GroundTruth_Cons-g2';
table_pvalues_characteristics{2,1}='DF (Classical)';
table_pvalues_characteristics{2,2}=p_value_DF_interp_classical;
table_pvalues_characteristics{2,3}=p_value_DF_g0_classical;
table_pvalues_characteristics{2,4}=p_value_DF_g1_classical;
table_pvalues_characteristics{2,5}=p_value_DF_g2_classical;
table_pvalues_characteristics{3,1}='DF (BS)';
table_pvalues_characteristics{3,2}=p_value_DF_interp_BS;
table_pvalues_characteristics{3,3}=p_value_DF_g0_BS;
table_pvalues_characteristics{3,4}=p_value_DF_g1_BS;
table_pvalues_characteristics{3,5}=p_value_DF_g2_BS;
table_pvalues_characteristics{4,1}='DF (BS Corrected)';
table_pvalues_characteristics{4,2}=p_value_DF_interp_BS_Corrected;
table_pvalues_characteristics{4,3}=p_value_DF_g0_BS_Corrected;
table_pvalues_characteristics{4,4}=p_value_DF_g1_BS_Corrected;
table_pvalues_characteristics{4,5}=p_value_DF_g2_BS_Corrected;

% table_pvalues_characteristics{3,1}='SampEn';
% table_pvalues_characteristics{3,2}=p_value_sampen_interp;
% table_pvalues_characteristics{3,3}=p_value_sampen_g0;
% table_pvalues_characteristics{3,4}=p_value_sampen_g1;
% table_pvalues_characteristics{3,5}=p_value_sampen_g2;
% table_pvalues_characteristics{4,1}='OI';
% table_pvalues_characteristics{4,2}=p_value_OI_interp;
% table_pvalues_characteristics{4,3}=p_value_OI_g0;
% table_pvalues_characteristics{4,4}=p_value_OI_g1;
% table_pvalues_characteristics{4,5}=p_value_OI_g2;
% table_pvalues_characteristics{5,1}='RI';
% table_pvalues_characteristics{5,2}=p_value_RI_interp;
% table_pvalues_characteristics{5,3}=p_value_RI_g0;
% table_pvalues_characteristics{5,4}=p_value_RI_g1;
% table_pvalues_characteristics{5,5}=p_value_RI_g2;

%% Driver Location and metrics
if isempty(strfind(model,'Sinusal'))
    
    % Interpolation
    %     try
    %         [driver_data_interp_classical]=driver_data(x_interp, DF_interp_classical, fs, xmodes_classical, atrial_model.areas, xSMF_classical, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Interp - Classical)');
    %         [driver_data_interp_classical]=[];
    %     end
    %     try
    %         [driver_data_interp_BS]=driver_data(x_interp, DF_interp_BS, fs, xmodes_BS, atrial_model.areas, xSMF_BS, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Interp - BS)');
    %         [driver_data_interp_BS]=[];
    %     end
    %     try
    %         [driver_data_interp_BS_Corrected]=driver_data(x_interp, DF_interp_BS_Corrected, fs, xmodes_BS_Corrected, atrial_model.areas, xSMF_BS_Corrected, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Interp - BS_Corrected)');
    %         [driver_data_interp_BS_Corrected]=[];
    %     end
    
    try
        [driver_data_interp_classical_noHDF]=driver_data(x_interp, DF_interp_classical, fs, xmodes_classical_noHDF, atrial_model.areas, xSMF_classical_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Interp - Classical, noHDF)');
        [driver_data_interp_classical_noHDF]=[];
    end
    try
        [driver_data_interp_BS_noHDF]=driver_data(x_interp, DF_interp_BS, fs, xmodes_BS_noHDF, atrial_model.areas, xSMF_BS_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Interp - BS, noHDF)');
        [driver_data_interp_BS_noHDF]=[];
    end
    try
        [driver_data_interp_BS_Corrected_noHDF]=driver_data(x_interp, DF_interp_BS_Corrected, fs, xmodes_BS_Corrected_noHDF, atrial_model.areas, xSMF_BS_Corrected_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Interp - BS_Corrected, noHDF)');
        [driver_data_interp_BS_Corrected_noHDF]=[];
    end
    
    % Tikh-g0
    %     try
    %         [driver_data_tikh_Classical]=driver_data(x_hat_tikh, DF_tikh_classical, fs, xmodes_classical, atrial_model.areas, xSMF_classical, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Tikh - Classical)');
    %         [driver_data_tikh_Classical]=[];
    %     end
    %     try
    %         [driver_data_tikh_BS]=driver_data(x_hat_tikh, DF_tikh_BS, fs, xmodes_BS, atrial_model.areas, xSMF_BS, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Tikh - BS)');
    %         [driver_data_tikh_BS]=[];
    %     end
    %     try
    %         [driver_data_tikh_BS_Corrected]=driver_data(x_hat_tikh, DF_tikh_BS_Corrected, fs, xmodes_BS_Corrected, atrial_model.areas, xSMF_BS_Corrected, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Tikh - BS_Corrected)');
    %         [driver_data_tikh_BS_Corrected]=[];
    %     end
    
    try
        [driver_data_tikh_Classical_noHDF]=driver_data(x_hat_tikh, DF_tikh_classical, fs, xmodes_classical_noHDF, atrial_model.areas, xSMF_classical_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Tikh - Classical, noHDF)');
        [driver_data_tikh_Classical_noHDF]=[];
    end
    try
        [driver_data_tikh_BS_noHDF]=driver_data(x_hat_tikh, DF_tikh_BS, fs, xmodes_BS_noHDF, atrial_model.areas, xSMF_BS_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Tikh - BS, noHDF)');
        [driver_data_tikh_BS_noHDF]=[];
    end
    try
        [driver_data_tikh_BS_Corrected_noHDF]=driver_data(x_hat_tikh, DF_tikh_BS_Corrected, fs, xmodes_BS_Corrected_noHDF, atrial_model.areas, xSMF_BS_Corrected_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Tikh - BS_Corrected, noHDF)');
        [driver_data_tikh_BS_Corrected_noHDF]=[];
    end
    
    % Cons-g1
    %     try
    %         [driver_data_consg1_Classical]=driver_data(x_hat_consg1, DF_consg1_classical, fs, xmodes_classical, atrial_model.areas, xSMF_classical, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Cons-g1 - Classical)');
    %         [driver_data_consg1_Classical]=[];
    %     end
    %     try
    %         [driver_data_consg1_BS]=driver_data(x_hat_consg1, DF_consg1_BS, fs, xmodes_BS, atrial_model.areas, xSMF_BS, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Cons-g1 - BS)');
    %         [driver_data_consg1_BS]=[];
    %     end
    %     try
    %         [driver_data_consg1_BS_Corrected]=driver_data(x_hat_consg1, DF_consg1_BS_Corrected, fs, xmodes_BS_Corrected, atrial_model.areas, xSMF_BS_Corrected, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Cons-g1 - BS Corrected)');
    %         [driver_data_consg1_BS_Corrected]=[];
    %     end
    
    try
        [driver_data_consg1_Classical_noHDF]=driver_data(x_hat_consg1, DF_consg1_classical, fs, xmodes_classical_noHDF, atrial_model.areas, xSMF_classical_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Cons-g1 - Classical, noHDF)');
        [driver_data_consg1_Classical_noHDF]=[];
    end
    try
        [driver_data_consg1_BS_noHDF]=driver_data(x_hat_consg1, DF_consg1_BS, fs, xmodes_BS_noHDF, atrial_model.areas, xSMF_BS_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Cons-g1 - BS, noHDF)');
        [driver_data_consg1_BS_noHDF]=[];
    end
    try
        [driver_data_consg1_BS_Corrected_noHDF]=driver_data(x_hat_consg1, DF_consg1_BS_Corrected, fs, xmodes_BS_Corrected_noHDF, atrial_model.areas, xSMF_BS_Corrected_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Cons-g1 - BS Corrected, noHDF)');
        [driver_data_consg1_BS_Corrected_noHDF]=[];
    end
    
    % Cons-g2
    %     try
    %         [driver_data_consg2_Classical]=driver_data(x_hat_consg2, DF_consg2_classical, fs, xmodes_classical, atrial_model.areas, xSMF_classical, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Cons-g2 - Classical)');
    %         [driver_data_consg2_Classical]=[];
    %     end
    %     try
    %         [driver_data_consg2_BS]=driver_data(x_hat_consg2, DF_consg2_BS, fs, xmodes_BS, atrial_model.areas, xSMF_BS, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Cons-g2 - BS)');
    %         [driver_data_consg2_BS]=[];
    %     end
    %     try
    %         [driver_data_consg2_BS_Corrected]=driver_data(x_hat_consg2, DF_consg2_BS_Corrected, fs, xmodes_BS_Corrected, atrial_model.areas, xSMF_BS_Corrected, load([data_path 'filled_geometry.mat']), atrial_model,1);
    %     catch ME
    %         disp('Error detectando rotores (Cons-g2 - BS_Corrected)');
    %         [driver_data_consg2_BS_Corrected]=[];
    %     end
    
    try
        [driver_data_consg2_Classical_noHDF]=driver_data(x_hat_consg2, DF_consg2_classical, fs, xmodes_classical_noHDF, atrial_model.areas, xSMF_classical_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Cons-g2 - Classical)');
        [driver_data_consg2_Classical_noHDF]=[];
    end
    try
        [driver_data_consg2_BS_noHDF]=driver_data(x_hat_consg2, DF_consg2_BS, fs, xmodes_BS_noHDF, atrial_model.areas, xSMF_BS_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Cons-g2 - BS)');
        [driver_data_consg2_BS_noHDF]=[];
    end
    try
        [driver_data_consg2_BS_Corrected_noHDF]=driver_data(x_hat_consg2, DF_consg2_BS_Corrected, fs, xmodes_BS_Corrected_noHDF, atrial_model.areas, xSMF_BS_Corrected_noHDF, load([data_path 'filled_geometry.mat']), atrial_model,0);
    catch ME
        disp('Error detectando rotores (Cons-g2 - BS_Corrected)');
        [driver_data_consg2_BS_Corrected_noHDF]=[];
    end
end

%% Store results
if ~isempty(x_hat_tikh)
    Results_Tikhonov_g0.xhat=x_hat_tikh;
    Results_Tikhonov_g0.xhat_BS=xhat_tikh_BS;
    Results_Tikhonov_g0.characteristics.DF.Classical = DF_tikh_classical;
    Results_Tikhonov_g0.characteristics.DF.BS = DF_tikh_BS;
    Results_Tikhonov_g0.characteristics.DF.BS_Corrected = DF_tikh_BS_Corrected;
    Results_Tikhonov_g0.characteristics.metrics_DF.Classical=metrics_DF_tikh_classical;
    Results_Tikhonov_g0.characteristics.metrics_DF.BS=metrics_DF_tikh_BS;
    Results_Tikhonov_g0.characteristics.metrics_DF.BS_Corrected=metrics_DF_tikh_BS_Corrected;
    %     Results_Tikhonov_g0.characteristics.sampen = sampen_tikh;
    %     Results_Tikhonov_g0.characteristics.metrics_sampen = samp_metrics_tikh;
    %     Results_Tikhonov_g0.characteristics.RI = RI_tikh;
    %     Results_Tikhonov_g0.characteristics.OI = OI_tikh;
    %     Results_Tikhonov_g0.characteristics.metrics_RI_OI = metrics_RI_OI_tikh;
    Results_Tikhonov_g0.phase.Classical=phase_tikh_classical;
    Results_Tikhonov_g0.phase_metrics.Classical=phase_metrics_tikh_classical;
    Results_Tikhonov_g0.phase.BS=phase_tikh_BS;
    Results_Tikhonov_g0.phase_metrics.BS=phase_metrics_tikh_BS;
    Results_Tikhonov_g0.phase.BS_Corrected=phase_tikh_BS_Corrected;
    Results_Tikhonov_g0.phase_metrics.BS_Corrected=phase_metrics_tikh_BS_Corrected;
    Results_Tikhonov_g0.phase.noHDF=phase_tikh_noHDF;
    Results_Tikhonov_g0.phase_metrics.noHDF=phase_metrics_tikh_noHDF;
    if isempty(strfind(model,'Sinusal'))
        %         Results_Tikhonov_g0.driver_data.Classical=driver_data_tikh_Classical;
        %         Results_Tikhonov_g0.driver_data.BS=driver_data_tikh_BS;
        %         Results_Tikhonov_g0.driver_data.BS_Corrected=driver_data_tikh_BS_Corrected;
        Results_Tikhonov_g0.driver_data_noHDF.Classical=driver_data_tikh_Classical_noHDF;
        Results_Tikhonov_g0.driver_data_noHDF.BS=driver_data_tikh_BS_noHDF;
        Results_Tikhonov_g0.driver_data_noHDF.BS_Corrected=driver_data_tikh_BS_Corrected_noHDF;
    end
    Results_Tikhonov_g0.lambda_opt=lambda_opt_tikh;
    Results_Tikhonov_g0.estimation_metrics=timemetrics_tikhg0;
    Results_Tikhonov_g0.estimation_metrics.DTW=DTW_tikhg0;
    Results_Tikhonov_g0.estimation_metrics.mDTW=metrics_DTW_tikhg0.mDTW;
    Results_Tikhonov_g0.estimation_metrics.stdDTW=metrics_DTW_tikhg0.stdDTW;
end

if ~isempty(x_hat_consg1)
    Results_Constrained_g1.xhat=x_hat_consg1;
    Results_Constrained_g1.xhat_BS=xhat_consg1_BS;
    Results_Constrained_g1.characteristics.DF.Classical = DF_consg1_classical;
    Results_Constrained_g1.characteristics.metrics_DF.Classical=metrics_DF_consg1_classical;
    Results_Constrained_g1.characteristics.DF.BS = DF_consg1_BS;
    Results_Constrained_g1.characteristics.metrics_DF.BS=metrics_DF_consg1_BS;
    Results_Constrained_g1.characteristics.DF.BS_Corrected = DF_consg1_BS_Corrected;
    Results_Constrained_g1.characteristics.metrics_DF.BS_Corrected=metrics_DF_consg1_BS_Corrected;
    %     Results_Constrained_g1.characteristics.sampen = sampen_consg1;
    %     Results_Constrained_g1.characteristics.metrics_sampen = samp_metrics_consg1;
    %     Results_Constrained_g1.characteristics.RI = RI_consg1;
    %     Results_Constrained_g1.characteristics.OI = OI_consg1;
    %     Results_Constrained_g1.characteristics.metrics_RI_OI = metrics_RI_OI_consg1;
    Results_Constrained_g1.phase.Classical=phase_consg1_classical;
    Results_Constrained_g1.phase_metrics.Classical=phase_metrics_consg1_classical;
    Results_Constrained_g1.phase.BS=phase_consg1_BS;
    Results_Constrained_g1.phase_metrics.BS=phase_metrics_consg1_BS;
    Results_Constrained_g1.phase.BS_Corrected=phase_consg1_BS_Corrected;
    Results_Constrained_g1.phase_metrics.BS_Corrected=phase_metrics_consg1_BS_Corrected;
    Results_Constrained_g1.phase.noHDF=phase_consg1_noHDF;
    Results_Constrained_g1.phase_metrics.noHDF=phase_metrics_consg1_noHDF;
    if isempty(strfind(model,'Sinusal'))
        %         Results_Constrained_g1.driver_data.Classical=driver_data_consg1_Classical;
        %         Results_Constrained_g1.driver_data.BS=driver_data_consg1_BS;
        %         Results_Constrained_g1.driver_data.BS_Corrected=driver_data_consg1_BS_Corrected;
        Results_Constrained_g1.driver_data_noHDF.Classical=driver_data_consg1_Classical_noHDF;
        Results_Constrained_g1.driver_data_noHDF.BS=driver_data_consg1_BS_noHDF;
        Results_Constrained_g1.driver_data_noHDF.BS_Corrected=driver_data_consg1_BS_Corrected_noHDF;
    end
    Results_Constrained_g1.lambda_opt=lambda_opt_consg1;
    Results_Constrained_g1.estimation_metrics=timemetrics_consg1;
    Results_Constrained_g1.estimation_metrics.DTW=DTW_consg1;
    Results_Constrained_g1.estimation_metrics.mDTW=metrics_DTW_consg1.mDTW;
    Results_Constrained_g1.estimation_metrics.stdDTW=metrics_DTW_consg1.stdDTW;
end

if ~isempty(x_hat_consg2)
    Results_Constrained_g2.xhat=x_hat_consg2;
    Results_Constrained_g2.xhat_BS=xhat_consg2_BS;
    Results_Constrained_g2.characteristics.DF.Classical = DF_consg2_classical;
    Results_Constrained_g2.characteristics.metrics_DF.Classical=metrics_DF_consg2_classical;
    Results_Constrained_g2.characteristics.DF.BS = DF_consg2_BS;
    Results_Constrained_g2.characteristics.metrics_DF.BS=metrics_DF_consg2_BS;
    Results_Constrained_g2.characteristics.DF.BS_Corrected = DF_consg2_BS_Corrected;
    Results_Constrained_g2.characteristics.metrics_DF.BS_Corrected=metrics_DF_consg2_BS_Corrected;
    %     Results_Constrained_g2.characteristics.sampen = sampen_consg2;
    %     Results_Constrained_g2.characteristics.metrics_sampen = samp_metrics_consg2;
    %     Results_Constrained_g2.characteristics.RI = RI_consg2;
    %     Results_Constrained_g2.characteristics.OI = OI_consg2;
    %     Results_Constrained_g2.characteristics.metrics_RI_OI = metrics_RI_OI_consg2;
    Results_Constrained_g2.phase.Classical=phase_consg2_classical;
    Results_Constrained_g2.phase_metrics.Classical=phase_metrics_consg2_classical;
    Results_Constrained_g2.phase.BS=phase_consg2_BS;
    Results_Constrained_g2.phase_metrics.BS=phase_metrics_consg2_BS;
    Results_Constrained_g2.phase.BS_Corrected=phase_consg2_BS_Corrected;
    Results_Constrained_g2.phase_metrics.BS_Corrected=phase_metrics_consg2_BS_Corrected;
    Results_Constrained_g2.phase.noHDF=phase_consg2_noHDF;
    Results_Constrained_g2.phase_metrics.noHDF=phase_metrics_consg2_noHDF;
    if isempty(strfind(model,'Sinusal'))
        %         Results_Constrained_g2.driver_data.Classical=driver_data_consg2_Classical;
        %         Results_Constrained_g2.driver_data.BS=driver_data_consg2_BS;
        %         Results_Constrained_g2.driver_data.BS_Corrected=driver_data_consg2_BS_Corrected;
        Results_Constrained_g2.driver_data_noHDF.Classical=driver_data_consg2_Classical_noHDF;
        Results_Constrained_g2.driver_data_noHDF.BS=driver_data_consg2_BS_noHDF;
        Results_Constrained_g2.driver_data_noHDF.BS_Corrected=driver_data_consg2_BS_Corrected_noHDF;
    end
    Results_Constrained_g2.lambda_opt=lambda_opt_consg2;
    Results_Constrained_g2.estimation_metrics=timemetrics_consg2;
    Results_Constrained_g2.estimation_metrics.DTW=DTW_consg2;
    Results_Constrained_g2.estimation_metrics.mDTW=metrics_DTW_consg2.mDTW;
    Results_Constrained_g2.estimation_metrics.stdDTW=metrics_DTW_consg2.stdDTW;
end

if ~isempty(x_interp)
    Results_Interpolation.xhat=x_interp;
    Results_Interpolation.xhat_BS=xhat_interp_BS;
    Results_Interpolation.characteristics.DF.Classical = DF_interp_classical;
    Results_Interpolation.characteristics.metrics_DF.Classical=metrics_DF_interp_classical;
    Results_Interpolation.characteristics.DF.BS = DF_interp_BS;
    Results_Interpolation.characteristics.metrics_DF.BS=metrics_DF_interp_BS;
    Results_Interpolation.characteristics.DF.BS_Corrected = DF_interp_BS_Corrected;
    Results_Interpolation.characteristics.metrics_DF.BS_Corrected=metrics_DF_interp_BS_Corrected;
    %     Results_Interpolation.characteristics.sampen = sampen_interp;
    %     Results_Interpolation.characteristics.metrics_sampen = samp_metrics_interp;
    %     Results_Interpolation.characteristics.RI = RI_interp;
    %     Results_Interpolation.characteristics.OI = OI_interp;
    %     Results_Interpolation.characteristics.metrics_RI_OI = metrics_RI_OI_interp;
    Results_Interpolation.phase.Classical=phase_interp_classical;
    Results_Interpolation.phase_metrics.Classical=phase_metrics_interp_classical;
    Results_Interpolation.phase.BS=phase_interp_BS;
    Results_Interpolation.phase_metrics.BS=phase_metrics_interp_BS;
    Results_Interpolation.phase.BS_Corrected=phase_interp_BS_Corrected;
    Results_Interpolation.phase_metrics.BS_Corrected=phase_metrics_interp_BS_Corrected;
    Results_Interpolation.phase.noHDF=phase_interp_noHDF;
    Results_Interpolation.phase_metrics.noHDF=phase_metrics_interp_noHDF;
    if isempty(strfind(model,'Sinusal'))
        %         Results_Interpolation.driver_data.Classical=driver_data_interp_classical;
        %         Results_Interpolation.driver_data.BS=driver_data_interp_BS;
        %         Results_Interpolation.driver_data.BS_Corrected=driver_data_interp_BS_Corrected;
        Results_Interpolation.driver_data_noHDF.Classical=driver_data_interp_classical_noHDF;
        Results_Interpolation.driver_data_noHDF.BS=driver_data_interp_BS_noHDF;
        Results_Interpolation.driver_data_noHDF.BS_Corrected=driver_data_interp_BS_Corrected_noHDF;
    end
    Results_Interpolation.estimation_metrics=timemetrics_interp;
    Results_Interpolation.estimation_metrics.DTW=DTW_interp;
    Results_Interpolation.estimation_metrics.mDTW=metrics_DTW_interp.mDTW;
    Results_Interpolation.estimation_metrics.stdDTW=metrics_DTW_interp.stdDTW;
end

Ground_Truth.x=x;
Ground_Truth.x_BS=x_BS;
Ground_Truth.y=y;
Ground_Truth.characteristics.xDF.Classical=xDF_Classical;
Ground_Truth.characteristics.metrics_xDF.Classical.mxDF=mxDF_Classical;
Ground_Truth.characteristics.metrics_xDF.Classical.stdxDF=stdxDF_Classical;
Ground_Truth.characteristics.xDF.BS=xDF_BS;
Ground_Truth.characteristics.metrics_xDF.BS.mxDF=mxDF_BS;
Ground_Truth.characteristics.metrics_xDF.BS.stdxDF=stdxDF_BS;
Ground_Truth.characteristics.xDF.BS_Corrected=xDF_BS;
Ground_Truth.characteristics.metrics_xDF.BS_Corrected.mxDF=mxDF_BS_Corrected;
Ground_Truth.characteristics.metrics_xDF.BS_Corrected.stdxDF=stdxDF_BS_Corrected;
% Ground_Truth.characteristics.sampen=xsampen;
% Ground_Truth.characteristics.metrics_xsampen.msampen=mxsampen;
% Ground_Truth.characteristics.metrics_xsampen.stdsampen=stdxsampen;
% Ground_Truth.characteristics.xRI=xRI;
% Ground_Truth.characteristics.xOI=xOI;
% Ground_Truth.characteristics.metrics_xRI_xOI=metrics_xRI_xOI;
Ground_Truth.xphase.Classical=xphase_classical;
Ground_Truth.xphase.BS=xphase_BS;
Ground_Truth.xphase.BS_Corrected=xphase_BS_Corrected;
Ground_Truth.xphase.noHDF=xphase_noHDF;
if isempty(strfind(model,'Sinusal'))
    %     Ground_Truth.xdriver.Classical=xdriver_classical;
    %     Ground_Truth.xmodes.Classical=xmodes_classical;
    %     Ground_Truth.xSMF.Classical=xSMF_classical;
    %     Ground_Truth.xdriver.BS=xdriver_BS;
    %     Ground_Truth.xmodes.BS=xmodes_BS;
    %     Ground_Truth.xSMF.BS=xSMF_BS;
    %     Ground_Truth.xdriver.BS_Corrected=xdriver_BS_Corrected;
    %     Ground_Truth.xmodes.BS_Corrected=xmodes_BS_Corrected;
    %     Ground_Truth.xSMF.BS_Corrected=xSMF_BS_Corrected;
    
    Ground_Truth.xdriver_noHDF.Classical=xdriver_classical_noHDF;
    Ground_Truth.xmodes_noHDF.Classical=xmodes_classical_noHDF;
    Ground_Truth.xSMF_noHDF.Classical=xSMF_classical_noHDF;
    Ground_Truth.xdriver_noHDF.BS=xdriver_BS_noHDF;
    Ground_Truth.xmodes_noHDF.BS=xmodes_BS_noHDF;
    Ground_Truth.xSMF_noHDF.BS=xSMF_BS_noHDF;
    Ground_Truth.xdriver_noHDF.BS_Corrected=xdriver_BS_Corrected_noHDF;
    Ground_Truth.xmodes_noHDF.BS_Corrected=xmodes_BS_Corrected_noHDF;
    Ground_Truth.xSMF_noHDF.BS_Corrected=xSMF_BS_Corrected_noHDF;
end

nodes_constrained=Constrained_nodes;
SNR_BSP=SNR;
if ~exist('SNR_cons','var')
    SNR_constrained='Max';
end

%% Save results
results_file = [results_path save_path];
mkdir([results_path model]);

% Store tables
xlswrite(sprintf('%s_metrics.xlsx',results_file),table_metrics);
xlswrite(sprintf('%s_characteristics.xlsx',results_file),table_characteristics);
xlswrite(sprintf('%s_pvalues_metrics.xlsx',results_file),table_pvalues_metrics);
xlswrite(sprintf('%s_pvalues_characteristics.xlsx',results_file),table_pvalues_characteristics);

% Store simulations
save(results_file,'-v7.3','fs','Ground_Truth','Results_Interpolation','Results_Tikhonov_g0',...
    'Results_Constrained_g1','Results_Constrained_g2','model','table_metrics',...
    'table_characteristics','table_pvalues_metrics','table_pvalues_characteristics',...
    'nodes_constrained','SNR_BSP','SNR_constrained');