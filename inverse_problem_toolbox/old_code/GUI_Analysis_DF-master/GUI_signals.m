function varargout = GUI_signals(varargin)
% GUI_SIGNALS MATLAB code for GUI_signals.fig
%      GUI_SIGNALS, by itself, creates a new GUI_SIGNALS or raises the existing
%      singleton*.
%
%      H = GUI_SIGNALS returns the handle to a new GUI_SIGNALS or the handle to
%      the existing singleton*.
%
%      GUI_SIGNALS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI_SIGNALS.M with the given input arguments.
%
%      GUI_SIGNALS('Property','Value',...) creates a new GUI_SIGNALS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_signals_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_signals_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI_signals

% Last Modified by GUIDE v2.5 22-Feb-2018 14:53:57

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @GUI_signals_OpeningFcn, ...
    'gui_OutputFcn',  @GUI_signals_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI_signals is made visible.
function GUI_signals_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI_signals (see VARARGIN)

% Choose default command line output for GUI_signals
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
addpath(genpath('Toolbox'));
addpath('Geometry');
load('filled_geometry.mat');
handles.atrial_model=atrial_model;
handles.analysisfigs={};
guidata(hObject, handles);
% UIWAIT makes GUI_signals wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_signals_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function open_simulation_Callback(hObject, eventdata, handles)
% hObject    handle to open_simulation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%% Cargo los datos
[handles.simulation_filename,handles.simulation_filepath_g0] = uigetfile({'*.mat',  'MATLAB files (*.mat)'}, ...
    'Select simulation');
if ~isequal(handles.simulation_filename, 0)
    if isunix
        path_str=strcat(handles.simulation_filepath_g0,'/',handles.simulation_filename);
        load(sprintf(path_str));
    else
        path_str=strcat(handles.simulation_filepath_g0,handles.simulation_filename);
        load(path_str);
    end
    
    if(isfield(handles,'simulation'))
        axes(handles.axes1);
        cla;
        handles = rmfield(handles,'simulation');
        axes(handles.axes4);
        cla;
    end
    
    if(isfield(handles,'estimation_metrics'))
        handles = rmfield(handles,'estimation_metrics');
        handles = rmfield(handles,'DF_metrics');
    end
    
    if(isfield(handles,'spectrums'))
        axes(handles.axes2);
        cla;
        handles = rmfield(handles,'spectrums');
        axes(handles.axes5);
        cla;
    end
    
    if isfield(handles,'nodes_constrained')
        handles = rmfield(handles,'nodes_constrained');
        set(handles.node_info,'String','Cons-tikh node information');
        set(handles.node_info,'ForegroundColor',[0,0,0]);
    else
        set(handles.node_info,'String','Cons-tikh node information');
        set(handles.node_info,'ForegroundColor',[0,0,0]);
    end
    
    
    if contains(model,'Sinusal')
        handles.instant_time = 826;
        handles.minF = 1;   % low frequency.
        handles.maxF = 1.4; % % high frequency.
    else
        handles.instant_time = 711;
        handles.minF = 2.5;   % low frequency.
        handles.maxF = 5.5; % % high frequency.
    end
        
    handles.fs=500;
    handles.Ground_Truth=Ground_Truth;
    handles.model=model;
    handles.nodes_constrained=nodes_constrained;
    handles.Results_Constrained_g1=Results_Constrained_g1;
    handles.Results_Constrained_g2=Results_Constrained_g2;
    handles.Results_Interpolation=Results_Interpolation;
    handles.Results_Tikhonov_g0=Results_Tikhonov_g0;
    handles.SNR_BSP=SNR_BSP;
    handles.SNR_constrained=SNR_constrained;
    
else
    return;
end



%% Calculo los espectros
[handles.spectrums.Pxx_real,handles.spectrums.f_real] = pwelch(handles.Ground_Truth.x'-mean(handles.Ground_Truth.x'),[],[],2048,fs);
[handles.spectrums.Pxx_interp,handles.spectrums.f_interp] = pwelch(handles.Results_Interpolation.xhat'-mean(handles.Results_Interpolation.xhat'),[],[],2048,fs);
[handles.spectrums.Pxx_tikhg0,handles.spectrums.f_tikhg0] = pwelch(handles.Results_Tikhonov_g0.xhat'-mean(handles.Results_Tikhonov_g0.xhat'),[],[],2048,fs);
[handles.spectrums.Pxx_consg1,handles.spectrums.f_consg1] = pwelch(handles.Results_Constrained_g1.xhat'-mean(handles.Results_Constrained_g1.xhat'),[],[],2048,fs);
[handles.spectrums.Pxx_consg2,handles.spectrums.f_consg2] = pwelch(handles.Results_Constrained_g2.xhat'-mean(handles.Results_Constrained_g2.xhat'),[],[],2048,fs);

[handles.spectrums.Pxx_real_BS,handles.spectrums.f_real_BS] = pwelch(handles.Ground_Truth.x_BS'-mean(handles.Ground_Truth.x_BS'),[],[],2048,fs);
[handles.spectrums.Pxx_interp_BS,handles.spectrums.f_interp_BS] = pwelch(handles.Results_Interpolation.xhat_BS'-mean(handles.Results_Interpolation.xhat_BS'),[],[],2048,fs);
[handles.spectrums.Pxx_tikhg0_BS,handles.spectrums.f_tikhg0_BS] = pwelch(handles.Results_Tikhonov_g0.xhat_BS'-mean(handles.Results_Tikhonov_g0.xhat_BS'),[],[],2048,fs);
[handles.spectrums.Pxx_consg1_BS,handles.spectrums.f_consg1_BS] = pwelch(handles.Results_Constrained_g1.xhat_BS'-mean(handles.Results_Constrained_g1.xhat_BS'),[],[],2048,fs);
[handles.spectrums.Pxx_consg2_BS,handles.spectrums.f_consg2_BS] = pwelch(handles.Results_Constrained_g2.xhat_BS'-mean(handles.Results_Constrained_g2.xhat_BS'),[],[],2048,fs);

%% Muestro las señales
handles.colormapsigs=lines(4);
selected_node=1;

axes(handles.axes1);
fs=500;
t=0:1/fs:length(handles.Ground_Truth.x)/fs-1/fs;
hold on
plot(t,handles.Ground_Truth.x(selected_node,:)./max(abs(handles.Ground_Truth.x(selected_node,:))),'Color',handles.colormapsigs(1,:),'LineWidth',3),
plot(t,handles.Ground_Truth.x_BS(selected_node,:)./max(abs(handles.Ground_Truth.x_BS(selected_node,:))),'LineStyle','--','Color',handles.colormapsigs(2,:),'LineWidth',3),
plot(t,handles.Results_Interpolation.xhat(selected_node,:)./max(abs(handles.Results_Interpolation.xhat(selected_node,:))),'Color',handles.colormapsigs(3,:),'LineWidth',1.5),
plot(t,handles.Results_Interpolation.xhat_BS(selected_node,:)./max(abs(handles.Results_Interpolation.xhat_BS(selected_node,:))),'LineStyle','--','Color',handles.colormapsigs(4,:),'LineWidth',1.5),
xlabel('Time (s)'),ylabel('Amplitude (normalized)'),xlim([t(1) t(end)/2]), grid on,
handles.legend1=legend('GT','GT (BS)','Interp','Interp (BS)');
hold off

axes(handles.axes4);
fs=500;
t=0:1/fs:length(handles.Ground_Truth.x)/fs-1/fs;
hold on
plot(t,handles.Results_Tikhonov_g0.xhat(selected_node,:)./max(abs(handles.Results_Tikhonov_g0.xhat(selected_node,:))),'Color',handles.colormapsigs(1,:),'LineWidth',3),
plot(t,handles.Results_Tikhonov_g0.xhat_BS(selected_node,:)./max(abs(handles.Results_Tikhonov_g0.xhat_BS(selected_node,:))),'LineStyle','--','Color',handles.colormapsigs(2,:),'LineWidth',3),
plot(t,handles.Results_Constrained_g1.xhat(selected_node,:)./max(abs(handles.Results_Constrained_g1.xhat(selected_node,:))),'Color',handles.colormapsigs(3,:),'LineWidth',1.5),
plot(t,handles.Results_Constrained_g1.xhat_BS(selected_node,:)./max(abs(handles.Results_Constrained_g1.xhat_BS(selected_node,:))),'LineStyle','--','Color',handles.colormapsigs(4,:),'LineWidth',1.5),
xlabel('Time (s)'),ylabel('Amplitude (normalized)'),xlim([t(1) t(end)/2]), grid on,
handles.legend2=legend('Tikh-g0','Tikh-g0 (BS)','Cons-g1','Cons-g1 (BS)');
hold off

%% Muestro los espectros con sus respectivas DFs
axes(handles.axes2);
hold on
plot(handles.spectrums.f_real(:,selected_node),handles.spectrums.Pxx_real(:,selected_node)./max(handles.spectrums.Pxx_real(:,selected_node)),'Color',handles.colormapsigs(1,:),'LineWidth',3),
plot(handles.spectrums.f_real_BS(:,selected_node),handles.spectrums.Pxx_real_BS(:,selected_node)./max(handles.spectrums.Pxx_real_BS(:,selected_node)),'LineStyle','--','Color',handles.colormapsigs(2,:),'LineWidth',3),
plot(handles.spectrums.f_interp(:,selected_node),handles.spectrums.Pxx_interp(:,selected_node)./max(handles.spectrums.Pxx_interp(:,selected_node)),'Color',handles.colormapsigs(3,:),'LineWidth',1.5),
plot(handles.spectrums.f_interp_BS(:,selected_node),handles.spectrums.Pxx_interp_BS(:,selected_node)./max(handles.spectrums.Pxx_interp_BS(:,selected_node)),'LineStyle','--','Color',handles.colormapsigs(4,:),'LineWidth',1.5),
plot([handles.Ground_Truth.characteristics.xDF.BS_Corrected(selected_node) handles.Ground_Truth.characteristics.xDF.BS_Corrected(selected_node)],[0 1],'Color',handles.colormapsigs(1,:),'LineWidth',2);
plot([handles.Ground_Truth.characteristics.xDF.BS(selected_node) handles.Ground_Truth.characteristics.xDF.BS(selected_node)],[0 1],'Color',handles.colormapsigs(2,:),'LineWidth',2);
plot([handles.Results_Interpolation.characteristics.DF.BS_Corrected(selected_node) handles.Results_Interpolation.characteristics.DF.BS_Corrected(selected_node)],[0 1],'Color',handles.colormapsigs(3,:),'LineWidth',2);
plot([handles.Results_Interpolation.characteristics.DF.BS(selected_node) handles.Results_Interpolation.characteristics.DF.BS(selected_node)],[0 1],'Color',handles.colormapsigs(4,:),'LineWidth',2);
xlabel('Frequency (Hz)'),ylabel('Pxx (normalized)'),
xlim([0 16]), ylim auto,
handles.legend3=legend('GT','GT (BS)','Interp','Interp (BS)');
hold off

axes(handles.axes5);
hold on
plot(handles.spectrums.f_tikhg0(:,selected_node),handles.spectrums.Pxx_tikhg0(:,selected_node)./max(handles.spectrums.Pxx_tikhg0(:,selected_node)),'Color',handles.colormapsigs(1,:),'LineWidth',3),
plot(handles.spectrums.f_tikhg0_BS(:,selected_node),handles.spectrums.Pxx_tikhg0_BS(:,selected_node)./max(handles.spectrums.Pxx_tikhg0_BS(:,selected_node)),'LineStyle','--','Color',handles.colormapsigs(2,:),'LineWidth',3),
plot(handles.spectrums.f_consg1(:,selected_node),handles.spectrums.Pxx_consg1(:,selected_node)./max(handles.spectrums.Pxx_consg1(:,selected_node)),'Color',handles.colormapsigs(3,:),'LineWidth',1.5),
plot(handles.spectrums.f_consg1_BS(:,selected_node),handles.spectrums.Pxx_consg1_BS(:,selected_node)./max(handles.spectrums.Pxx_consg1_BS(:,selected_node)),'LineStyle','--','Color',handles.colormapsigs(4,:),'LineWidth',1.5),
plot([handles.Results_Tikhonov_g0.characteristics.DF.BS_Corrected(selected_node) handles.Results_Tikhonov_g0.characteristics.DF.BS_Corrected(selected_node)],[0 1],'Color',handles.colormapsigs(1,:),'LineWidth',2);
plot([handles.Results_Tikhonov_g0.characteristics.DF.BS(selected_node) handles.Results_Tikhonov_g0.characteristics.DF.BS(selected_node)],[0 1],'Color',handles.colormapsigs(2,:),'LineWidth',2);
plot([handles.Results_Constrained_g1.characteristics.DF.BS_Corrected(selected_node) handles.Results_Constrained_g1.characteristics.DF.BS_Corrected(selected_node)],[0 1],'Color',handles.colormapsigs(3,:),'LineWidth',2);
plot([handles.Results_Constrained_g1.characteristics.DF.BS(selected_node) handles.Results_Constrained_g1.characteristics.DF.BS(selected_node)],[0 1],'Color',handles.colormapsigs(4,:),'LineWidth',2);
xlabel('Frequency (Hz)'),ylabel('Pxx (normalized)'),
xlim([0 16]), ylim auto,
handles.legend4=legend('Tikh-g0','Tikh-g0 (BS)','Cons-g1','Cons-g1 (BS)');
hold off

%% Check if node is contained on constrained algorithm
set(handles.popupmenu1,'String',num2cell(1:2048));
set(handles.popupmenu1,'Value',selected_node);

if isfield(handles,'nodes_constrained')
    if ismember(selected_node,handles.nodes_constrained)
        set(handles.node_info,'String','Node INCLUDED in Cons-Tikh simulation');
        set(handles.node_info,'ForegroundColor',[0,1,0]);
    else
        set(handles.node_info,'String','Node NOT INCLUDED in Cons-Tikh simulation');
        set(handles.node_info,'ForegroundColor',[1,0,0]);
    end
else
    set(handles.node_info,'String','Node information not available');
    set(handles.node_info,'ForegroundColor',[0,0,0]);
end

guidata(hObject,handles);



% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1
selected_node=get(hObject,'Value');

axes(handles.axes1);
cla;
axes(handles.axes2);
cla;
axes(handles.axes4);
cla;
axes(handles.axes5);
cla;

%% Muestro las señales
axes(handles.axes1);
fs=500;
t=0:1/fs:length(handles.Ground_Truth.x)/fs-1/fs;
hold on
plot(t,handles.Ground_Truth.x(selected_node,:)./max(abs(handles.Ground_Truth.x(selected_node,:))),'Color',handles.colormapsigs(1,:),'LineWidth',3),
plot(t,handles.Ground_Truth.x_BS(selected_node,:)./max(abs(handles.Ground_Truth.x_BS(selected_node,:))),'LineStyle','--','Color',handles.colormapsigs(2,:),'LineWidth',3),
% plot(t,handles.Results_Interpolation.xhat(selected_node,:)./max(abs(handles.Results_Interpolation.xhat(selected_node,:))),'Color',handles.colormapsigs(3,:),'LineWidth',1.5),
% plot(t,handles.Results_Interpolation.xhat_BS(selected_node,:)./max(abs(handles.Results_Interpolation.xhat_BS(selected_node,:))),'LineStyle','--','Color',handles.colormapsigs(4,:),'LineWidth',1.5),
xlabel('Time (s)'),ylabel('Amplitude (normalized)'),xlim([t(1) t(end)/2]), grid on,
% handles.legend1=legend('GT','GT (BS)','Interp','Interp (BS)');
handles.legend1=legend('GT','GT (BS)');
hold off

axes(handles.axes4);
fs=500;
t=0:1/fs:length(handles.Ground_Truth.x)/fs-1/fs;
hold on
%plot(t,handles.Results_Tikhonov_g0.xhat(selected_node,:)./max(abs(handles.Results_Tikhonov_g0.xhat(selected_node,:))),'Color',handles.colormapsigs(1,:),'LineWidth',3),
%plot(t,handles.Results_Tikhonov_g0.xhat_BS(selected_node,:)./max(abs(handles.Results_Tikhonov_g0.xhat_BS(selected_node,:))),'LineStyle','--','Color',handles.colormapsigs(2,:),'LineWidth',3),
plot(t,handles.Results_Constrained_g1.xhat(selected_node,:)./max(abs(handles.Results_Constrained_g1.xhat(selected_node,:))),'Color',handles.colormapsigs(3,:),'LineWidth',1.5),
plot(t,handles.Results_Constrained_g1.xhat_BS(selected_node,:)./max(abs(handles.Results_Constrained_g1.xhat_BS(selected_node,:))),'LineStyle','--','Color',handles.colormapsigs(4,:),'LineWidth',1.5),
xlabel('Time (s)'),ylabel('Amplitude (normalized)'),xlim([t(1) t(end)/2]), grid on,
% handles.legend2=legend('Tikh-g0','Tikh-g0 (BS)','Cons-g1','Cons-g1 (BS)');
handles.legend2=legend('Cons-g1','Cons-g1 (BS)');
hold off

%% Muestro los espectros con sus respectivas DFs
axes(handles.axes2);
hold on
plot(handles.spectrums.f_real(:,1),handles.spectrums.Pxx_real(:,selected_node)./max(handles.spectrums.Pxx_real(:,selected_node)),'Color',handles.colormapsigs(1,:),'LineWidth',3),
plot(handles.spectrums.f_real_BS(:,1),handles.spectrums.Pxx_real_BS(:,selected_node)./max(handles.spectrums.Pxx_real_BS(:,selected_node)),'LineStyle','--','Color',handles.colormapsigs(2,:),'LineWidth',3),
% plot(handles.spectrums.f_interp(:,1),handles.spectrums.Pxx_interp(:,selected_node)./max(handles.spectrums.Pxx_interp(:,selected_node)),'Color',handles.colormapsigs(3,:),'LineWidth',1.5),
% plot(handles.spectrums.f_interp_BS(:,1),handles.spectrums.Pxx_interp_BS(:,selected_node)./max(handles.spectrums.Pxx_interp_BS(:,selected_node)),'LineStyle','--','Color',handles.colormapsigs(4,:),'LineWidth',1.5),
plot([handles.Ground_Truth.characteristics.xDF.BS_Corrected(selected_node) handles.Ground_Truth.characteristics.xDF.BS_Corrected(selected_node)],[0 1],'Color',handles.colormapsigs(1,:),'LineWidth',2);
plot([handles.Ground_Truth.characteristics.xDF.BS(selected_node) handles.Ground_Truth.characteristics.xDF.BS(selected_node)],[0 1],'Color',handles.colormapsigs(2,:),'LineWidth',2);
% plot([handles.Results_Interpolation.characteristics.DF.BS_Corrected(selected_node) handles.Results_Interpolation.characteristics.DF.BS_Corrected(selected_node)],[0 1],'Color',handles.colormapsigs(3,:),'LineWidth',2);
% plot([handles.Results_Interpolation.characteristics.DF.BS(selected_node) handles.Results_Interpolation.characteristics.DF.BS(selected_node)],[0 1],'Color',handles.colormapsigs(4,:),'LineWidth',2);
xlabel('Frequency (Hz)'),ylabel('Pxx (normalized)'),
xlim([0 15]),
set(gca,'Xtick',0:1:15)
set(gca,'XtickLabel',0:1:15), 
ylim auto,
% handles.legend3=legend('GT','GT (BS)','Interp','Interp (BS)');
handles.legend3=legend('GT(BSH)','GT (BS)');
hold off

axes(handles.axes5);
hold on
%plot(handles.spectrums.f_tikhg0(:,1),handles.spectrums.Pxx_tikhg0(:,selected_node)./max(handles.spectrums.Pxx_tikhg0(:,selected_node)),'Color',handles.colormapsigs(1,:),'LineWidth',3),
%plot(handles.spectrums.f_tikhg0_BS(:,1),handles.spectrums.Pxx_tikhg0_BS(:,selected_node)./max(handles.spectrums.Pxx_tikhg0_BS(:,selected_node)),'LineStyle','--','Color',handles.colormapsigs(2,:),'LineWidth',3),
plot(handles.spectrums.f_consg1(:,1),handles.spectrums.Pxx_consg1(:,selected_node)./max(handles.spectrums.Pxx_consg1(:,selected_node)),'Color',handles.colormapsigs(3,:),'LineWidth',1.5),
plot(handles.spectrums.f_consg1_BS(:,1),handles.spectrums.Pxx_consg1_BS(:,selected_node)./max(handles.spectrums.Pxx_consg1_BS(:,selected_node)),'LineStyle','--','Color',handles.colormapsigs(4,:),'LineWidth',1.5),
%  plot([handles.Results_Tikhonov_g0.characteristics.DF.BS_Corrected(selected_node) handles.Results_Tikhonov_g0.characteristics.DF.BS_Corrected(selected_node)],[0 1],'Color',handles.colormapsigs(1,:),'LineWidth',2);
%  plot([handles.Results_Tikhonov_g0.characteristics.DF.BS(selected_node) handles.Results_Tikhonov_g0.characteristics.DF.BS(selected_node)],[0 1],'Color',handles.colormapsigs(2,:),'LineWidth',2);
plot([handles.Results_Constrained_g1.characteristics.DF.BS_Corrected(selected_node) handles.Results_Constrained_g1.characteristics.DF.BS_Corrected(selected_node)],[0 1],'Color',handles.colormapsigs(3,:),'LineWidth',2);
plot([handles.Results_Constrained_g1.characteristics.DF.BS(selected_node) handles.Results_Constrained_g1.characteristics.DF.BS(selected_node)],[0 1],'Color',handles.colormapsigs(4,:),'LineWidth',2);
xlabel('Frequency (Hz)'),ylabel('Pxx (normalized)'),
xlim([0 15]),
set(gca,'Xtick',0:1:15)
set(gca,'XtickLabel',0:1:15), 
ylim auto,
% handles.legend4=legend('Tikh-g0','Tikh-g0 (BS)','Cons-g1','Cons-g1 (BS)');
handles.legend4=legend('Cons-g1(BSH)','Cons-g1 (BS)');
hold off


% Listo todas las figuras existentes.
figHandles = get(groot, 'Children');

% Miro si existe una figura de análisis abierta, y pinto el nodo
if isfield(handles,'graph_selection')
    if size(figHandles,1)>1 && ~handles.graph_selection
        last_figure=figHandles(2);
        figure(last_figure),
        aundreucallbackClickA3DPoint(handles.atrial_model.vertices',last_figure,selected_node);
    end
end
handles.graph_selection=0;

if isfield(handles,'nodes_constrained')
    if ismember(selected_node,handles.nodes_constrained)
        set(handles.node_info,'String','Node INCLUDED in Cons-Tikh simulation');
        set(handles.node_info,'ForegroundColor',[0,1,0]);
    else
        set(handles.node_info,'String','Node NOT INCLUDED in Cons-Tikh simulation');
        set(handles.node_info,'ForegroundColor',[1,0,0]);
    end
end
guidata(hObject,handles);

% Cojo la última figura abierta, y la uso para la selección del nodo.



% --- Executes on button press in fig_node_selection.
function fig_node_selection_Callback(hObject, eventdata, handles)
% hObject    handle to fig_node_selection (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Listo todas las figuras existentes.
figHandles = get(groot, 'Children');

% Si solo existe la figura de la ventana principal: error.
if size(figHandles,1)==1
    msgbox('Please, open an analysis figure to continue','Error');
    return;
end

% Cojo la última figura abierta, y la uso para la selección del nodo.
last_figure=figHandles(2);
figure(last_figure),
[nodos_out]=pincel(handles.atrial_model,0,last_figure);
handles.popupmenu1.Value=nodos_out;
handles.graph_selection=1;
guidata(hObject,handles);

popupmenu1_Callback(handles.popupmenu1,eventdata,handles);


% --- Executes on button press in show_cons_nodes.
function show_cons_nodes_Callback(hObject, eventdata, handles)
% hObject    handle to show_cons_nodes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Miro si tengo la información de los nodos disponible
if ~isfield(handles,'nodes_constrained')
    msgbox('Nodes information not available','Error');
    return;
end

% Listo todas las figuras existentes.
figHandles = get(groot, 'Children');

% Si solo existe la figura de la ventana principal: error.
if size(figHandles,1)==1
    msgbox('Please, open an analysis figure to continue','Error');
    return;
end

% Cojo la última figura abierta, y la uso para la selección del nodo.
last_figure=figHandles(2);

for i=1:length(handles.nodes_constrained)
    figure(last_figure),
    aundreucallbackClickA3DPoint_all(handles.atrial_model.vertices',last_figure,handles.nodes_constrained(i));
end
guidata(hObject,handles);



% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --------------------------------------------------------------------
function save_figure_Callback(hObject, eventdata, handles)
% hObject    handle to save_figure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file_name=inputdlg('Figure filename');
if ~isempty(file_name)
    file_name=file_name{1};
    set(gcf,'PaperPositionMode','auto');
    export_fig (strcat('Figures/',file_name,'.png'),'-native')
    fh = figure;
    copyobj([handles.legend1,handles.axes1], fh);
    copyobj([handles.legend2,handles.axes2], fh);
    copyobj([handles.legend3,handles.axes4], fh);
    copyobj([handles.legend4,handles.axes5], fh);
    saveas(fh, strcat('Figures/',file_name),'fig');
    close(fh);
end


% --------------------------------------------------------------------
function Analysis_menu_Callback(hObject, eventdata, handles)
% hObject    handle to Analysis_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function ep_potentials_Callback(hObject, eventdata, handles)
% hObject    handle to ep_potentials (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Mensaje preguntando qué orden quiero analizar.
[order_question,v] = listdlg('PromptString','Select a model to analyse:',...
    'SelectionMode','single',...
    'ListString',{'Real','Interp','Tikh g0','Cons g1','Cons g2'});

if isempty(order_question)|| ~v
    return;
end

switch order_question
    case 1
        loadnamebase='GT';
    case 2
        loadnamebase='Interp';
    case 3
        loadnamebase=strcat('Tikhonov_g0',handles.SNR_BSP,'_',handles.SNR_constrained');
    case 4
        loadnamebase=strcat('Constrained_g1',handles.SNR_BSP,'_',handles.SNR_constrained');
    case 5
        loadnamebase=strcat('Constrained_g2',handles.SNR_BSP,'_',handles.SNR_constrained');
end
loadpath=handles.simulation_filepath_g0;

% Mensaje preguntando si quiero sacar el video o una imagen estática.
video_question = questdlg('Would you like to view an static map or a video?', ...
    'Epicardial Potentials Maps', ...
    'Image','Video','Image');
switch video_question
    case 'Image'
        try
            if order_question~=1
                if isunix
                    path_str=strcat(loadpath,'/',loadnamebase,'flag1.fig');
                    handles.analysisfigs{size(handles.analysisfigs+1),1}=openfig(sprintf(path_str));
                else
                    path_str=strcat(loadpath,loadnamebase,'flag1.fig');
                    openfig(path_str);
                end
            else
                ME = MException('Model mismatch','Model selected: %s',order_question);
                throw(ME)
            end
        catch
            SNR_cons=strcat('Cons=',handles.SNR_constrained);
            SNR=strcat('SNR\_BSP=',num2str(handles.SNR_BSP),'dB');
            switch order_question
                case 1
                    resultset=handles.Ground_Truth.x(1:2039,:);
                    SNR = 'Ep. potentials. Model';
                    SNR_cons = 'Real';
                case 2
                    resultset=handles.Results_Interpolation.xhat(1:2039,:);
                    SNR = 'Ep. potentials. Model';
                    SNR_cons = 'Interp';
                case 3
                    resultset=handles.Results_Tikhonov_g0.xhat(1:2039,:);
                    SNR = ['Ep. potentials. ' SNR];
                    SNR_cons = strrep(SNR_cons,'_','\_');
                case 4
                    resultset=handles.Results_Constrained_g1.xhat(1:2039,:);
                    SNR = ['Ep. potentials. ' SNR];
                    SNR_cons = strrep(SNR_cons,'_','\_');
                case 5
                    resultset=handles.Results_Constrained_g2.xhat(1:2039,:);
                    SNR = ['Ep. potentials. ' SNR];
                    SNR_cons = strrep(SNR_cons,'_','\_');
            end
            normal = 3*std(resultset(:));
            resultset=resultset./normal;
            plotting (handles.atrial_model, resultset(:,handles.instant_time), [], -1, 1, 1, 0, handles.instant_time, SNR, SNR_cons, []);
        end
    case 'Video'
        try
            if order_question~=1
                if isunix
                    path_str=strcat(loadpath,'/',loadnamebase,'_flag1.avi');
                    implay(sprintf(path_str));
                else
                    path_str=strcat(loadpath,loadnamebase,'_flag1.avi');
                    implay(path_str);
                end
            else
                ME = MException('Model mismatch','Model selected: %s',order_question);
                throw(ME)
            end
        catch
            switch order_question
                case 1
                    resultset=handles.Ground_Truth.x(1:2039,:);
                    SNR='Ep. potentials. Model';
                    SNR_cons='Real';
                    visualization_file_name=[handles.model,'_Ep_pot_Real', '.mp4'];
                case 2
                    resultset=handles.Results_Interpolation.xhat(1:2039,:);
                    SNR = 'Ep. potentials. Model';
                    SNR_cons = 'Interp';
                    visualization_file_name=[handles.model,'_Ep_pot_Interp', '.mp4'];
                case 3
                    resultset=handles.Results_Tikhonov_g0.xhat(1:2039,:);
                    SNR = ['Ep. potentials. ' SNR];
                    SNR_cons = strrep(SNR_cons,'_','\_');
                    visualization_file_name=[handles.model,'_Ep_pot_Tikhg0_',handles.SNR_BSP,'_',handles.SNR_constrained, '.mp4'];
                case 4
                    resultset=handles.Results_Constrained_g1.xhat(1:2039,:);
                    SNR = ['Ep. potentials. ' SNR];
                    SNR_cons = strrep(SNR_cons,'_','\_');
                    visualization_file_name=[handles.model,'_Ep_pot_Consg1_',handles.SNR_BSP,'_',handles.SNR_constrained, '.mp4'];
                case 5
                    resultset=handles.Results_Constrained_g2.xhat(1:2039,:);
                    SNR = ['Ep. potentials. ' SNR];
                    SNR_cons = strrep(SNR_cons,'_','\_');
                    visualization_file_name=[handles.model,'_Ep_pot_Consg2_',handles.SNR_BSP,'_',handles.SNR_constrained, '.mp4'];
            end
            normal = 3*std(resultset(:));
            resultset=resultset./normal;
            frame_array=plotting (handles.atrial_model, resultset(:,1:120), [], -1, 1, 1, 1, handles.instant_time, SNR, SNR_cons, []);
            for i=1:120
                video_frames(:,:,:,i)=frame2im(frame_array(i));
            end
            video_filename=strjoin(['Figures\',visualization_file_name]);
            video_filename=strrep(video_filename,' ','');
            v = VideoWriter(video_filename,'MPEG-4');
            v.Quality=10;
            open(v),
            writeVideo(v,frame_array),
            close(v)
            
            implay(video_frames,30);
        end
end
guidata(hObject,handles);
% --------------------------------------------------------------------
function DF_maps_Callback(hObject, eventdata, handles)
% hObject    handle to DF_maps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Mensaje preguntando qué orden quiero analizar.
[order_question,v] = listdlg('PromptString','Select a model to analyse:',...
    'SelectionMode','single',...
    'ListString',{'Real','Interp','Tikh g0','Cons g1','Cons g2'});

if isempty(order_question)|| ~v
    return;
end

switch order_question
    case 1
        loadnamebase='GT';
    case 2
        loadnamebase='Interp';
    case 3
        loadnamebase=strcat('Tikhonov_g0',handles.SNR_BSP,'_',handles.SNR_constrained');
    case 4
        loadnamebase=strcat('Constrained_g1',handles.SNR_BSP,'_',handles.SNR_constrained');
    case 5
        loadnamebase=strcat('Constrained_g2',handles.SNR_BSP,'_',handles.SNR_constrained');
end
loadpath=handles.simulation_filepath_g0;

try
    if order_question~=1
        if isunix
            path_str=strcat(loadpath,'/',loadnamebase,'flag2.fig');
            handles.analysisfigs{size(handles.analysisfigs+1),1}=openfig(sprintf(path_str));
        else
            path_str=strcat(loadpath,loadnamebase,'flag2.fig');
            openfig(path_str);
        end
    else
        ME = MException('Model mismatch','Model selected: %s',order_question);
        throw(ME)
    end
catch
    BS_question = questdlg('Which DF method do you want to analyse?', ...
        'DF maps', ...
        'BS Corrected','Botterom-Smith','Classical','BS Corrected');
    
    if isempty(BS_question)
        return;
    end
    
    SNR_cons=strcat('Cons=',handles.SNR_constrained);
    SNR=strcat('SNR\_BSP=',num2str(handles.SNR_BSP),'dB');
    switch order_question
        case 1
            if strcmp(BS_question,'Botterom-Smith')
                resultset=handles.Ground_Truth.characteristics.xDF.BS(1:2039);
                SNR='DF (BS). Model';
                SNR_cons='Real (B-S)';
            elseif strcmp(BS_question,'BS Corrected')
                resultset=handles.Ground_Truth.characteristics.xDF.BS_Corrected(1:2039);
                SNR='DF. Model';
                SNR_cons='Real (BS Corrected)';
            else
                resultset=handles.Ground_Truth.characteristics.xDF.Classical(1:2039);
                SNR='DF. Model';
                SNR_cons='Real';
            end
        case 2
            if strcmp(BS_question,'Botterom-Smith')
                resultset=handles.Results_Interpolation.characteristics.DF.BS(1:2039);
                SNR='DF (BS). Model';
                SNR_cons='Interp (B-S)';
            elseif strcmp(BS_question,'BS Corrected')
                resultset=handles.Results_Interpolation.characteristics.DF.BS_Corrected(1:2039);
                SNR='DF (BS Corrected). Model';
                SNR_cons='Interp (BS Corrected)';
            else
                resultset=handles.Results_Interpolation.characteristics.DF.Classical(1:2039);
                SNR='DF. Model';
                SNR_cons='Interp';
            end
        case 3
            if strcmp(BS_question,'Botterom-Smith')
                resultset=handles.Results_Tikhonov_g0.characteristics.DF.BS(1:2039);
                SNR = ['DF (BS). ' SNR];
                SNR_cons = strrep(SNR_cons,'_','\_');
            elseif strcmp(BS_question,'BS Corrected')
                resultset=handles.Results_Tikhonov_g0.characteristics.DF.BS_Corrected(1:2039);
                SNR = ['DF (BS Corrected). ' SNR];
                SNR_cons = strrep(SNR_cons,'_','\_');
            else
                resultset=handles.Results_Tikhonov_g0.characteristics.DF.Classical(1:2039);
                SNR = ['DF. ' SNR];
                SNR_cons = strrep(SNR_cons,'_','\_');
            end
        case 4
            if strcmp(BS_question,'Botterom-Smith')
                resultset=handles.Results_Constrained_g1.characteristics.DF.BS(1:2039);
                SNR = ['DF (BS). ' SNR];
                SNR_cons = strrep(SNR_cons,'_','\_');
            elseif strcmp(BS_question,'BS Corrected')
                resultset=handles.Results_Constrained_g1.characteristics.DF.BS_Corrected(1:2039);
                SNR = ['DF (BS Corrected). ' SNR];
                SNR_cons = strrep(SNR_cons,'_','\_');
            else
                resultset=handles.Results_Constrained_g1.characteristics.DF.Classical(1:2039);
                SNR = ['DF.' SNR];
                SNR_cons = strrep(SNR_cons,'_','\_');
            end
        case 5
            if strcmp(BS_question,'Botterom-Smith')
                resultset=handles.Results_Constrained_g2.characteristics.DF.BS(1:2039);
                SNR = ['DF (BS). ' SNR];
                SNR_cons = strrep(SNR_cons,'_','\_');
            elseif strcmp(BS_question,'BS Corrected')
                resultset=handles.Results_Constrained_g2.characteristics.DF.BS_Corrected(1:2039);
                SNR = ['DF (BS Corrected). ' SNR];
                SNR_cons = strrep(SNR_cons,'_','\_');
            else
                resultset=handles.Results_Constrained_g2.characteristics.DF.Classical(1:2039);
                SNR = ['DF. ' SNR];
                SNR_cons = strrep(SNR_cons,'_','\_');
            end
    end
    plotting (handles.atrial_model, resultset, [], 0, 10, 2, 0, handles.instant_time, SNR, SNR_cons, []);
    
end
guidata(hObject,handles);

% --------------------------------------------------------------------
function Phase_maps_Callback(hObject, eventdata, handles)
% hObject    handle to Phase_maps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[order_question,v] = listdlg('PromptString','Select a model to analyse:',...
    'SelectionMode','single',...
    'ListString',{'Real','Interp','Tikh g0','Cons g1','Cons g2'});

if isempty(order_question)|| ~v
    return;
end

switch order_question
    case 3
        loadnamebase=strcat('Tikhonov_',handles.orderg0,'_g_',handles.SNR_BSP,'_',handles.SNR_constrained,...
            '_Classical_Tikh');
        loadpath=handles.simulation_filepath_g0;
    case 4
        loadnamebase=strcat('Tikhonov_',handles.orderg1,'_g_',handles.snrg1,'_',handles.snrconsg1,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g1;
    case 5
        loadnamebase=strcat('Tikhonov_',handles.orderg2,'_g_',handles.snrg2,'_',handles.snrconsg2,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g2;
end

% Mensaje preguntando si quiero sacar el video o una imagen estática.
video_question = questdlg('Would you like to view an static map or a video?', ...
    'Phase Maps', ...
    'Image','Video','Image');
switch video_question
    case 'Image'
        try
            if order_question~=1
                if isunix
                    path_str=strcat(loadpath,'/',loadnamebase,'flag3.fig');
                    openfig(sprintf(path_str));
                else
                    path_str=strcat(loadpath,loadnamebase,'flag3.fig');
                    openfig(path_str);
                end
            else
                ME = MException('Model mismatch','Model selected: %s',order_question);
                throw(ME)
            end
        catch
            switch order_question
                case 1
                    resultset=handles.phase_real(1:2039,:);
                    SNR='Phase maps. Model';
                    SNR_cons='Real';
                case 2
                    resultset=handles.phase_data.interp(1:2039,:);
                    SNR='Phase maps. Model';
                    SNR_cons='Interp';
                case 3
                    resultset=handles.phase_data.tikhg0(1:2039,:);
                    SNR=handles.SNR_BSP;
                    SNR = ['Phase maps. ' SNR];
                    SNR_cons=handles.SNR_constrained;
                    SNR_cons = strrep(SNR_cons,'_','\_');
                case 4
                    resultset=handles.phase_data.consg1(1:2039,:);
                    SNR=handles.snrg1;
                    SNR = ['Phase maps. ' SNR];
                    SNR_cons=handles.snrconsg1;
                    SNR_cons = strrep(SNR_cons,'_','\_');
                case 5
                    resultset=handles.phase_data.consg2(1:2039,:);
                    SNR=handles.snrg2;
                    SNR = ['Phase maps. ' SNR];
                    SNR_cons=handles.snrconsg2;
                    SNR_cons = strrep(SNR_cons,'_','\_');
            end
            plotting (handles.atrial_model, resultset(:,handles.instant_time), [], -pi, pi, 3, 0, handles.instant_time, SNR, SNR_cons, []);
            
        end
    case 'Video'
        try
            if order_question~=1
                if isunix
                    path_str=strcat(loadpath,'/',loadnamebase,'_flag3.avi');
                    implay(sprintf(path_str));
                else
                    path_str=strcat(loadpath,loadnamebase,'_flag3.avi');
                    implay(path_str);
                end
            else
                ME = MException('Model mismatch','Model selected: %s',order_question);
                throw(ME)
            end
        catch
            switch order_question
                case 1
                    resultset=handles.phase_real(1:2039,:);
                    SNR='Phase maps. Model';
                    SNR_cons='Real';
                    visualization_file_name=[handles.model,'_Phase_map_Real', '.mp4'];
                case 2
                    resultset=handles.phase_data.interp(1:2039,:);
                    SNR='Phase maps. Model';
                    SNR_cons='Interp';
                    visualization_file_name=[handles.model,'_Phase_map_Interp', '.mp4'];
                case 3
                    resultset=handles.phase_data.tikhg0(1:2039,:);
                    SNR=handles.SNR_BSP;
                    SNR = ['Phase maps. ' SNR];
                    SNR_cons=handles.SNR_constrained;
                    SNR_cons = strrep(SNR_cons,'_','\_');
                    visualization_file_name=[handles.model,'_Phase_map_Tikhg0_',handles.SNR_BSP,'_',handles.SNR_constrained, '.mp4'];
                case 4
                    resultset=handles.phase_data.consg1(1:2039,:);
                    SNR=handles.snrg1;
                    SNR = ['Phase maps. ' SNR];
                    SNR_cons=handles.snrconsg1;
                    SNR_cons = strrep(SNR_cons,'_','\_');
                    visualization_file_name=[handles.model,'_Phase_map_Consg1_',handles.snrg1,'_',handles.snrconsg1, '.mp4'];
                case 5
                    resultset=handles.phase_data.consg2(1:2039,:);
                    SNR=handles.snrg2;
                    SNR = ['Phase maps. ' SNR];
                    SNR_cons=handles.snrconsg2;
                    SNR_cons = strrep(SNR_cons,'_','\_');                    
                    visualization_file_name=[handles.model,'_Phase_map_Consg2_',handles.snrg2,'_',handles.snrconsg2, '.mp4'];
            end
            frame_array=plotting (handles.atrial_model, resultset(:,1:120), [], -pi, pi, 3, 1, handles.instant_time, SNR, SNR_cons, []);
            %frame_array=plotting (handles.atrial_model, resultset, [], -pi, pi, 3, 1, handles.instant_time, SNR, SNR_cons, []);

            for i=1:120
                video_frames(:,:,:,i)=frame2im(frame_array(i));
            end
            
            video_filename=strjoin(['Figures\',visualization_file_name]);
            video_filename=strrep(video_filename,' ','');
            v = VideoWriter(video_filename,'MPEG-4');
            v.Quality=10;
            open(v),
            writeVideo(v,frame_array)
            close(v)
                      
            implay(video_frames,30);
        end
end
guidata(hObject,handles);

% --------------------------------------------------------------------
function Driver_position_Callback(hObject, eventdata, handles)
% hObject    handle to Driver_position (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if strcmp(handles.model,'SR')
    errordlg('The model selected is not an AF model','Error');
    return;
end

[order_question,v] = listdlg('PromptString','Select a model to analyse:',...
    'SelectionMode','single',...
    'ListString',{'Real','Interp','Tikh g0','Cons g1','Cons g2'});


if isempty(order_question)||~v
    return;
end

switch order_question
    case 2
        loadnamebase=strcat('Tikhonov_',handles.orderg0,'_g_',handles.SNR_BSP,'_',handles.SNR_constrained,...
            '_Classical_Tikh');
        loadpath=handles.simulation_filepath_g0;
    case 3
        loadnamebase=strcat('Tikhonov_',handles.orderg1,'_g_',handles.snrg1,'_',handles.snrconsg1,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g1;
    case 4
        loadnamebase=strcat('Tikhonov_',handles.orderg2,'_g_',handles.snrg2,'_',handles.snrconsg2,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g2;
end

% Mensaje preguntando si quiero sacar el video o una imagen estática.
video_question = questdlg('Would you like to view an static map or a video?', ...
    'Driver position', ...
    'Image','Video','Image');
switch video_question
    case 'Image'
        try
            if order_question~=1
                if isunix
                    path_str=strcat(loadpath,'/',loadnamebase,'flag4.fig');
                    openfig(sprintf(path_str));
                else
                    path_str=strcat(loadpath,loadnamebase,'flag4.fig');
                    openfig(path_str);
                end
            else
                ME = MException('Model mismatch','Model selected: %s',order_question);
                throw(ME)
            end
        catch
            switch order_question
                case 1
                    resultset=handles.SMF_real(1:2039,:);
                    SNR='SMF. Model';
                    SNR_cons='Real';
                case 2
                    resultset=handles.driver_data.interp.SMF(1:2039,:);
                    SNR='SMF. Model';
                    SNR_cons='Interp';
                case 3
                    resultset=handles.driver_data.tikhg0.SMF(1:2039,:);
                    SNR=handles.SNR_BSP;
                    SNR = ['SMF. ' SNR];
                    SNR_cons=handles.SNR_constrained;
                    SNR_cons = strrep(SNR_cons,'_','\_');
                case 4
                    resultset=handles.driver_data.consg1.SMF(1:2039,:);
                    SNR=handles.snrg1;
                    SNR = ['SMF. ' SNR];
                    SNR_cons=handles.snrconsg1;
                    SNR_cons = strrep(SNR_cons,'_','\_');
                case 5
                    resultset=handles.driver_data.consg2.SMF(1:2039,:);
                    SNR=handles.snrg2;
                    SNR = ['SMF. ' SNR];
                    SNR_cons=handles.snrconsg2;
                    SNR_cons = strrep(SNR_cons,'_','\_');
            end
%             if strcmp(handles.model, 'SAF')
                plotting (handles.atrial_model, resultset, [], 0, 0.1, 4, 0, handles.instant_time, SNR, SNR_cons, []);
%             else
%                 plotting (handles.atrial_model, resultset, [], 0, 0.4, 4, 0, handles.instant_time, SNR, SNR_cons, []);
%             end
        end
    case 'Video'
        try
            if order_question~=1
                if isunix
                    path_str=strcat(loadpath,'/',loadnamebase,'_flag4.avi');
                    implay(sprintf(path_str));
                else
                    path_str=strcat(loadpath,loadnamebase,'_flag4.avi');
                    implay(path_str);
                end
            else
                ME = MException('Model mismatch','Model selected: %s',order_question);
                throw(ME)
            end
        catch
            switch order_question
                case 1
                    resultset=handles.driver_real;
                    SNR='Driver position. Model';
                    SNR_cons='Real';
                    visualization_file_name=[handles.model,'_Driver_pos_Tikhg0_Real', '.mp4'];
                case 2
                    resultset=handles.driver_data.interp.driver;
                    SNR='Driver position. Model';
                    SNR_cons='Interp';
                    visualization_file_name=[handles.model,'_Driver_pos_Tikhg0_Interp', '.mp4'];
                case 3
                    resultset=handles.driver_data.tikhg0.driver;
                    SNR=handles.SNR_BSP;
                    SNR = ['Driver position. ' SNR];
                    SNR_cons=handles.SNR_constrained;
                    SNR_cons = strrep(SNR_cons,'_','\_');
                    visualization_file_name=[handles.model,'_Driver_pos_Tikhg0_',handles.SNR_BSP,'_',handles.SNR_constrained, '.mp4'];
                case 4
                    resultset=handles.driver_data.consg1.driver;
                    SNR=handles.snrg1;
                    SNR = ['Driver position. ' SNR];
                    SNR_cons=handles.snrconsg1;
                    SNR_cons = strrep(SNR_cons,'_','\_');
                    visualization_file_name=[handles.model,'_Driver_pos_Consg1_',handles.snrg1,'_',handles.snrconsg1, '.mp4'];
                case 5
                    resultset=handles.driver_data.consg2.driver;
                    SNR=handles.snrg2;
                    SNR = ['Driver position. ' SNR];
                    SNR_cons=handles.snrconsg2;
                    SNR_cons = strrep(SNR_cons,'_','\_');
                    visualization_file_name=[handles.model,'_Driver_pos_Consg2_',handles.snrg2,'_',handles.snrconsg2, '.mp4'];
            end
            
            drivers2 = zeros(2039,length(resultset(:,selected_node)));
            for i = 1:length(drivers2(selected_node,:))
                if sum(resultset(i,:)) ~= 0
                    drivers2(resultset(i,resultset(i,:)~=0),i) = 1;
                end
            end
            resultset = drivers2;
            clear drivers2;
            
            frame_array=plotting (handles.atrial_model, resultset(:,1:120), [], 0, 1, 4, 1, handles.instant_time, SNR, SNR_cons, []);
            for i=1:120
                video_frames(:,:,:,i)=frame2im(frame_array(i));
            end
            
            video_filename=strjoin(['Figures\',visualization_file_name]);
            video_filename=strrep(video_filename,' ','');
            v = VideoWriter(video_filename,'MPEG-4');
            v.Quality=10;
            open(v),
            writeVideo(v,frame_array)
            close(v)
                      
            implay(video_frames,30);
        end
end
guidata(hObject,handles);

% --------------------------------------------------------------------
function Time_metrics_Callback(hObject, eventdata, handles)
% hObject    handle to Time_metrics (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function RAE_maps_Callback(hObject, eventdata, handles)
% hObject    handle to RAE_maps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[order_question,v] = listdlg('PromptString','Select a model to analyse:',...
    'SelectionMode','single',...
    'ListString',{'Interp','Tikh g0','Cons g1','Cons g2'});

if isempty(order_question)||~v
    return;
end

switch order_question
    case 2
        loadnamebase=strcat('Tikhonov_',handles.orderg0,'_g_',handles.SNR_BSP,'_',handles.SNR_constrained,...
            '_Classical_Tikh');
        loadpath=handles.simulation_filepath_g0;
    case 3
        loadnamebase=strcat('Tikhonov_',handles.orderg1,'_g_',handles.snrg1,'_',handles.snrconsg1,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g1;
    case 4
        loadnamebase=strcat('Tikhonov_',handles.orderg2,'_g_',handles.snrg2,'_',handles.snrconsg2,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g2;
end

try
    if isunix
        path_str=strcat(loadpath,'/',loadnamebase,'flag6.fig');
        openfig(sprintf(path_str));
    else
        path_str=strcat(loadpath,loadnamebase,'flag6.fig');
        openfig(path_str);
    end
catch
    BS_question = questdlg('Which DF method do you want to analyse?', ...
        'DF maps', ...
        'Classical','Botterom-Smith','Classical');
    
    if isempty(BS_question)
        return;
    end
    
    switch order_question
        case 1
            if strcmp(BS_question,'Botterom-Smith')
                resultset=handles.metrics_RAE.RAE_interp_BS(1:2039)';
                SNR='RAE. Model';
                SNR_cons='Interp (B-S)';
            else
                resultset=handles.metrics_RAE.interp.RAE(1:2039);
                SNR='RAE. Model';
                SNR_cons='Interp';
            end
        case 2
            if strcmp(BS_question,'Botterom-Smith')
                resultset=handles.metrics_RAE.RAE_tikhg0_BS(1:2039)';
                SNR=handles.SNR_BSP;
                SNR = ['RAE. ' SNR];
                SNR_cons=handles.SNR_constrained;
                SNR_cons = strrep(SNR_cons,'_','\_');
                SNR_cons = [SNR_cons ' (B-S).'];
            else
                resultset=handles.metrics_RAE.tikhg0.RAE(1:2039);
                SNR=handles.SNR_BSP;
                SNR = ['RAE. ' SNR];
                SNR_cons=handles.SNR_constrained;
                SNR_cons = strrep(SNR_cons,'_','\_');
            end
        case 3
            if strcmp(BS_question,'Botterom-Smith')
                resultset=handles.metrics_RAE.RAE_consg1_BS(1:2039)';
                SNR=handles.snrg1;
                SNR = ['RAE. ' SNR];
                SNR_cons=handles.snrconsg1;
                SNR_cons = strrep(SNR_cons,'_','\_');
                SNR_cons = [SNR_cons ' (B-S).'];
            else
                resultset=handles.metrics_RAE.consg1.RAE(1:2039);
                SNR=handles.snrg1;
                SNR = ['RAE. ' SNR];
                SNR_cons=handles.snrconsg1;
                SNR_cons = strrep(SNR_cons,'_','\_');
            end
        case 4
            if strcmp(BS_question,'Botterom-Smith')
                resultset=handles.metrics_RAE.RAE_consg2_BS(1:2039)';
                SNR=handles.snrg2;
                SNR = ['RAE. ' SNR];
                SNR_cons=handles.snrconsg2;
                SNR_cons = strrep(SNR_cons,'_','\_');
                SNR_cons = [SNR_cons ' (B-S).'];
            else
                resultset=handles.metrics_RAE.consg2.RAE(1:2039);
                SNR=handles.snrg2;
                SNR = ['RAE. ' SNR];
                SNR_cons=handles.snrconsg2;
                SNR_cons = strrep(SNR_cons,'_','\_');
            end
    end
    plotting (handles.atrial_model, resultset, [], 0, 75, 6, 0, handles.instant_time, SNR, SNR_cons, []);
    
end
guidata(hObject,handles);

% --------------------------------------------------------------------
function Phase_metrics_Callback(hObject, eventdata, handles)
% hObject    handle to Phase_metrics (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Phase_RDMS_Callback(hObject, eventdata, handles)
% hObject    handle to Phase_RDMS (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[order_question,v] = listdlg('PromptString','Select a model to analyse:',...
    'SelectionMode','single',...
    'ListString',{'Interp','Tikh g0','Cons g1','Cons g2'});

switch order_question
    case 2
        loadnamebase=strcat('Tikhonov_',handles.orderg0,'_g_',handles.SNR_BSP,'_',handles.SNR_constrained,...
            '_Classical_Tikh');
        loadpath=handles.simulation_filepath_g0;
    case 3
        loadnamebase=strcat('Tikhonov_',handles.orderg1,'_g_',handles.snrg1,'_',handles.snrconsg1,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g1;
    case 4
        loadnamebase=strcat('Tikhonov_',handles.orderg2,'_g_',handles.snrg2,'_',handles.snrconsg2,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g2;
end

if isempty(order_question)||~v
    return;
end
try
    if isunix
        path_str=strcat(loadpath,'/',loadnamebase,'flag7_RDMS.fig');
        openfig(sprintf(path_str));
    else
        path_str=strcat(loadpath,loadnamebase,'flag7_RDMS.fig');
        openfig(path_str);
    end
catch
    switch order_question
        case 1
            resultset=handles.phase_metrics_data.interp.RDMSt_phase(1:2039,:);
            SNR='Phase RDMS. Model';
            SNR_cons='Interp';
        case 2
            resultset=handles.phase_metrics_data.tikhg0.RDMSt_phase(1:2039,:);
            SNR=handles.SNR_BSP;
            SNR = ['Phase RDMS. ' SNR];
            SNR_cons=handles.SNR_constrained;
            SNR_cons = strrep(SNR_cons,'_','\_');
        case 3
            resultset=handles.phase_metrics_data.consg1.RDMSt_phase(1:2039,:);
            SNR=handles.snrg1;
            SNR = ['Phase RDMS. ' SNR];
            SNR_cons=handles.snrconsg1;
            SNR_cons = strrep(SNR_cons,'_','\_');
        case 4
            resultset=handles.phase_metrics_data.consg2.RDMSt_phase(1:2039,:);
            SNR=handles.snrg2;
            SNR = ['Phase RDMS. ' SNR];
            SNR_cons=handles.snrconsg2;
            SNR_cons = strrep(SNR_cons,'_','\_');
    end
    plotting (handles.atrial_model, resultset, [], 0.5, 2, 7, 0, handles.instant_time, SNR, SNR_cons, 0);
    
end
guidata(hObject,handles);

% --------------------------------------------------------------------
function Phase_CC_Callback(hObject, eventdata, handles)
% hObject    handle to Phase_CC (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[order_question,v] = listdlg('PromptString','Select a model to analyse:',...
    'SelectionMode','single',...
    'ListString',{'Interp','Tikh g0','Cons g1','Cons g2'});

switch order_question
    case 2
        loadnamebase=strcat('Tikhonov_',handles.orderg0,'_g_',handles.SNR_BSP,'_',handles.SNR_constrained,...
            '_Classical_Tikh');
        loadpath=handles.simulation_filepath_g0;
    case 3
        loadnamebase=strcat('Tikhonov_',handles.orderg1,'_g_',handles.snrg1,'_',handles.snrconsg1,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g1;
    case 4
        loadnamebase=strcat('Tikhonov_',handles.orderg2,'_g_',handles.snrg2,'_',handles.snrconsg2,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g2;
end

if isempty(order_question)||~v
    return;
end

try
    if isunix
        path_str=strcat(loadpath,'/',loadnamebase,'flag7_CC.fig');
        openfig(sprintf(path_str));
    else
        path_str=strcat(loadpath,loadnamebase,'flag7_CC.fig');
        openfig(path_str);
    end
catch
    switch order_question
        case 1
            resultset=handles.phase_metrics_data.interp.CCt_phase(1:2039,:);
            SNR='Phase CC. Model';
            SNR_cons='Interp';
        case 2
            resultset=handles.phase_metrics_data.tikhg0.CCt_phase(1:2039,:);
            SNR=handles.SNR_BSP;
            SNR = ['Phase CC. ' SNR];
            SNR_cons=handles.SNR_constrained;
            SNR_cons = strrep(SNR_cons,'_','\_');
        case 3
            resultset=handles.phase_metrics_data.consg1.CCt_phase(1:2039,:);
            SNR=handles.snrg1;
            SNR = ['Phase CC. ' SNR];
            SNR_cons=handles.snrconsg1;
            SNR_cons = strrep(SNR_cons,'_','\_');
        case 4
            resultset=handles.phase_metrics_data.consg2.CCt_phase(1:2039,:);
            SNR=handles.snrg2;
            SNR = ['Phase CC. ' SNR];
            SNR_cons=handles.snrconsg2;
            SNR_cons = strrep(SNR_cons,'_','\_');
    end
    plotting (handles.atrial_model, resultset, [], -1, 1, 7, 0, handles.instant_time, SNR, SNR_cons, 1);
    
end
guidata(hObject,handles);

% --------------------------------------------------------------------
function RDMS_maps_Callback(hObject, eventdata, handles)
% hObject    handle to RDMS_maps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[order_question,v] = listdlg('PromptString','Select a model to analyse:',...
    'SelectionMode','single',...
    'ListString',{'Interp','Tikh g0','Cons g1','Cons g2'});

switch order_question
    case 2
        loadnamebase=strcat('Tikhonov_',handles.orderg0,'_g_',handles.SNR_BSP,'_',handles.SNR_constrained,...
            '_Classical_Tikh');
        loadpath=handles.simulation_filepath_g0;
    case 3
        loadnamebase=strcat('Tikhonov_',handles.orderg1,'_g_',handles.snrg1,'_',handles.snrconsg1,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g1;
    case 4
        loadnamebase=strcat('Tikhonov_',handles.orderg2,'_g_',handles.snrg2,'_',handles.snrconsg2,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g2;
end

if isempty(order_question)||~v
    return;
end

try
    if isunix
        path_str=strcat(loadpath,'/',loadnamebase,'flag5_RDMS.fig');
        openfig(sprintf(path_str));
    else
        path_str=strcat(loadpath,loadnamebase,'flag5_RDMS.fig');
        openfig(path_str);
    end
catch
    switch order_question
        case 1
            resultset=handles.estimation_metrics.interp.RDMSt(:,1:2039)';
            SNR='RDMS. Model';
            SNR_cons='Interp';
        case 2
            resultset=handles.estimation_metrics.tikhg0.RDMSt(:,1:2039)';
            SNR=handles.SNR_BSP;
            SNR = ['RDMS. ' SNR];
            SNR_cons=handles.SNR_constrained;
            SNR_cons = strrep(SNR_cons,'_','\_');
        case 3
            resultset=handles.estimation_metrics.consg1.RDMSt(:,1:2039)';
            SNR=handles.snrg1;
            SNR = ['RDMS. ' SNR];
            SNR_cons=handles.snrconsg1;
            SNR_cons = strrep(SNR_cons,'_','\_');
        case 4
            resultset=handles.estimation_metrics.consg2.RDMSt(:,1:2039)';
            SNR=handles.snrg2;
            SNR = ['RDMS. ' SNR];
            SNR_cons=handles.snrconsg2;
            SNR_cons = strrep(SNR_cons,'_','\_');
    end
    plotting (handles.atrial_model, resultset, [], 0.5, 2, 5, 0, handles.instant_time, SNR, SNR_cons, 0);
    
end
guidata(hObject,handles);

% --------------------------------------------------------------------
function CC_maps_Callback(hObject, eventdata, handles)
% hObject    handle to CC_maps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[order_question,v] = listdlg('PromptString','Select a model to analyse:',...
    'SelectionMode','single',...
    'ListString',{'Interp','Tikh g0','Cons g1','Cons g2'});

if isempty(order_question)||~v
    return;
end

switch order_question
    case 2
        loadnamebase=strcat('Tikhonov_',handles.orderg0,'_g_',handles.SNR_BSP,'_',handles.SNR_constrained,...
            '_Classical_Tikh');
        loadpath=handles.simulation_filepath_g0;
    case 3
        loadnamebase=strcat('Tikhonov_',handles.orderg1,'_g_',handles.snrg1,'_',handles.snrconsg1,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g1;
    case 4
        loadnamebase=strcat('Tikhonov_',handles.orderg2,'_g_',handles.snrg2,'_',handles.snrconsg2,...
            '_Cons_Tikh');
        loadpath=handles.simulation_filepath_g2;
end

try
    if isunix
        path_str=strcat(loadpath,'/',loadnamebase,'flag5_CC.fig');
        openfig(sprintf(path_str));
    else
        path_str=strcat(loadpath,loadnamebase,'flag5_CC.fig');
        openfig(path_str);
    end
catch
    switch order_question
        case 1
            resultset=handles.estimation_metrics.interp.CCt(:,1:2039)';
            SNR='CC. Model';
            SNR_cons='Interp';
        case 2
            resultset=handles.estimation_metrics.tikhg0.CCt(:,1:2039)';
            SNR=handles.SNR_BSP;
            SNR = ['CC. ' SNR];
            SNR_cons=handles.SNR_constrained;
            SNR_cons = strrep(SNR_cons,'_','\_');
        case 3
            resultset=handles.estimation_metrics.consg1.CCt(:,1:2039)';
            SNR=handles.snrg1;
            SNR = ['CC. ' SNR];
            SNR_cons=handles.snrconsg1;
            SNR_cons = strrep(SNR_cons,'_','\_');
        case 4
            resultset=handles.estimation_metrics.consg2.CCt(:,1:2039)';
            SNR=handles.snrg2;
            SNR = ['CC. ' SNR];
            SNR_cons=handles.snrconsg2;
            SNR_cons = strrep(SNR_cons,'_','\_');
    end
    plotting (handles.atrial_model, resultset, [], -1, 1, 5, 0, handles.instant_time, SNR, SNR_cons, 1);
    
end
guidata(hObject,handles);

% --------------------------------------------------------------------
function metrics_Callback(hObject, eventdata, handles)
% hObject    handle to metrics (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in estimation_metrics.
function estimation_metrics_Callback(hObject, eventdata, handles)
% hObject    handle to estimation_metrics (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figure(),
subplot(311),
barwitherr([handles.estimation_metrics.interp.stdRDMS handles.estimation_metrics.tikhg0.stdRDMS handles.estimation_metrics.consg1.stdRDMS handles.estimation_metrics.consg2.stdRDMS],...
    [handles.estimation_metrics.interp.MRDMS handles.estimation_metrics.tikhg0.MRDMS handles.estimation_metrics.consg1.MRDMS handles.estimation_metrics.consg2.MRDMS]);
title('MRDMS comparison'),
ylabel('RDMS'),
ylim([0 inf])
set(gca,'XTickLabel',{'Interp','Tikh-g0', 'Cons-g1', 'Cons-g2'})
subplot(312),
barwitherr([handles.estimation_metrics.interp.stdCC handles.estimation_metrics.tikhg0.stdCC handles.estimation_metrics.consg1.stdCC handles.estimation_metrics.consg2.stdCC],...
    [handles.estimation_metrics.interp.MCC handles.estimation_metrics.tikhg0.MCC handles.estimation_metrics.consg1.MCC handles.estimation_metrics.consg2.MCC]);
title('MCC comparison'),
ylabel('CC'),
ylim([0 inf])
set(gca,'XTickLabel',{'Interp','Tikh-g0', 'Cons-g1', 'Cons-g2'})
subplot(313),
barwitherr([handles.estimation_metrics.interp.stdRMSE handles.estimation_metrics.tikhg0.stdRMSE handles.estimation_metrics.consg1.stdRMSE handles.estimation_metrics.consg2.stdRMSE],...
    [handles.estimation_metrics.interp.MRMSE handles.estimation_metrics.tikhg0.MRMSE handles.estimation_metrics.consg1.MRMSE handles.estimation_metrics.consg2.MRMSE]);
title('MRMSE comparison'),
ylabel('RMSE'),
ylim([0 inf])
set(gca,'XTickLabel',{'Interp','Tikh-g0', 'Cons-g1', 'Cons-g2'})

% --- Executes on button press in DF_metrics.
function DF_metrics_Callback(hObject, eventdata, handles)
% hObject    handle to DF_metrics (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figure(),
subplot(211),
barwitherr([handles.metrics_RAE.interp.std_mRAE handles.metrics_RAE.tikhg0.std_mRAE handles.metrics_RAE.consg1.std_mRAE handles.metrics_RAE.consg2.std_mRAE],...
    [handles.metrics_RAE.interp.mRAE handles.metrics_RAE.tikhg0.mRAE handles.metrics_RAE.consg1.mRAE handles.metrics_RAE.consg2.mRAE]);
title('RAE comparison (Standard DF calculation)'),
ylabel('RAE'),
ylim([0 50])
set(gca,'XTickLabel',{'Interp','Tikh-g0', 'Cons-g1', 'Cons-g2'})
subplot(212),
barwitherr([handles.metrics_RAE.std_mRAE_interp_BS handles.metrics_RAE.std_mRAE_tikhg0_BS handles.metrics_RAE.std_mRAE_consg1_BS handles.metrics_RAE.std_mRAE_consg2_BS],...
    [handles.metrics_RAE.mRAE_interp_BS handles.metrics_RAE.mRAE_tikhg0_BS handles.metrics_RAE.mRAE_consg1_BS handles.metrics_RAE.mRAE_consg2_BS]);
title('RAE comparison (Botterom-Smith processing)'),
ylabel('RAE'),
ylim([0 50])
set(gca,'XTickLabel',{'Interp','Tikh-g0', 'Cons-g1', 'Cons-g2'})
% --------------------------------------------------------------------
function phase_metrics_Callback(hObject, eventdata, handles)
% hObject    handle to phase_metrics (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figure(),
subplot(211),
barwitherr([handles.phase_metrics_data.interp.std_MRDMSt_phase handles.phase_metrics_data.tikhg0.std_MRDMSt_phase handles.phase_metrics_data.consg1.std_MRDMSt_phase handles.phase_metrics_data.consg2.std_MRDMSt_phase],...
    [handles.phase_metrics_data.interp.MRDMSt_phase handles.phase_metrics_data.tikhg0.MRDMSt_phase handles.phase_metrics_data.consg1.MRDMSt_phase handles.phase_metrics_data.consg2.MRDMSt_phase]);
title('Phase MRDMS comparison'),
ylabel('Phase RDMS'),
ylim([0 inf])
set(gca,'XTickLabel',{'Interp','Tikh-g0', 'Cons-g1', 'Cons-g2'})
subplot(212),
barwitherr([handles.phase_metrics_data.interp.std_MCCt_phase handles.phase_metrics_data.tikhg0.std_MCCt_phase handles.phase_metrics_data.consg1.std_MCCt_phase handles.phase_metrics_data.consg2.std_MCCt_phase],...
    [handles.phase_metrics_data.interp.MCCt_phase handles.phase_metrics_data.tikhg0.MCCt_phase handles.phase_metrics_data.consg1.MCCt_phase handles.phase_metrics_data.consg2.MCCt_phase]);
title('Phase MCC comparison'),
ylabel('Phase CC'),
ylim([0 inf])
set(gca,'XTickLabel',{'Interp','Tikh-g0', 'Cons-g1', 'Cons-g2'})


% --------------------------------------------------------------------
function driver_metrics_Callback(hObject, eventdata, handles)
% hObject    handle to driver_metrics (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figure(),
subplot(411),
bar([handles.driver_data.interp.WUI handles.driver_data.tikhg0.WUI handles.driver_data.consg1.WUI handles.driver_data.consg2.WUI])
title('WUI comparison'),
ylabel('WUI'),
ylim([0 inf])
set(gca,'XTickLabel',{'Interp','Tikh-g0', 'Cons-g1', 'Cons-g2'})
subplot(412),
bar([handles.driver_data.interp.WOI handles.driver_data.tikhg0.WOI handles.driver_data.consg1.WOI handles.driver_data.consg2.WOI])
title('WOI comparison'),
ylabel('WOI'),
ylim([0 inf])
set(gca,'XTickLabel',{'Interp','Tikh-g0', 'Cons-g1', 'Cons-g2'})
subplot(413),
bar([handles.driver_data.interp.CCdriver handles.driver_data.tikhg0.CCdriver handles.driver_data.consg1.CCdriver handles.driver_data.consg2.CCdriver])
title('CC driver comparison'),
ylabel('CC driver'),
ylim([0 inf])
set(gca,'XTickLabel',{'Interp','Tikh-g0', 'Cons-g1', 'Cons-g2'})
subplot(414),
bar([handles.driver_data.interp.MD handles.driver_data.tikhg0.MD handles.driver_data.consg1.MD handles.driver_data.consg2.MD])
title('MD comparison'),
ylabel('MD driver'),
ylim([0 inf])
set(gca,'XTickLabel',{'Interp','Tikh-g0', 'Cons-g1', 'Cons-g2'})
