function visualization (model, SNR, SNR_cons, available_nodes, metric, front_rear, instant_time, video, time_recording)
% Metrics and graphical results from different reconstruction models.
%
% This routine by Miguel Ángel Cámara Vázquez (miguelangel.camara@urjc.es)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUT:
% Model: Model to be visualized.
% SNR: signal noise rate used to (dBs).
% SNR_cons: EGM signal noise rate used to (dBs).
% metric: metrics which will be visualizated.
% video: 0=only a picture. 1=video.
% time_recording: Only if video is enabled. Video record duration.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Paths
close all;
addpath('toolbox');
data_path    = [pwd '/data/'];
results_path = [pwd '/results/' model '/'];
mkdir([pwd '/visualization/' model '/']);
visualization_path = [pwd '/visualization/' model '/'];

%% Load atrial and torso models.
load([data_path 'geometry.mat']);

%% Load reconstruction file.
if ischar(available_nodes)
    nodes_name = '2basket';
else
    nodes_name = [num2str(available_nodes) 'nodes'];
end
if ischar(SNR_cons)
    SNR_cons_name = 'SNR_consMax';
else
    SNR_cons_name = ['SNR_cons' SNR_cons];
end

model_name = [model '_Tikhonov_g_SNR' num2str(SNR) '_' SNR_cons_name '_' nodes_name];
load([results_path model_name '.mat']);

%% Load visualization parameters:
% Epicardic Potentials
minPot = -1;
maxPot = 1;

% DF
minF = 3;   % low frequency.
maxF = 8; % % high frequency.
if contains(model,'Sinusal')
    instant_time = 826;
    minF = 1;   % low frequency.
    maxF = 1.4; % % high frequency.
end

% SampEn
minSampEn = -3;
maxSampEn = 0;

% OI
minOI = 0;
maxOI = 1;

% RI
minRI = 0;
maxRI = 1;

% SMF
minSMF = 0;
maxSMF = 0.1;

% Driver
minDriver = 0;
maxDriver = 1;

% Phase
minPhase = -pi;
maxPhase = pi;

% RDMS
minRDMS = 0;
maxRDMS = 1.6;

% CC
minCC = -1;
maxCC = 1;

% DTW
minDTW = 0;
maxDTW = 15;

% %% Body Surface Potentials
% if strcmp(metric,'BSPs')
%     normal = 10;    % dataset normalization.
%     fprintf('Tikhonov SNR_%i:\n',SNR);
%     if video
%         my_frames = plotting (torso_model, y./normal, [], -1, 1, flag, video, [], SNR, []);
%         v = VideoWriter([visualization_file_name, '_', 'flag' num2str(flag), '.avi'],'MPEG-4');
%         v.Quality=50;
%         open(v),writeVideo(v,my_frames)
%         close(v)
%         close all;
%     else
%         plotting (torso_model, y(:,instant_time)./normal, [], -1, 1, flag, video, instant_time, SNR, []);
%     end
% end

%% Epicardial potentials
if strcmp(metric,'EpPot')
    % Normalization
    x_GT = Ground_Truth.x;
    normal_GT = 3*std(x_GT(:));
    x_GT=x_GT./normal_GT;
    x_GT = x_GT(1:2039,:);
    
    x_interp = Results_Interpolation.xhat;
    normal_interp = 3*std(x_interp(:));
    x_interp=x_interp./normal_interp;
    x_interp = x_interp(1:2039,:);
    
    x_tikh = Results_Tikhonov_g0.xhat;
    normal_tikh = 3*std(x_tikh(:));
    x_tikh=x_tikh./normal_tikh;
    x_tikh = x_tikh(1:2039,:);
    
    x_cons = Results_Constrained_g1.xhat;
    normal_cons = 3*std(x_cons(:));
    x_cons=x_cons./normal_cons;
    x_cons = x_cons(1:2039,:);
    
    mkdir([visualization_path '/EpPot/']);
    Ep_filepath = [visualization_path '/EpPot/EpPot_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name];
    
    if ~video
        x_GT = x_GT(:,instant_time);
        x_interp = x_interp(:,instant_time);
        x_tikh = x_tikh(:,instant_time);
        x_cons = x_cons(:,instant_time);
        
        plotting (atrial_model, model, metric, x_GT, x_interp, x_tikh, x_cons, minPot, maxPot, front_rear, instant_time, video, time_recording);
        
        saveas(gcf,Ep_filepath,'png');
        saveas(gcf,Ep_filepath,'fig');
        saveas(gcf,Ep_filepath,'epsc');
    else
        frame_array = plotting (atrial_model, model, metric, x_GT, x_interp, x_tikh, x_cons, minPot, maxPot, front_rear, [], video, time_recording);
        v = VideoWriter(Ep_filepath,'MPEG-4');
        v.Quality=50;
        open(v),
        writeVideo(v,frame_array),
        close(v)
    end
end

%% Dominant Frequency

if strcmp(metric,'DF_Classical')
    
    xDF = Ground_Truth.characteristics.xDF.Classical(1:2039);
    DF_interp = Results_Interpolation.characteristics.DF.Classical(1:2039);
    DF_tikh = Results_Tikhonov_g0.characteristics.DF.Classical(1:2039);
    DF_cons = Results_Constrained_g1.characteristics.DF.Classical(1:2039);
    
    mkdir([visualization_path '/DF/']);
    DF_filepath = [visualization_path '/DF/DF_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name '_Classical'];
    
    plotting (atrial_model, model, metric, xDF, DF_interp, DF_tikh, DF_cons, minF, maxF, front_rear, [], video, time_recording);
    
    saveas(gcf,DF_filepath,'png');
    saveas(gcf,DF_filepath,'fig');
    saveas(gcf,DF_filepath,'epsc');
end

if strcmp(metric,'DF_BS_Corrected')
    
    xDF = Ground_Truth.characteristics.xDF.BS_Corrected(1:2039);
    DF_interp = Results_Interpolation.characteristics.DF.BS_Corrected(1:2039);
    DF_tikh = Results_Tikhonov_g0.characteristics.DF.BS_Corrected(1:2039);
    DF_cons = Results_Constrained_g1.characteristics.DF.BS_Corrected(1:2039);
    
    mkdir([visualization_path '/DF/']);
    DF_filepath = [visualization_path '/DF/DF_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name '_BS_Corrected'];
    
    plotting (atrial_model, model, metric, xDF, DF_interp, DF_tikh, DF_cons, minF, maxF, front_rear, [], video, time_recording);
    
    saveas(gcf,DF_filepath,'png');
    saveas(gcf,DF_filepath,'fig');
    saveas(gcf,DF_filepath,'epsc');
    
    set(gcf,'Units','Inches');
    pos_pdf = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos_pdf(3), pos_pdf(4)])
    print(gcf,DF_filepath,'-dpdf','-painters');

end

if strcmp(metric,'DF_BS')
    
    xDF = Ground_Truth.characteristics.xDF.BS(1:2039);
    DF_interp = Results_Interpolation.characteristics.DF.BS(1:2039);
    DF_tikh = Results_Tikhonov_g0.characteristics.DF.BS(1:2039);
    DF_cons = Results_Constrained_g1.characteristics.DF.BS(1:2039);
    
    mkdir([visualization_path '/DF/']);
    DF_filepath = [visualization_path '/DF/DF_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name '_BS'];
    
    plotting (atrial_model, model, metric, xDF, DF_interp, DF_tikh, DF_cons, minF, maxF, front_rear, [], video, time_recording);
    
    saveas(gcf,DF_filepath,'png');
    saveas(gcf,DF_filepath,'fig');
    saveas(gcf,DF_filepath,'epsc');
end

%% SampEn
if strcmp(metric,'SampEn')
    
    xsampen = log(Ground_Truth.characteristics.sampen(1:2039)');
    sampen_interp = log(Results_Interpolation.characteristics.sampen(1:2039)');
    sampen_tikh = log(Results_Tikhonov_g0.characteristics.sampen(1:2039)');
    sampen_cons = log(Results_Constrained_g1.characteristics.sampen(1:2039)');
    
    mkdir([visualization_path '/SampEn/']);
    SampEn_filepath = [visualization_path '/SampEn/' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name];
    
    plotting (atrial_model, model, metric, xsampen, sampen_interp, sampen_tikh, sampen_cons, minSampEn, maxSampEn, front_rear, [], video, time_recording);
    
    saveas(gcf,SampEn_filepath,'png');
    saveas(gcf,SampEn_filepath,'fig');
    saveas(gcf,SampEn_filepath,'epsc');
end

%% Organization Indexes
if strcmp(metric,'Org_Index')
    
    xOI = Ground_Truth.characteristics.xOI(1:2039)';
    OI_interp = Results_Interpolation.characteristics.OI(1:2039)';
    OI_tikh = Results_Tikhonov_g0.characteristics.OI(1:2039)';
    OI_cons = Results_Constrained_g1.characteristics.OI(1:2039)';
    
    xRI = Ground_Truth.characteristics.xRI(1:2039)';
    RI_interp = Results_Interpolation.characteristics.RI(1:2039)';
    RI_tikh = Results_Tikhonov_g0.characteristics.RI(1:2039)';
    RI_cons = Results_Constrained_g1.characteristics.RI(1:2039)';
    
    mkdir([visualization_path '/RI_OI/']);
    OI_filepath = [visualization_path '/RI_OI/OI_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name];
    
    % OI
    temp_metric = 'OI';
    plotting (atrial_model, model, temp_metric, xOI, OI_interp, OI_tikh, OI_cons, minOI, maxOI, front_rear, [], video, time_recording);
    
    saveas(gcf,OI_filepath,'png');
    saveas(gcf,OI_filepath,'fig');
    saveas(gcf,OI_filepath,'epsc');
    
    
    % RI
    temp_metric = 'RI';
    RI_filepath = [visualization_path '/RI_OI/RI_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name];
    
    plotting (atrial_model, model, temp_metric, xRI, RI_interp, RI_tikh, RI_cons, minRI, maxRI, front_rear, [], video, time_recording);
    
    saveas(gcf,RI_filepath,'png');
    saveas(gcf,RI_filepath,'fig');
    saveas(gcf,RI_filepath,'epsc');
end

%% Phase maps
if strcmp(metric,'Phase_Classical')
    phase_GT = Ground_Truth.xphase.Classical(1:2039,:);
    phase_interp = Results_Interpolation.phase.Classical(1:2039,:);
    phase_tikh = Results_Tikhonov_g0.phase.Classical(1:2039,:);
    phase_cons = Results_Constrained_g1.phase.Classical(1:2039,:);
    
    mkdir([visualization_path '/Phase/']);
    Phase_filepath = [visualization_path '/Phase/Phase_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name '_Classical'];
    
    if ~video
        phase_GT = phase_GT(:,instant_time);
        phase_interp = phase_interp(:,instant_time);
        phase_tikh = phase_tikh(:,instant_time);
        phase_cons = phase_cons(:,instant_time);
        
        plotting (atrial_model, model, metric, phase_GT, phase_interp, phase_tikh, phase_cons, minPhase, maxPhase, front_rear, instant_time, video, time_recording);
        
        saveas(gcf,Phase_filepath,'png');
        saveas(gcf,Phase_filepath,'fig');
        saveas(gcf,Phase_filepath,'epsc');
    else
        frame_array = plotting (atrial_model, model, metric, phase_GT, phase_interp, phase_tikh, phase_cons, minPhase, maxPhase, front_rear, [], video, time_recording);
        v = VideoWriter(Phase_filepath,'MPEG-4');
        v.Quality=50;
        open(v),
        writeVideo(v,frame_array),
        close(v)
    end
elseif strcmp(metric,'Phase_BS_Corrected')
    phase_GT = Ground_Truth.xphase.BS_Corrected(1:2039,:);
    phase_interp = Results_Interpolation.phase.BS_Corrected(1:2039,:);
    phase_tikh = Results_Tikhonov_g0.phase.BS_Corrected(1:2039,:);
    phase_cons = Results_Constrained_g1.phase.BS_Corrected(1:2039,:);
    
    mkdir([visualization_path '/Phase/']);
    Phase_filepath = [visualization_path '/Phase/Phase_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name '_BS_Corrected'];
    
    if ~video
        phase_GT = phase_GT(:,instant_time);
        phase_interp = phase_interp(:,instant_time);
        phase_tikh = phase_tikh(:,instant_time);
        phase_cons = phase_cons(:,instant_time);
        
        plotting (atrial_model, model, metric, phase_GT, phase_interp, phase_tikh, phase_cons, minPhase, maxPhase, front_rear, instant_time, video, time_recording);
        
        saveas(gcf,Phase_filepath,'png');
        saveas(gcf,Phase_filepath,'fig');
        saveas(gcf,Phase_filepath,'epsc');
    else
        frame_array = plotting (atrial_model, model, metric, phase_GT, phase_interp, phase_tikh, phase_cons, minPhase, maxPhase, front_rear, [], video, time_recording);
        v = VideoWriter(Phase_filepath,'MPEG-4');
        v.Quality=50;
        open(v),
        writeVideo(v,frame_array),
        close(v)
    end
elseif strcmp(metric,'Phase_BS')
    phase_GT = Ground_Truth.xphase.BS(1:2039,:);
    phase_interp = Results_Interpolation.phase.BS(1:2039,:);
    phase_tikh = Results_Tikhonov_g0.phase.BS(1:2039,:);
    phase_cons = Results_Constrained_g1.phase.BS(1:2039,:);
    
    mkdir([visualization_path '/Phase/']);
    Phase_filepath = [visualization_path '/Phase/Phase_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name '_BS'];
    
    if ~video
        phase_GT = phase_GT(:,instant_time);
        phase_interp = phase_interp(:,instant_time);
        phase_tikh = phase_tikh(:,instant_time);
        phase_cons = phase_cons(:,instant_time);
        
        plotting (atrial_model, model, metric, phase_GT, phase_interp, phase_tikh, phase_cons, minPhase, maxPhase, front_rear, instant_time, video, time_recording);
        
        saveas(gcf,Phase_filepath,'png');
        saveas(gcf,Phase_filepath,'fig');
        saveas(gcf,Phase_filepath,'epsc');
    else
        frame_array = plotting (atrial_model, model, metric, phase_GT, phase_interp, phase_tikh, phase_cons, minPhase, maxPhase, front_rear, [], video, time_recording);
        v = VideoWriter(Phase_filepath,'MPEG-4');
        v.Quality=50;
        open(v),
        writeVideo(v,frame_array),
        close(v)
    end  
elseif strcmp(metric,'Phase_noHDF')
    phase_GT = Ground_Truth.xphase.noHDF(1:2039,:);
    phase_interp = Results_Interpolation.phase.noHDF(1:2039,:);
    phase_tikh = Results_Tikhonov_g0.phase.noHDF(1:2039,:);
    phase_cons = Results_Constrained_g1.phase.noHDF(1:2039,:);
    
    mkdir([visualization_path '/Phase/']);
    Phase_filepath = [visualization_path '/Phase/Phase_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name '_noHDF'];
    
    if ~video
        phase_GT = phase_GT(:,instant_time);
        phase_interp = phase_interp(:,instant_time);
        phase_tikh = phase_tikh(:,instant_time);
        phase_cons = phase_cons(:,instant_time);
        
        plotting (atrial_model, model, metric, phase_GT, phase_interp, phase_tikh, phase_cons, minPhase, maxPhase, front_rear, instant_time, video, time_recording);
        
        saveas(gcf,Phase_filepath,'png');
        saveas(gcf,Phase_filepath,'fig');
        saveas(gcf,Phase_filepath,'epsc');
        
        set(gcf,'Units','Inches');
        pos_pdf = get(gcf,'Position');
        set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos_pdf(3), pos_pdf(4)])
        print(gcf,Phase_filepath,'-dpdf','-painters');
    
    else
        frame_array = plotting (atrial_model, model, metric, phase_GT, phase_interp, phase_tikh, phase_cons, minPhase, maxPhase, front_rear, [], video, time_recording);
        v = VideoWriter(Phase_filepath,'MPEG-4');
        v.Quality=50;
        open(v),
        writeVideo(v,frame_array),
        close(v)
    end   
end

%% Driver position
if contains(metric,'Driver')
        
    mkdir([visualization_path '/Driver/']);
    
    if contains(metric,'noHDF')
        if strcmp(metric,'Driver_Classical_noHDF')
            Driver_filepath = [visualization_path '/Driver/Driver_' nodes_name...
                '_SNR' num2str(SNR) '_' SNR_cons_name '_Classical_noHDF'];
        elseif strcmp(metric,'Driver_BS_Corrected_noHDF')
            Driver_filepath = [visualization_path '/Driver/Driver_' nodes_name...
                '_SNR' num2str(SNR) '_' SNR_cons_name '_BS_Corrected_noHDF'];
        elseif strcmp(metric,'Driver_BS_noHDF')
            Driver_filepath = [visualization_path '/Driver/Driver_' nodes_name...
                '_SNR' num2str(SNR) '_' SNR_cons_name '_BS_noHDF'];
        end
        
        if ~video
            if strcmp(metric,'Driver_Classical_noHDF')
                try
                    xSMF = Ground_Truth.xSMF_noHDF.Classical(1:2039);
                catch
                    xSMF = zeros(2039,1);
                end
                try
                    SMF_interp = Results_Interpolation.driver_data_noHDF.Classical.SMF(1:2039);
                catch
                    SMF_interp = zeros(2039,1);
                end
                try
                    SMF_tikh = Results_Tikhonov_g0.driver_data_noHDF.Classical.SMF(1:2039);
                catch
                    SMF_tikh = zeros(2039,1);
                end
                try
                    SMF_cons = Results_Constrained_g1.driver_data_noHDF.Classical.SMF(1:2039);
                catch
                    SMF_cons = zeros(2039,1);
                end
            elseif strcmp(metric,'Driver_BS_Corrected_noHDF')
                try
                    xSMF = Ground_Truth.xSMF_noHDF.BS_Corrected(1:2039);
                catch
                    xSMF = zeros(2039,1);
                end
                try
                    SMF_interp = Results_Interpolation.driver_data_noHDF.BS_Corrected.SMF(1:2039);
                catch
                    SMF_interp = zeros(2039,1);
                end
                try
                    SMF_tikh = Results_Tikhonov_g0.driver_data_noHDF.BS_Corrected.SMF(1:2039);
                catch
                    SMF_tikh = zeros(2039,1);
                end
                try
                    SMF_cons = Results_Constrained_g1.driver_data_noHDF.BS_Corrected.SMF(1:2039);
                catch
                    SMF_cons = zeros(2039,1);
                end
            elseif strcmp(metric,'Driver_BS_noHDF')
                try
                    xSMF = Ground_Truth.xSMF_noHDF.BS(1:2039);
                catch
                    xSMF = zeros(2039,1);
                end
                try
                    SMF_interp = Results_Interpolation.driver_data_noHDF.BS.SMF(1:2039);
                catch
                    SMF_interp = zeros(2039,1);
                end
                try
                    SMF_tikh = Results_Tikhonov_g0.driver_data_noHDF.BS.SMF(1:2039);
                catch
                    SMF_tikh = zeros(2039,1);
                end
                try
                    SMF_cons = Results_Constrained_g1.driver_data_noHDF.BS.SMF(1:2039);
                catch
                    SMF_cons = zeros(2039,1);
                end
            end
            
            plotting (atrial_model, model, metric, xSMF, SMF_interp, SMF_tikh, SMF_cons, minSMF, maxSMF, front_rear, [], video, time_recording);
            
            saveas(gcf,Driver_filepath,'png');
            saveas(gcf,Driver_filepath,'fig');
            saveas(gcf,Driver_filepath,'epsc');
        else
            % Xdriver
            if strcmp(metric,'Driver_Classical_noHDF')
                try
                    drivers = Ground_Truth.xdriver_noHDF.Classical;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    xdriver = drivers2;
                    clear drivers2;
                catch
                    xdriver = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Interpolation.driver_data_noHDF.Classical.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_interp = drivers2;
                    clear drivers2;
                catch
                    driver_interp = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Tikhonov_g0.driver_data_noHDF.Classical.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_tikh = drivers2;
                    clear drivers2;
                catch
                    driver_tikh = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Constrained_g1.driver_data_noHDF.Classical.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_cons = drivers2;
                    clear drivers2;
                catch
                    driver_cons = zeros(2039,time_recording);
                end
                
            elseif strcmp(metric,'Driver_BS_Corrected_noHDF')
                try
                    drivers = Ground_Truth.xdriver_noHDF.BS_Corrected;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    xdriver = drivers2;
                    clear drivers2;
                catch
                    xdriver = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Interpolation.driver_data_noHDF.BS_Corrected.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_interp = drivers2;
                    clear drivers2;
                catch
                    driver_interp = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Tikhonov_g0.driver_data_noHDF.BS_Corrected.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_tikh = drivers2;
                    clear drivers2;
                catch
                    driver_tikh = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Constrained_g1.driver_data_noHDF.BS_Corrected.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_cons = drivers2;
                    clear drivers2;
                catch
                    driver_cons = zeros(2039,time_recording);
                end
                
            elseif strcmp(metric,'Driver_BS_noHDF')
                try
                    drivers = Ground_Truth.xdriver_noHDF.BS;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    xdriver = drivers2;
                    clear drivers2;
                catch
                    xdriver = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Interpolation.driver_data_noHDF.BS.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_interp = drivers2;
                    clear drivers2;
                catch
                    driver_interp = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Tikhonov_g0.driver_data_noHDF.BS.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_tikh = drivers2;
                    clear drivers2;
                catch
                    driver_tikh = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Constrained_g1.driver_data_noHDF.BS.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_cons = drivers2;
                    clear drivers2;
                catch
                    driver_cons = zeros(2039,time_recording);
                end
            end
            
            frame_array = plotting (atrial_model, model, metric, xdriver, driver_interp, driver_tikh, driver_cons, minDriver, maxDriver, front_rear, [], video, time_recording);
            
            v = VideoWriter(Driver_filepath,'MPEG-4');
            v.Quality=50;
            open(v),
            writeVideo(v,frame_array),
            close(v)
        end
    else     
        if strcmp(metric,'Driver_Classical')
            Driver_filepath = [visualization_path '/Driver/Driver_' nodes_name...
                '_SNR' num2str(SNR) '_' SNR_cons_name '_Classical'];
        elseif strcmp(metric,'Driver_BS_Corrected')
            Driver_filepath = [visualization_path '/Driver/Driver_' nodes_name...
                '_SNR' num2str(SNR) '_' SNR_cons_name '_BS_Corrected'];
        elseif strcmp(metric,'Driver_BS')
            Driver_filepath = [visualization_path '/Driver/Driver_' nodes_name...
                '_SNR' num2str(SNR) '_' SNR_cons_name '_BS'];
        end
        
        if ~video
            if strcmp(metric,'Driver_Classical')
                try
                    xSMF = Ground_Truth.xSMF.Classical(1:2039);
                catch
                    xSMF = zeros(2039,1);
                end
                try
                    SMF_interp = Results_Interpolation.driver_data.Classical.SMF(1:2039);
                catch
                    SMF_interp = zeros(2039,1);
                end
                try
                    SMF_tikh = Results_Tikhonov_g0.driver_data.Classical.SMF(1:2039);
                catch
                    SMF_tikh = zeros(2039,1);
                end
                try
                    SMF_cons = Results_Constrained_g1.driver_data.Classical.SMF(1:2039);
                catch
                    SMF_cons = zeros(2039,1);
                end
            elseif strcmp(metric,'Driver_BS_Corrected')
                try
                    xSMF = Ground_Truth.xSMF.BS_Corrected(1:2039);
                catch
                    xSMF = zeros(2039,1);
                end
                try
                    SMF_interp = Results_Interpolation.driver_data.BS_Corrected.SMF(1:2039);
                catch
                    SMF_interp = zeros(2039,1);
                end
                try
                    SMF_tikh = Results_Tikhonov_g0.driver_data.BS_Corrected.SMF(1:2039);
                catch
                    SMF_tikh = zeros(2039,1);
                end
                try
                    SMF_cons = Results_Constrained_g1.driver_data.BS_Corrected.SMF(1:2039);
                catch
                    SMF_cons = zeros(2039,1);
                end
            elseif strcmp(metric,'Driver_BS')
                try
                    xSMF = Ground_Truth.xSMF.BS(1:2039);
                catch
                    xSMF = zeros(2039,1);
                end
                try
                    SMF_interp = Results_Interpolation.driver_data.BS.SMF(1:2039);
                catch
                    SMF_interp = zeros(2039,1);
                end
                try
                    SMF_tikh = Results_Tikhonov_g0.driver_data.BS.SMF(1:2039);
                catch
                    SMF_tikh = zeros(2039,1);
                end
                try
                    SMF_cons = Results_Constrained_g1.driver_data.BS.SMF(1:2039);
                catch
                    SMF_cons = zeros(2039,1);
                end
                
                
            elseif contains(metric,'Kuklik')
                try
                    xSMF = Ground_Truth.xSMF.Kuklik(1:2039);
                catch
                    xSMF = zeros(2039,1);
                end
                try
                    SMF_interp = Results_Interpolation.driver_data.Kuklik.SMF(1:2039);
                catch
                    SMF_interp = zeros(2039,1);
                end
                try
                    SMF_tikh = Results_Tikhonov_g0.driver_data.Kuklik.SMF(1:2039);
                catch
                    SMF_tikh = zeros(2039,1);
                end
                try
                    SMF_cons = Results_Constrained_g1.driver_data.Kuklik.SMF(1:2039);
                catch
                    SMF_cons = zeros(2039,1);
                end
                
                
            elseif contains(metric,'MR2017')
                try
                    xSMF = Ground_Truth.xSMF.MR2017(1:2039);
                catch
                    xSMF = zeros(2039,1);
                end
                try
                    SMF_interp = Results_Interpolation.driver_data.MR2017.SMF(1:2039);
                catch
                    SMF_interp = zeros(2039,1);
                end
                try
                    SMF_tikh = Results_Tikhonov_g0.driver_data.MR2017.SMF(1:2039);
                catch
                    SMF_tikh = zeros(2039,1);
                end
                try
                    SMF_cons = Results_Constrained_g1.driver_data.MR2017.SMF(1:2039);
                catch
                    SMF_cons = zeros(2039,1);
                end
            end
            
            plotting (atrial_model, model, metric, xSMF, SMF_interp, SMF_tikh, SMF_cons, minSMF, maxSMF, front_rear, [], video, time_recording);
            
            saveas(gcf,Driver_filepath,'png');
            saveas(gcf,Driver_filepath,'fig');
            saveas(gcf,Driver_filepath,'epsc');
        else
            % Xdriver
            if strcmp(metric,'Driver_Classical')
                try
                    drivers = Ground_Truth.xdriver.Classical;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    xdriver = drivers2;
                    clear drivers2;
                catch
                    xdriver = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Interpolation.driver_data.Classical.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_interp = drivers2;
                    clear drivers2;
                catch
                    driver_interp = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Tikhonov_g0.driver_data.Classical.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_tikh = drivers2;
                    clear drivers2;
                catch
                    driver_tikh = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Constrained_g1.driver_data.Classical.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_cons = drivers2;
                    clear drivers2;
                catch
                    driver_cons = zeros(2039,time_recording);
                end
                
            elseif strcmp(metric,'Driver_BS_Corrected')
                try
                    drivers = Ground_Truth.xdriver.BS_Corrected;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    xdriver = drivers2;
                    clear drivers2;
                catch
                    xdriver = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Interpolation.driver_data.BS_Corrected.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_interp = drivers2;
                    clear drivers2;
                catch
                    driver_interp = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Tikhonov_g0.driver_data.BS_Corrected.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_tikh = drivers2;
                    clear drivers2;
                catch
                    driver_tikh = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Constrained_g1.driver_data.BS_Corrected.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_cons = drivers2;
                    clear drivers2;
                catch
                    driver_cons = zeros(2039,time_recording);
                end
                
            elseif strcmp(metric,'Driver_BS')
                try
                    drivers = Ground_Truth.xdriver.BS;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    xdriver = drivers2;
                    clear drivers2;
                catch
                    xdriver = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Interpolation.driver_data.BS.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_interp = drivers2;
                    clear drivers2;
                catch
                    driver_interp = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Tikhonov_g0.driver_data.BS.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_tikh = drivers2;
                    clear drivers2;
                catch
                    driver_tikh = zeros(2039,time_recording);
                end
                try
                    drivers = Results_Constrained_g1.driver_data.BS.driver;
                    drivers2 = zeros(2039,length(drivers(:,1)));
                    for i = 1:length(drivers2(1,:))
                        if sum(drivers(i,:)) ~= 0
                            drivers2(drivers(i,drivers(i,:)~=0),i) = 1;
                        end
                    end
                    driver_cons = drivers2;
                    clear drivers2;
                catch
                    driver_cons = zeros(2039,time_recording);
                end
            end
            
            frame_array = plotting (atrial_model, model, metric, xdriver, driver_interp, driver_tikh, driver_cons, minDriver, maxDriver, front_rear, [], video, time_recording);
            
            v = VideoWriter(Driver_filepath,'MPEG-4');
            v.Quality=50;
            open(v),
            writeVideo(v,frame_array),
            close(v)
        end
    end
end
%% Time metrics (RDMS, CC, DTW)
if strcmp(metric,'Time_metrics')
    
    RDMS_interp = Results_Interpolation.estimation_metrics.RDMSt(1:2039)';
    RDMS_tikh = Results_Tikhonov_g0.estimation_metrics.RDMSt(1:2039)';
    RDMS_cons = Results_Constrained_g1.estimation_metrics.RDMSt(1:2039)';
    
    CC_interp = Results_Interpolation.estimation_metrics.CCt(1:2039)';
    CC_tikh = Results_Tikhonov_g0.estimation_metrics.CCt(1:2039)';
    CC_cons = Results_Constrained_g1.estimation_metrics.CCt(1:2039)';
    
%     DTW_interp = Results_Interpolation.estimation_metrics.DTW(1:2039)';
%     DTW_tikh = Results_Tikhonov_g0.estimation_metrics.DTW(1:2039)';
%     DTW_cons = Results_Constrained_g1.estimation_metrics.DTW(1:2039)';
    
    mkdir([visualization_path '/Time_metrics/']);
    
    % RDMS
    RDMS_filepath = [visualization_path '/Time_metrics/RDMS_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name];
    temp_metric = 'RDMS';
    plotting (atrial_model, model, temp_metric, [], RDMS_interp, RDMS_tikh, RDMS_cons, minRDMS, maxRDMS, front_rear, [], video, time_recording);
    
    saveas(gcf,RDMS_filepath,'png');
    saveas(gcf,RDMS_filepath,'fig');
    saveas(gcf,RDMS_filepath,'epsc');
    
    % CC
    temp_metric = 'CC';
    mkdir([visualization_path '/Time_metrics/']);
    CC_filepath = [visualization_path '/Time_metrics/CC_' nodes_name...
        '_SNR' num2str(SNR) '_' SNR_cons_name];
    
    plotting (atrial_model, model, temp_metric, [], CC_interp, CC_tikh, CC_cons, minCC, maxCC, front_rear, [], video, time_recording);
    
    saveas(gcf,CC_filepath,'png');
    saveas(gcf,CC_filepath,'fig');
    saveas(gcf,CC_filepath,'epsc');
    
    % DTW
%     temp_metric = 'DTW';
%     mkdir([visualization_path '/Time_metrics/']);
%     DTW_filepath = [visualization_path '/Time_metrics/DTW_' nodes_name...
%         '_SNR' num2str(SNR) '_' SNR_cons_name];
%     
%     plotting (atrial_model, model, temp_metric, [], DTW_interp, DTW_tikh, DTW_cons, minDTW, maxDTW, front_rear, [], video, time_recording);
%     
%     saveas(gcf,DTW_filepath,'png');
%     saveas(gcf,DTW_filepath,'fig');
%     saveas(gcf,DTW_filepath,'epsc');
end

%% Phase RDMS & Phase CC
if contains(metric,'Phase_metrics')
    if strcmp(metric,'Phase_metrics_BS_Corrected')
        RDMS_phase_interp = Results_Interpolation.phase_metrics.BS_Corrected.RDMSt_phase(1:2039);
        RDMS_phase_tikh = Results_Tikhonov_g0.phase_metrics.BS_Corrected.RDMSt_phase(1:2039);
        RDMS_phase_cons = Results_Constrained_g1.phase_metrics.BS_Corrected.RDMSt_phase(1:2039);
        
        CC_phase_interp = Results_Interpolation.phase_metrics.BS_Corrected.CCt_phase(1:2039);
        CC_phase_tikh = Results_Tikhonov_g0.phase_metrics.BS_Corrected.CCt_phase(1:2039);
        CC_phase_cons = Results_Constrained_g1.phase_metrics.BS_Corrected.CCt_phase(1:2039);
        
        mkdir([visualization_path '/Phase_metrics/']);
        RDMS_filepath = [visualization_path '/Phase_metrics/RDMS_phase_' nodes_name...
            '_SNR' num2str(SNR) '_' SNR_cons_name '_BS_Corrected'];
        CC_filepath = [visualization_path '/Phase_metrics/CC_phase_' nodes_name...
            '_SNR' num2str(SNR) '_' SNR_cons_name '_BS_Corrected'];
    elseif strcmp(metric,'Phase_metrics_BS')
        RDMS_phase_interp = Results_Interpolation.phase_metrics.BS.RDMSt_phase(1:2039);
        RDMS_phase_tikh = Results_Tikhonov_g0.phase_metrics.BS.RDMSt_phase(1:2039);
        RDMS_phase_cons = Results_Constrained_g1.phase_metrics.BS.RDMSt_phase(1:2039);
        
        CC_phase_interp = Results_Interpolation.phase_metrics.BS.CCt_phase(1:2039);
        CC_phase_tikh = Results_Tikhonov_g0.phase_metrics.BS.CCt_phase(1:2039);
        CC_phase_cons = Results_Constrained_g1.phase_metrics.BS.CCt_phase(1:2039);
        
        mkdir([visualization_path '/Phase_metrics/']);
        RDMS_filepath = [visualization_path '/Phase_metrics/RDMS_phase_' nodes_name...
            '_SNR' num2str(SNR) '_' SNR_cons_name '_BS'];
        CC_filepath = [visualization_path '/Phase_metrics/CC_phase_' nodes_name...
            '_SNR' num2str(SNR) '_' SNR_cons_name '_BS'];
        
    elseif contains(metric,'Classical')
        RDMS_phase_interp = Results_Interpolation.phase_metrics.Classical.RDMSt_phase(1:2039);
        RDMS_phase_tikh = Results_Tikhonov_g0.phase_metrics.Classical.RDMSt_phase(1:2039);
        RDMS_phase_cons = Results_Constrained_g1.phase_metrics.Classical.RDMSt_phase(1:2039);
        
        CC_phase_interp = Results_Interpolation.phase_metrics.Classical.CCt_phase(1:2039);
        CC_phase_tikh = Results_Tikhonov_g0.phase_metrics.Classical.CCt_phase(1:2039);
        CC_phase_cons = Results_Constrained_g1.phase_metrics.Classical.CCt_phase(1:2039);
        
        mkdir([visualization_path '/Phase_metrics/']);
        RDMS_filepath = [visualization_path '/Phase_metrics/RDMS_phase_' nodes_name...
            '_SNR' num2str(SNR) '_' SNR_cons_name '_Classical'];
        CC_filepath = [visualization_path '/Phase_metrics/CC_phase_' nodes_name...
            '_SNR' num2str(SNR) '_' SNR_cons_name '_Classical'];
        
    elseif contains(metric,'noHDF')
        RDMS_phase_interp = Results_Interpolation.phase_metrics.noHDF.RDMSt_phase(1:2039);
        RDMS_phase_tikh = Results_Tikhonov_g0.phase_metrics.noHDF.RDMSt_phase(1:2039);
        RDMS_phase_cons = Results_Constrained_g1.phase_metrics.noHDF.RDMSt_phase(1:2039);
        
        CC_phase_interp = Results_Interpolation.phase_metrics.noHDF.CCt_phase(1:2039);
        CC_phase_tikh = Results_Tikhonov_g0.phase_metrics.noHDF.CCt_phase(1:2039);
        CC_phase_cons = Results_Constrained_g1.phase_metrics.noHDF.CCt_phase(1:2039);
        
        mkdir([visualization_path '/Phase_metrics/']);
        RDMS_filepath = [visualization_path '/Phase_metrics/RDMS_phase_' nodes_name...
            '_SNR' num2str(SNR) '_' SNR_cons_name '_noHDF'];
        CC_filepath = [visualization_path '/Phase_metrics/CC_phase_' nodes_name...
            '_SNR' num2str(SNR) '_' SNR_cons_name '_noHDF'];
    end
    
    
    % RDMS
    temp_metric = 'Phase_RDMS';
    plotting (atrial_model, model, temp_metric, [], RDMS_phase_interp, RDMS_phase_tikh, RDMS_phase_cons, minRDMS, maxRDMS, front_rear, [], video, time_recording);
    
    saveas(gcf,RDMS_filepath,'png');
    saveas(gcf,RDMS_filepath,'fig');
    saveas(gcf,RDMS_filepath,'epsc');
    
    % CC
    temp_metric = 'Phase_CC';
    plotting (atrial_model, model, temp_metric, [], CC_phase_interp, CC_phase_tikh, CC_phase_cons, minCC, maxCC, front_rear, [], video, time_recording);
    
    saveas(gcf,CC_filepath,'png');
    saveas(gcf,CC_filepath,'fig');
    saveas(gcf,CC_filepath,'epsc');
end

end


