function frame_array = plotting (atrial_model, model, metric, ground_truth, interp_var, tikh, cons, minScale, maxScale, front_rear, instant_time, video, time_recording)
% It displays a video or a picture.
%
% Input variables:
% atrial_moodel: structure´s model.
% model: data of the estimated signal.
% metric: parameter to display.
% ground_truth: reference data.
% interp_var: reconstruction by interpolation.
% tikh: reconstruction using Tikh-g0.
% cons: reconstruction using Cons-g1.
% minScale: minimal value of the scale.
% maxScale: maximum value of the scale.
% front_rear: orientation of the atrial model.
% instant_time: time instant to show.
% video: option to show a picture or a video.
% time_recording: video duration.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if video
    for i=1:time_recording
        figure('pos',[516 168 1230 400]),
        [ha, pos] = tight_subplot(1,4,0.000001);
        
        if strcmp(metric,'EpPot')
            cmap = buildcmap('bcyr');
            colormap(cmap);
            title_model = [model ' (Ep. Potentials t=' num2str(i) ')'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.04 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.14 0.09 0.1 0.1],...
                'FitBoxToText','on','String','Amplitude (normalized)','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif strcmp(metric,'Phase_Classical') || strcmp(metric,'Phase_BS') || ...
                strcmp(metric,'Phase_BS_Corrected') || strcmp(metric,'Phase_noHDF')
            cmap = buildcmap('wyrgbmk');
            colormap(cmap);
            title_model = [model ' (Phase t=' num2str(i) ')'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.07 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.19 0.09 0.1 0.1],...
                'FitBoxToText','on','String','Phase','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif contains(metric,'Driver')
            cmap = buildcmap('bwr');
            colormap(cmap);
            title_model = [model ' (Driver position t=' num2str(i) ')'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.04 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.15 0.09 0.1 0.1],...
                'FitBoxToText','on','String','Rotor occurrences (%)','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        end
        
        % Ground Truth
        axes(ha(1)),atrialrepresentation (atrial_model, ground_truth(:,i), minScale, maxScale);
        if strcmp(front_rear,'rear')
            view(-90,30),camlight, camlight, camlight;
        elseif strcmp(front_rear,'down')
            view(-120,-35),camlight, camlight, camlight,camlight, camlight;
        elseif strcmp(front_rear,'left')
            view(-123,12),camlight, camlight, camlight,camlight, camlight;
        end
        annotation(gcf,'textbox',...
            [pos{1,1}(1)+0.04 pos{1,1}(2)+0.72 0.1 0.1],...
            'FitBoxToText','on','String','Ground Truth','FontSize',16,'EdgeColor','none','FontWeight','bold');
        
        % Interpolation
        axes(ha(2)),atrialrepresentation (atrial_model, interp_var(:,i), minScale, maxScale);
        if strcmp(front_rear,'rear')
            view(-90,30),camlight, camlight, camlight;
        elseif strcmp(front_rear,'down')
            view(-120,-35),camlight, camlight, camlight,camlight, camlight;
        elseif strcmp(front_rear,'left')
            view(-123,12),camlight, camlight, camlight,camlight, camlight;
        end
        annotation(gcf,'textbox',...
            [pos{2,1}(1)+0.043 pos{1,1}(2)+0.72 0.1 0.1],...
            'FitBoxToText','on','String','Interpolation','FontSize',16,'EdgeColor','none','FontWeight','bold');
        
        % Tikh-g0
        axes(ha(3)),atrialrepresentation (atrial_model, tikh(:,i), minScale, maxScale);
        if strcmp(front_rear,'rear')
            view(-90,30),camlight, camlight, camlight;
        elseif strcmp(front_rear,'down')
            view(-120,-35),camlight, camlight, camlight,camlight, camlight;
        elseif strcmp(front_rear,'left')
            view(-123,12),camlight, camlight, camlight,camlight, camlight;
        end
        annotation(gcf,'textbox',...
            [pos{3,1}(1)+0.085 pos{1,1}(2)+0.72 0.1 0.1],...
            'FitBoxToText','on','String','Tikh','FontSize',16,'EdgeColor','none','FontWeight','bold');
        
        % Cons-g1
        axes(ha(4)),atrialrepresentation (atrial_model, cons(:,i), minScale, maxScale);
        if strcmp(front_rear,'rear')
            view(-90,30),camlight, camlight, camlight;
        elseif strcmp(front_rear,'down')
            view(-120,-35),camlight, camlight, camlight,camlight, camlight;
        elseif strcmp(front_rear,'left')
            view(-123,12),camlight, camlight, camlight,camlight, camlight;
        end
        annotation(gcf,'textbox',...
            [pos{4,1}(1)+0.07 pos{1,1}(2)+0.72 0.1 0.1],...
            'FitBoxToText','on','String','Cons-Tikh','FontSize',16,'EdgeColor','none','FontWeight','bold');
        
        % Colorbar
        c=colorbar('South');
        c.Position=[0.354558041666666 0.0843410947464589 0.276336267276423 0.0264922385868744];
        c.FontSize=12;
        c.AxisLocation='out';
        
        pause(0.001)
        
        frame_array(i) = getframe(gcf);   % frame saved.
        close(gcf)
    end
else
    if contains(metric,'RDMS') || contains(metric,'CC') || contains(metric,'DTW')
        figure('pos',[516 168 930 400]),
        [ha, pos] = tight_subplot(1,3,0.000001);
        
        if strcmp(metric,'RDMS')
            cmap = buildcmap('bcy');
            colormap(cmap);
            title_model = [model ' (RDMS)'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1) pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.11 0.09 0.1 0.1],...
                'FitBoxToText','on','String','RDMS','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif strcmp(metric,'CC')
            cmap = buildcmap('kry');
            colormap(cmap);
            title_model = [model ' (CC)'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1) pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.125 0.09 0.1 0.1],...
                'FitBoxToText','on','String','CC','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif strcmp(metric,'Phase_RDMS')
            cmap = buildcmap('mbcy');
            colormap(cmap);
            title_model = [model ' (Phase RDMS)'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)-0.04 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.08 0.09 0.1 0.1],...
                'FitBoxToText','on','String','Phase RDMS','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif strcmp(metric,'Phase_CC')
            cmap = buildcmap('krmy');
            colormap(cmap);
            title_model = [model ' (Phase CC)'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)-0.03 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.10 0.09 0.1 0.1],...
                'FitBoxToText','on','String','Phase CC','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif strcmp(metric,'DTW')
            cmap = buildcmap('bcyk');
            colormap(cmap);
            title_model = [model ' (DTW)'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1) pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.12 0.09 0.1 0.1],...
                'FitBoxToText','on','String','DTW','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        end
        
        % Interpolation
        axes(ha(1)),atrialrepresentation (atrial_model, interp_var, minScale, maxScale);
        if strcmp(front_rear,'rear')
            view(-90,30),camlight, camlight, camlight;
        elseif strcmp(front_rear,'down')
            view(-120,-35),camlight, camlight, camlight,camlight, camlight;
        elseif strcmp(front_rear,'left')
            view(-123,12),camlight, camlight, camlight,camlight, camlight;
        end
        annotation(gcf,'textbox',...
            [pos{1,1}(1)+0.04 pos{1,1}(2)+0.72 0.1 0.1],...
            'FitBoxToText','on','String','Interpolation','FontSize',16,'EdgeColor','none','FontWeight','bold');
        
        % Tikh-g0
        axes(ha(2)),atrialrepresentation (atrial_model, tikh, minScale, maxScale);
        if strcmp(front_rear,'rear')
            view(-90,30),camlight, camlight, camlight;
        elseif strcmp(front_rear,'down')
            view(-120,-35),camlight, camlight, camlight,camlight, camlight;
        elseif strcmp(front_rear,'left')
            view(-123,12),camlight, camlight, camlight,camlight, camlight;
        end
        annotation(gcf,'textbox',...
            [pos{2,1}(1)+0.11 pos{1,1}(2)+0.72 0.1 0.1],...
            'FitBoxToText','on','String','Tikh','FontSize',16,'EdgeColor','none','FontWeight','bold');
        
        % Cons-g1
        axes(ha(3)),atrialrepresentation (atrial_model, cons, minScale, maxScale);
        if strcmp(front_rear,'rear')
            view(-90,30),camlight, camlight, camlight;
        elseif strcmp(front_rear,'down')
            view(-120,-35),camlight, camlight, camlight,camlight, camlight;
        elseif strcmp(front_rear,'left')
            view(-123,12),camlight, camlight, camlight,camlight, camlight;
        end
        annotation(gcf,'textbox',...
            [pos{3,1}(1)+0.085 pos{1,1}(2)+0.72 0.1 0.1],...
            'FitBoxToText','on','String','Cons-Tikh','FontSize',16,'EdgeColor','none','FontWeight','bold');
        
        c=colorbar('South');
        c.Position=[0.354558041666666 0.0843410947464589 0.276336267276423 0.0264922385868744];
        c.FontSize=12;
        c.AxisLocation='out';
    else
        figure('pos',[516 168 1230 400]),
        [ha, pos] = tight_subplot(1,4,0.000001);
        
        if strcmp(metric,'EpPot')
            cmap = buildcmap('bcyr');
            colormap(cmap);
            title_model = [model ' (Ep. Potentials t=' num2str(instant_time) ')'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.04 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.14 0.09 0.1 0.1],...
                'FitBoxToText','on','String','Amplitude (normalized)','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif contains(metric,'DF') && ~contains(metric,'Driver')&& ~contains(metric,'Phase')
            cmap = buildcmap('wbr');
            colormap(cmap);
            title_model = [model ' (DF)'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.09 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.17 0.09 0.1 0.1],...
                'FitBoxToText','on','String','Frequency (Hz)','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif strcmp(metric,'OI')
            cmap = buildcmap('ryc');
            colormap(cmap);
            title_model = [model ' (OI)'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.09 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.14 0.09 0.1 0.1],...
                'FitBoxToText','on','String','Organization Index (OI)','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif strcmp(metric,'RI')
            cmap = buildcmap('ckyb');
            colormap(cmap);
            title_model = [model ' (RI)'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.09 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.15 0.09 0.1 0.1],...
                'FitBoxToText','on','String','Regularity Index (RI)','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif strcmp(metric,'SampEn')
            cmap = buildcmap('wbk');
            colormap(cmap);
            title_model = [model ' (SampEn)'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.075 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.17 0.09 0.1 0.1],...
                'FitBoxToText','on','String','log(SampEn)','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif contains(metric,'Phase')
            cmap = buildcmap('wyrgbmk');
            colormap(cmap);
            title_model = [model ' (Phase t=' num2str(instant_time) ')'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.07 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.19 0.09 0.1 0.1],...
                'FitBoxToText','on','String','Phase','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        elseif contains(metric,'Driver')
            cmap = buildcmap('bwr');
            colormap(cmap);
            title_model = [model ' (SMF)'];
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.09 pos{1,1}(2)+0.85 0.1 0.1],...
                'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
            annotation(gcf,'textbox',...
                [pos{2,1}(1)+0.15 0.09 0.1 0.1],...
                'FitBoxToText','on','String','Rotor occurrences (%)','FontSize',12,...
                'EdgeColor','none','FontWeight','bold');
        end
        
        axes(ha(1)),atrialrepresentation (atrial_model, ground_truth, minScale, maxScale);
        if strcmp(front_rear,'rear')
            view(-90,30),camlight, camlight, camlight;
        elseif strcmp(front_rear,'down')
            view(-120,-35),camlight, camlight, camlight,camlight, camlight;
        elseif strcmp(front_rear,'left')
            view(-123,12),camlight, camlight, camlight,camlight, camlight;
        end
        annotation(gcf,'textbox',...
            [pos{1,1}(1)+0.04 pos{1,1}(2)+0.72 0.1 0.1],...
            'FitBoxToText','on','String','Ground Truth','FontSize',16,'EdgeColor','none','FontWeight','bold');
        
        % Interpolation
        axes(ha(2)),atrialrepresentation (atrial_model, interp_var, minScale, maxScale);
        if strcmp(front_rear,'rear')
            view(-90,30),camlight, camlight, camlight;
        elseif strcmp(front_rear,'down')
            view(-120,-35),camlight, camlight, camlight,camlight, camlight;
        elseif strcmp(front_rear,'left')
            view(-123,12),camlight, camlight, camlight,camlight, camlight;
        end
        annotation(gcf,'textbox',...
            [pos{2,1}(1)+0.043 pos{1,1}(2)+0.72 0.1 0.1],...
            'FitBoxToText','on','String','Interpolation','FontSize',16,'EdgeColor','none','FontWeight','bold');
        
        % Tikh-g0
        axes(ha(3)),atrialrepresentation (atrial_model, tikh, minScale, maxScale);
        if strcmp(front_rear,'rear')
            view(-90,30),camlight, camlight, camlight;
        elseif strcmp(front_rear,'down')
            view(-120,-35),camlight, camlight, camlight,camlight, camlight;
        elseif strcmp(front_rear,'left')
            view(-123,12),camlight, camlight, camlight,camlight, camlight;
        end
        annotation(gcf,'textbox',...
            [pos{3,1}(1)+0.085 pos{1,1}(2)+0.72 0.1 0.1],...
            'FitBoxToText','on','String','Tikh','FontSize',16,'EdgeColor','none','FontWeight','bold');
        
        % Cons-g1
        axes(ha(4)),atrialrepresentation (atrial_model, cons, minScale, maxScale);
        if strcmp(front_rear,'rear')
            view(-90,30),camlight, camlight, camlight;
        elseif strcmp(front_rear,'down')
            view(-120,-35),camlight, camlight, camlight,camlight, camlight;
        elseif strcmp(front_rear,'left')
            view(-123,12),camlight, camlight, camlight,camlight, camlight;
        end
        annotation(gcf,'textbox',...
            [pos{4,1}(1)+0.07 pos{1,1}(2)+0.72 0.1 0.1],...
            'FitBoxToText','on','String','Cons-Tikh','FontSize',16,'EdgeColor','none','FontWeight','bold');
        
        c=colorbar('South');
        c.Position=[0.354558041666666 0.0843410947464589 0.276336267276423 0.0264922385868744];
        c.FontSize=12;
        c.AxisLocation='out';
    end
end
end