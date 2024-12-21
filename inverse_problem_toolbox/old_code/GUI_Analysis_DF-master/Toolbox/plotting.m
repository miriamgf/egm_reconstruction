function my_frames = plotting (Model, MAG, EST, ming, maxg, flag, video, instant_time, SNR, SNR_cons, CC_RDMS)
% It displays a video or a picture.

% Input variables:
% Model: structure´s model.
% MAG: data of the structure´s model.
% EST: data of the estimated signal.
% ming: minimal value of the scale.
% maxg: maximum value of the scale.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if video
    [~, c] = size(MAG);
    j=1;
    h = waitbar(0,'Rendering video...');
    for i=1:c
        % computations take place here
        f=figure('Visible','off');
        
        if flag == 0
            cmap = buildcmap('bcyr');
            colormap(cmap);
            aux = torsorepresentation (Model, MAG(:,i), ming, maxg);
            
            ax = axes('Units','Normal','Position',[.075 .075 .87 .82],'Visible','off');
            set(get(ax,'Title'),'Visible','on')
            text = (['Time :  t = ' num2str(i)]);
            title(text, 'Fontsize',25);
            
            ax = axes('Units','Normal','Position',[.075 .075 .88 .73],'Visible','off');
            set(get(ax,'Title'),'Visible','on')
            text = (['SNR ' num2str(SNR)]);
            title(text, 'Fontsize',18);
            
            ax = axes('Units','Normal','Position',[.075 .075 .56 .68],'Visible','off');
            set(get(ax,'Title'),'Visible','on')
            text = ('Front');
            title(text, 'Fontsize',14);
            
            ax = axes('Units','Normal','Position',[.075 .075 1.19 .68],'Visible','off');
            set(get(ax,'Title'),'Visible','on')
            text = ('Back');
            title(text, 'Fontsize',14);
        end
        if (flag == 1) || (flag == 3) || (flag == 4)
            if flag == 1
                cmap = buildcmap('bcyr');
                colormap(cmap);
            end
            if flag == 3
                cmap = buildcmap('wyrgbmk');
                colormap(cmap);
            end
            if flag == 4
                cmap = buildcmap('bwr');
                cmap=cmap(30:end,:);
                colormap(cmap);
            end
            
            atrialrepresentation (Model, MAG(:,i), ming, maxg);
            
            ax = axes('Units','Normal','Position',[.075 .05 .81 .85],'Visible','off');
            uistack(ax,'bottom');
            set(get(ax,'Title'),'Visible','on')
            text = ([num2str(SNR) ', ' SNR_cons]);
            title(text, 'Fontsize',18);
            
            ax = axes('Units','Normal','Position',[.075 .04 .815 .80],'Visible','off');
            uistack(ax,'bottom');
            set(get(ax,'Title'),'Visible','on')
            text = (['Time :  t = ' num2str(i)]);
            title(text, 'Fontsize',15);
            
            ax = axes('Units','Normal','Position',[.075 .02 .46 .75],'Visible','off');
            uistack(ax,'bottom');
            set(get(ax,'Title'),'Visible','on')
            text = ('Front');
            title(text, 'Fontsize',13);
            
            ax = axes('Units','Normal','Position',[.075 .02 1.19 .75],'Visible','off');
            uistack(ax,'bottom');
            set(get(ax,'Title'),'Visible','on')
            text = ('Back');
            title(text, 'Fontsize',13);
        end
        pause(0.1)
        
        my_frames(j) = getframe(f);   % frame saved.
        j=j+1;
        close(f)
        waitbar(i / c)
    end
    close(h)
else
    figure(),
    my_frames = [];
    if flag == 0
        cmap = buildcmap('bcyr');
        colormap(cmap);
        aux = torsorepresentation (Model, MAG, ming, maxg);
        
        ax = axes('Units','Normal','Position',[.075 .075 .87 .85],'Visible','off');
        set(get(ax,'Title'),'Visible','on')
        text = (['Time :  t = ' num2str(instant_time)]);
        title(text, 'Fontsize',25);

        ax = axes('Units','Normal','Position',[.075 .075 .88 .80],'Visible','off');
        set(get(ax,'Title'),'Visible','on')
        text = (['SNR ' num2str(SNR) ', SNR_cons ' SNR_cons]);
        title(text, 'Fontsize',22);

        ax = axes('Units','Normal','Position',[.075 .075 .56 .79],'Visible','off');
        set(get(ax,'Title'),'Visible','on')
        text = ('Front');
        title(text, 'Fontsize',18);

        ax = axes('Units','Normal','Position',[.075 .075 1.19 .79],'Visible','off');
        set(get(ax,'Title'),'Visible','on')
        text = ('Back');
        title(text, 'Fontsize',18);
    end
    if (flag == 1) || (flag == 3)
        if flag == 1
            cmap = buildcmap('bcyr');
            colormap(cmap);
        end
        if flag == 3
            cmap = buildcmap('wyrgbmk');
            colormap(cmap);
        end
        
        atrialrepresentation (Model, MAG, ming, maxg);
        
        ax = axes('Units','Normal','Position',[.075 .05 .81 .85],'Visible','off');
        uistack(ax,'bottom');
        set(get(ax,'Title'),'Visible','on')
        text = ([num2str(SNR) ', ' SNR_cons]);
        title(text, 'Fontsize',18);
        
        ax = axes('Units','Normal','Position',[.075 .04 .815 .80],'Visible','off');
        uistack(ax,'bottom');
        set(get(ax,'Title'),'Visible','on')
        text = (['Time :  t = ' num2str(instant_time)]);
        title(text, 'Fontsize',15);
        
        ax = axes('Units','Normal','Position',[.075 .02 .46 .75],'Visible','off');
        uistack(ax,'bottom');
        set(get(ax,'Title'),'Visible','on')
        text = ('Front');
        title(text, 'Fontsize',13);
        
        ax = axes('Units','Normal','Position',[.075 .02 1.19 .75],'Visible','off');
        uistack(ax,'bottom');
        set(get(ax,'Title'),'Visible','on')
        text = ('Back');
        title(text, 'Fontsize',13);
        
    end
    if (flag == 2) || (flag == 4) || (flag == 5) || (flag == 6) || (flag == 7)
        if flag == 2
            colormap parula;
        end
        if flag == 4
            cmap = buildcmap('bwr');
            cmap=cmap(30:end,:);
            colormap(cmap);
        end
        if flag == 5
            if ~CC_RDMS
                cmap = buildcmap('kcr');
                colormap(cmap);
            else
                cmap = buildcmap('kyr');
                colormap(cmap);
            end 
        end
        if flag == 6
            cmap = buildcmap('bcyw');
            colormap(cmap);
        end
        if flag == 7
            if ~CC_RDMS
                cmap = buildcmap('kygbw');
                colormap(cmap);
            else
                cmap = buildcmap('wygcm');
                colormap(cmap);
            end
        end

        atrialrepresentation (Model, MAG, ming, maxg);
        
        ax = axes('Units','Normal','Position',[.075 .05 .81 .85],'Visible','off');
        uistack(ax,'bottom');
        set(get(ax,'Title'),'Visible','on')
        text = ([num2str(SNR) ', ' SNR_cons]);
        title(text, 'Fontsize',18);
        
        ax = axes('Units','Normal','Position',[.075 .04 .815 .80],'Visible','off');
        uistack(ax,'bottom');
        set(get(ax,'Title'),'Visible','on')
        text = (['Time :  t = ' num2str(instant_time)]);
        title(text, 'Fontsize',15);
        
        ax = axes('Units','Normal','Position',[.075 .02 .46 .75],'Visible','off');
        uistack(ax,'bottom');
        set(get(ax,'Title'),'Visible','on')
        text = ('Front');
        title(text, 'Fontsize',13);
        
        ax = axes('Units','Normal','Position',[.075 .02 1.19 .75],'Visible','off');
        uistack(ax,'bottom');
        set(get(ax,'Title'),'Visible','on')
        text = ('Back');
        title(text, 'Fontsize',13);
    end
end

end