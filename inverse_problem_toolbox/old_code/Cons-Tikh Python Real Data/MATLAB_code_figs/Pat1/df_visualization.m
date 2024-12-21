%% DF Maps representation
%% Load atrial model
load('atrial_model.mat')

%% Load mesh indexes
load('EGM_mesh_index.mat')

%% Load DF data
df_tikh0 = csvread('df_tikh0.txt');
df_cons1 = csvread('df_cons1.txt');

%% Plot atrial maps
figure('pos',[516 424 642 544]),
[ha, pos] = tight_subplot(2,2,0.000001);
cmap = buildcmap('rygcbm');
colormap(cmap);
title_model = ['Pat 110114 (DF)'];
annotation(gcf,'textbox',...
    [pos{1,1}(1)+0.35 pos{1,1}(2)+0.40 0.1 0.1],...
    'FitBoxToText','on','String',title_model,'FontSize',16,'EdgeColor','none','FontWeight','bold', 'Interpreter', 'none');
     

% Tikh0
axes(ha(1)),
atrialrepresentation (Atria, df_tikh0, 4, 8);
view(55,0),
annotation(gcf,'textbox',...
    [pos{1,1}(1)-0.05 pos{1,1}(2)+0.2 0.1 0.1],...
    'FitBoxToText','on','String','Tikh0','FontSize',16,'EdgeColor','none','FontWeight','bold');
for i=1:length(EGM_mesh_index)
    [~] = ClickA3DPoint(Atria.node,gcf,EGM_mesh_index(i),1);
end
axes(ha(2)),
atrialrepresentation (Atria, df_tikh0, 4, 8);
view(210,0),camlight
for i=1:length(EGM_mesh_index)
    [~] = ClickA3DPoint(Atria.node,gcf,EGM_mesh_index(i),1);
end

% Cons1
axes(ha(3)),
atrialrepresentation (Atria, df_cons1, 4, 8);
view(55,0),
annotation(gcf,'textbox',...
    [pos{3,1}(1)-0.05 pos{3,1}(2)+0.2 0.1 0.1],...
    'FitBoxToText','on','String','Cons1','FontSize',16,'EdgeColor','none','FontWeight','bold');
for i=1:length(EGM_mesh_index)
    [~] = ClickA3DPoint(Atria.node,gcf,EGM_mesh_index(i),1);
end
axes(ha(4)),
atrialrepresentation (Atria, df_cons1, 4, 8);
view(210,0),camlight
for i=1:length(EGM_mesh_index)
    [~] = ClickA3DPoint(Atria.node,gcf,EGM_mesh_index(i),1);
end

c=colorbar(gca,'Position',...
    [0.891870860927157 0.317083333333334 0.0177777777777777 0.423332833333334],...
    'FontWeight','bold');
c.Label.String = 'Frequency (Hz)';