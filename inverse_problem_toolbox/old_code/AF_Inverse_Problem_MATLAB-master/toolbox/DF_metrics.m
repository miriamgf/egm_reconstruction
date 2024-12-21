function [DF, DF_metrics,z_egm]=DF_metrics(x_hat, fs, model,xDF,selected_nodes)
if strcmp(model,'Classical')
    DF = dominant_frequency (x_hat, fs);
    z_egm=[];
elseif strcmp(model,'BS')
    [DF,z_egm] = df_BS(x_hat,fs);
elseif strcmp(model,'Kuklik')
    DF = df_kuklik(x_hat,fs);
elseif strcmp(model,'MR2017')
    DF = dominant_frequency_MR2017 (x_hat, fs);
end

if isempty(selected_nodes)
    %RAE_metrics.RAE = abs(((xDF(1:2039)-DF(1:2039))./xDF(1:2039))*100);
    DF_metrics.mDF=mean(DF);
    DF_metrics.stdDF=std(DF);
    DF_metrics.RAE = abs((xDF(1:2039)-DF(1:2039)));
    DF_metrics.mRAE=mean(DF_metrics.RAE);
    DF_metrics.std_mRAE=std(DF_metrics.RAE);
else
    %RAE_metrics.RAE = abs(((xDF(1:2039)-DF(1:2039))./xDF(1:2039))*100);
    DF_metrics.mDF=mean(DF(selected_nodes));
    DF_metrics.stdDF=std(DF(selected_nodes));
    DF_metrics.RAE = abs((xDF(1:2039)-DF(1:2039)));
    DF_metrics.mRAE=mean(DF_metrics.RAE(selected_nodes));
    DF_metrics.std_mRAE=std(DF_metrics.RAE(selected_nodes));
end
end
