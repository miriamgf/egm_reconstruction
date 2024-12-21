plusminus = sprintf('%s','±');
for i=2:size(table_metrics,1)
    for j=2:size(table_metrics,2)
        table_metrics{i,j}=strrep(table_metrics{i,j},'?',plusminus);
    end
end

for i=2:size(table_characteristics,1)
    for j=2:size(table_characteristics,2)
        table_characteristics{i,j}=strrep(table_characteristics{i,j},'?',plusminus);
    end
end

results_path = [pwd '/'];
path_pattern = [model '_Tikhonov_g_SNR' num2str(SNR_BSP) '_SNR_cons'];
save_path = [path_pattern sprintf('%s_%s',SNR_constrained,'2basket')];
results_file = [results_path save_path];

save(results_file,'-v7.3','fs','Ground_Truth','Results_Interpolation','Results_Tikhonov_g0',...
    'Results_Constrained_g1','Results_Constrained_g2','model','table_metrics',...
    'table_characteristics','table_pvalues_metrics','table_pvalues_characteristics',...
    'nodes_constrained','SNR_BSP','SNR_constrained');

xlswrite(sprintf('%s_metrics.xlsx',results_file),table_metrics);
xlswrite(sprintf('%s_characteristics.xlsx',results_file),table_characteristics);
xlswrite(sprintf('%s_pvalues_metrics.xlsx',results_file),table_pvalues_metrics);
xlswrite(sprintf('%s_pvalues_characteristics.xlsx',results_file),table_pvalues_characteristics);