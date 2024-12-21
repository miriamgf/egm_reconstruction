function [metrics_DF] = DF_metrics_BS_datapython(DF_BS,xDF_BS)

metrics_DF.mDF=mean(DF_BS);
metrics_DF.stdDF=std(DF_BS);
metrics_DF.RAE = abs((xDF_BS(1:2039)-DF_BS(1:2039)));
metrics_DF.mRAE=mean(metrics_DF.RAE);
metrics_DF.std_mRAE=std(metrics_DF.RAE);