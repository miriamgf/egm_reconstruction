function [DF_corrected,metrics_DF_corrected] = BS_correction(DF_Classical,DF_BS,xDF_BS_Corrected)
N = length(DF_Classical);
DF_corrected = zeros(N,1);
for i=1:length(DF_Classical)
   if ((DF_Classical(i)+2)<DF_BS(i))
       DF_corrected(i) = DF_Classical(i);
   else
       DF_corrected(i) = DF_BS(i);
   end
end

if nargin>2
    metrics_DF_corrected.mDF=mean(DF_corrected);
    metrics_DF_corrected.stdDF=std(DF_corrected);
    metrics_DF_corrected.RAE = abs((xDF_BS_Corrected(1:2039)-DF_corrected(1:2039)));
    metrics_DF_corrected.mRAE=mean(metrics_DF_corrected.RAE);
    metrics_DF_corrected.std_mRAE=std(metrics_DF_corrected.RAE);
else
    metrics_DF_corrected = [];
end
end