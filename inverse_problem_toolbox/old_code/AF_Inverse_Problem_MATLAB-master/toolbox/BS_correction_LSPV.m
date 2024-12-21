function [DF_corrected,metrics_DF_corrected] = BS_correction_LSPV(egms,DF_Classical, DF_BS,xDF_BS_Corrected)

[P_z,f] = pwelch(egms',[],[],2048,500);

[~,I]= max(P_z);
DF_peak = f(I);

N = length(DF_peak);
DF_corrected = zeros(N,1);
%cont = 1;
for i=1:length(DF_peak)
    if ((DF_peak(i)>8))
        %print(c)
        %c = c+1;
        DF_corrected(i) = DF_BS(i);
    else
        DF_corrected(i) = DF_Classical(i);
    end
end

if nargin>3
    metrics_DF_corrected.mDF=mean(DF_corrected);
    metrics_DF_corrected.stdDF=std(DF_corrected);
    metrics_DF_corrected.RAE = abs((xDF_BS_Corrected(1:2039)-DF_corrected(1:2039)));
    metrics_DF_corrected.mRAE=mean(metrics_DF_corrected.RAE);
    metrics_DF_corrected.std_mRAE=std(metrics_DF_corrected.RAE);
else
    metrics_DF_corrected = [];
end
end