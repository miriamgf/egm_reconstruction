function [DF_corrected,metrics_DF_corrected] = BS_correction_TR(egms,DF_Classical, DF_BS,xDF_BS_Corrected)

% [DF_corrected,~] = BS_correction(DF_Classical,DF_BS);
% 
% index_underestimated = find(DF_corrected<5.5);
% egms = egms(index_underestimated,:);
% 
% DF_BS = DF_BS(index_underestimated);
% DF_Classical = DF_Classical(index_underestimated);
% 
% [P_z,f] = pwelch(egms'-mean(egms'),[],[],2048,500);
% 
% [~,I]= max(P_z);
% DF_peak = f(I);
% 
% N = length(DF_peak);
% 
% %cont = 1;
% for i=1:length(DF_peak)
%     if ((DF_peak(i)>8)) && DF_peak(i)-2 > DF_BS(i)
%         %print(c)
%         %c = c+1;
%         DF_corrected_TR(i) = DF_BS(i);
%     else
%         DF_corrected_TR(i) = DF_Classical(i);
%     end
% end
% 
% DF_corrected(index_underestimated) = DF_corrected_TR;

[P_z,f] = pwelch(egms',[],[],2048,500);

[~,I]= max(P_z);
DF_peak = f(I);

N = length(DF_peak);
DF_corrected = zeros(N,1);
%cont = 1;
for i=1:length(DF_peak)
    if ((DF_peak(i)>8)) || DF_BS(i)+1.5 > DF_Classical(i)
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