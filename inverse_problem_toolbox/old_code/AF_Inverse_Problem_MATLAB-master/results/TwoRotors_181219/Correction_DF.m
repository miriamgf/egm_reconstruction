egms = Results_Constrained_g1.xhat;
DF_corrected = Results_Constrained_g1.characteristics.DF.BS_Corrected;

index_underestimated = find(DF_corrected<5.5);
egms = egms(index_underestimated,:);

DF_BS = Results_Constrained_g1.characteristics.DF.BS(index_underestimated);
DF_Classical = Results_Constrained_g1.characteristics.DF.Classical(index_underestimated);

[P_z,f] = pwelch(egms'-mean(egms'),[],[],2048,500);

[~,I]= max(P_z);
DF_peak = f(I);

N = length(DF_peak);

%cont = 1;
for i=1:length(DF_peak)
    if ((DF_peak(i)>8)) && DF_peak(i)-1.5 > DF_BS(i)
        %print(c)
        %c = c+1;
        DF_Corrected_underestimated(i) = DF_BS(i);
    else
        DF_Corrected_underestimated(i) = DF_peak(i);
    end
end

Results_Constrained_g1.characteristics.DF.BS_Corrected(index_underestimated) = DF_Corrected_underestimated;