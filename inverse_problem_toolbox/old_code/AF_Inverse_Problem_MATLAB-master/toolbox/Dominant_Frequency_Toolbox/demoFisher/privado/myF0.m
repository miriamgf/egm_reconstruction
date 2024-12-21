function [f01,f02] = myf0(egm)

%

% Estimate f0, only with the maximum and with zero padding
flow = 2.5; fup = 15.5;

%egm = egm(1:1024);

fs = 1000;
[P,f] = psd(egm-mean(egm),length(egm),fs);
[forget,indlow] = min(abs(f-flow));
[forget,indup] = min(abs(f-fup));
[forget,ind1] = max(P(indlow:indup));
f01 = f(ind1+indlow-1);

[P,f] = psd(egm-mean(egm),16*length(egm),fs);
[forget,indlow] = min(abs(f-flow));
[forget,indup] = min(abs(f-fup));
[forget,ind2] = max(P(indlow:indup));
f02 = f(ind2+indlow-1);

