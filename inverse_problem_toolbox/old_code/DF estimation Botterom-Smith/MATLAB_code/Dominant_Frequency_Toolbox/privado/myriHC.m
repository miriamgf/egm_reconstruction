function [ri,ejexplot,ejeyplot,ejexbase] = myriHC(egm,f0,P,f)
%
%Function that computes RI (or is OI) CHECK THAT
%
%[ri,ejexplot,ejeyplot,ejexbase] = myriHC(egm,f0,fs,P,f)
%
%TO_DO CHECK IF THE COMPUTATION IS OK WITH THE PAPERS

% Proposal: harmonic correction

% Estimate RI, assume we know f0

% Number of harmonics in the band 3-30;
n_harmonics = floor(30/f0);

% Some common calculations for all the harmonics
delta_f = f(2);
% spectral indices for band in 5-30 Hz
[forget,indlow] = min(abs(f - 3));
[forget,indup] = min(abs(f - 30));
% Band around f_0 detected
band_peak = 1;%0.75;   % Hz
delta_f0 = round(band_peak/delta_f/2);
delta_search = ceil(f0/delta_f/3);

P_f0 = 0;

ejexplot = []; ejeyplot = [];

for k=1:n_harmonics

    % find peak around k*f0
    ind_f0 = round(k*f0/delta_f);
    [forget,ind_f0_detected] = max(P(ind_f0-delta_search:ind_f0+delta_search));
    ind_f0_detected = ind_f0 - delta_search - 1 + ind_f0_detected;
    if ind_f0_detected-delta_f0 -round(f0/3)<= indlow
        indlow = ind_f0_detected-round(f0/3); %delta_f0;
    end
    if ind_f0_detected + delta_f0 + round(f0/3)>= indup
        indup = ind_f0_detected + round(f0/3); %delta_f0;
    end
    P_f0 = P_f0 + sum(P(ind_f0_detected - delta_f0:ind_f0_detected + delta_f0));
    ejexplot = [ejexplot,f(ind_f0_detected - delta_f0:ind_f0_detected + delta_f0)',NaN];
    ejeyplot = [ejeyplot,P(ind_f0_detected - delta_f0:ind_f0_detected + delta_f0)',NaN];
end

Ptot = sum(P(indlow:indup));

ejexbase = f(indlow:indup);

% RI
ri = P_f0/Ptot;

