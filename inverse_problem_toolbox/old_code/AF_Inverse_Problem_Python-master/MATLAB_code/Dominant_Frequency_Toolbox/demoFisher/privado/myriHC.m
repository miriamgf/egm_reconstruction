function ri = myriHC(egm,f0,P,f)

% Proposal: harmonic correction

% Estimate RI, assume we know f0

fs = 1000;
%[P,f] = psd(egm-mean(egm),length(egm),fs); TO_DO: THIS IS FROM THE OLD
%VERSION, VERIFY THAT THE RESULTS ARE THE SAMEN THAN THOSE USING THE ACTUAL
%P AND F FROM THE INPUT PARAMETERS

% Number of harmonics in the band 3-15;
n_harmonics = ceil(15/f0);

% Some common calculations for all the harmonics
delta_f = f(2);
% spectral indices for band in 5-30 Hz
[forget,indlow] = min(abs(f - 3));
[forget,indup] = min(abs(f - 15));
% Band around f_0 detected
band_peak = 0.75;   % Hz
delta_f0 = round(band_peak/delta_f/2);
delta_search = ceil(f0/delta_f/4);

P_f0 = 0;

for k=1:n_harmonics

    % find peak around k*f0
    ind_f0 = round(k*f0/delta_f);
    [forget,ind_f0_detected] = max(P(ind_f0-delta_search:ind_f0+delta_search));
    ind_f0_detected = ind_f0 - delta_search - 1 + ind_f0_detected;
    if ind_f0_detected-delta_f0 < indlow
        indlow = ind_f0_detected-delta_f0;
    end
    if ind_f0_detected + delta_f0 > indup
        indup = ind_f0_detected + delta_f0;
    end
    P_f0 = P_f0 + sum(P(ind_f0_detected - delta_f0:ind_f0_detected + delta_f0));
end

Ptot = sum(P(indlow:indup));


% RI
ri = P_f0/Ptot;

