function ri = myri(egm,f0)

%

% Estimate RI, assume we know f0

fs = 1000;
[P,f] = psd(egm-mean(egm),length(egm),fs);

% find peak around f0
delta_f = f(2);
ind_f0 = round(f0/delta_f);
delta_search = ceil(f0/delta_f/4);
[forget,ind_f0_detected] = max(P(ind_f0-delta_search:ind_f0+delta_search));
ind_f0_detected = ind_f0 - delta_search - 1 + ind_f0_detected;

% Band around f_0 detected
band_peak = 0.75;   % Hz
delta_f0 = round(band_peak/delta_f/2);

% spectral indices for band in 5-30 Hz
[forget,indlow] = min(abs(f - 3));
[forget,indup] = min(abs(f - 15));

if ind_f0_detected-delta_f0 < indlow
    indlow = ind_f0_detected-delta_f0;
end
if ind_f0_detected + delta_f0 > indup
    indup = ind_f0_detected + delta_f0;
end

Ptot = sum(P(indlow:indup));
P_f0 = sum(P(ind_f0_detected - delta_f0:ind_f0_detected + delta_f0));


% RI
ri = P_f0/Ptot;

