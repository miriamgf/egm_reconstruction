function [egm,tt] = pseudoegm(t,f0,d,a)

% 

if nargin==2,
    d = 0;
    a = 1;
end
if nargin == 3
    a = 1;
end


fs = 977;%1000/2; %500 
%fs = 1000;
cycle_length = 1/f0;
%t = 4.096;  % length in secs

delta_in_samples = round(d/1000*fs); % Uncertainty is in msec

single_egm = pseudobipolar;

egm = zeros(round(t*fs),1);
tt = 1/fs*(0:length(egm)-1);
ind_deltas = round((1+delta_in_samples:fs*cycle_length:length(egm)));

% Cycle uncertainty
uncertainty = ceil(delta_in_samples*( 2*(rand(1,length(ind_deltas))-.5) ));
ind_deltas = ind_deltas + uncertainty;
if ind_deltas(1)<1,
    ind_deltas = ind_deltas(2:end);
end
if ind_deltas(end)>length(egm)
    ind_deltas = ind_deltas(1:end-1);
end

% Amplitude uncertainty
for i=1:length(ind_deltas)
    mya= (1-a)*randn+a;
    egm(ind_deltas(i)) = mya;
end

%egm(ind_deltas) = 1;
egm = filter(single_egm,1,egm);


