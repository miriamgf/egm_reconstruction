function [s_det,s_pp] = detrendSpline(signal,t,w_length,drawflag,mensaje)
%Function that remove base line noise using cubic splines
%
%[s_det,s_pp] = detrendSpline(signal,t,w_length,drawflag,mensaje)
%
%Issue: the free parameter is the length of the window in seconds w_length.
%It has to be verifed the adequacy of the w_length value, by visual inspection
%
%
%authors: 
%Oscar Barquero Perez oscar.barquero@urjc.es
%Rebeca Goya Esteban rebeca.goyaesteban@urjc.es
%Base on 7.1 Polynomial Fitting Sörnmo & Laguna Bioelectrical Signal
%Processing

if nargin < 4
    drawflag = 0;
end

%first time point zero
t = t - t(1);

L_s=length(signal);


%w_length = [13,25], default w_length = 15 seg => 25 = 0.04
%L_w=round(fs*w_length); %(L_w=50 muestras cuando fs=200 s^-1)

numSeg = floor(t(end)/w_length);
t_m = zeros(numSeg,1);
s_m = zeros(numSeg,1);

for m = 1:numSeg

    ind_seg = (t >= (m-1)*w_length) & (t < m*w_length);
%     start_p = find(t >= (m-1)*w_length);
%     end_p = find(t < m*w_length);
    %control that there is no sample in the window
    
    if sum(ind_seg) == 0
        t_m(m) = nan;
        s_m(m) = nan;
        continue
    end
    %points
    t_aux = t(ind_seg);
    t_m(m) = t_aux(round(length(t_aux)/2));
    s_m(m) = mean(signal(ind_seg));
end


%remove segments without samples
t_m(isnan(t_m)) = [];
s_m(isnan(s_m)) = [];


pp=csaps(t_m,s_m);
s_pp=ppval(pp,t);
s_det=signal-s_pp;

if drawflag
     figure(1), clf, subplot(211), plot(t,signal), 
        hold on,  plot(t,s_pp,'r-.'), axis tight;
     subplot(212), plot(t,s_det), axis tight;
     title(mensaje)
end