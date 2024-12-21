function [s_det,s_pp] = detrendSpline(s,fs,drawflag,mensaje)
%Function that remove base line noise using cubic splines
%
%[s_det,s_pp] = detrendSpline(s,fs,drawflag,mensaje)
%
%Issue: the free parameter is the length of the window in seconds T_w.
%It has to be verifed the adequacy of the T_w value, by visual inspection
%
%df_toolbox
%
%
L_s=length(s);

%L_w=100;
T_w=.25; %duraci�n de la ventana en segundos
L_w=round(fs*T_w); %(L_w=50 muestras cuando fs=200 s^-1)

t = 1/fs*(0:L_s-1);
t = t';
t_m=[];
s_m=[];

for n_t=1:round(L_w/2):L_s-L_w
    t_m=[t_m t(n_t+round(L_w/2)-1)];
    s_m=[s_m mean(s(n_t:n_t+L_w-1))];
end

pp=csaps(t_m,s_m);
s_pp=ppval(pp,t);
s_det=s-s_pp;

if drawflag,
     figure(1), clf, subplot(211), plot(t,s), 
        hold on,  plot(t,s_pp,'r-.'), axis tight;
     subplot(212), plot(t,s_det), axis tight;
     title(mensaje)
end
%--------------------------------------------------------------------------
