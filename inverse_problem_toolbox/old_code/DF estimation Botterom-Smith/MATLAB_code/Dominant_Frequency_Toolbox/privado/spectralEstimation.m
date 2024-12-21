function [P,f] = spectralEstimation(egm,fs,norm_flag,condiciones)
%Function that estimate de psd from the emg, using the struct condiciones
%as input of the parameters to be used in the welch periodogram.
%
%[P,f] = spectralEstimation(egm,fs,norm_flag,condiciones)
%
%condicions has the following fields:
%
%     condiciones.w='hamming';
%     condiciones.L=256;
%     condiciones.R=0.5;
%     condiciones.nfft=4096; %1024;
%     condiciones.fl1=30;
%     condiciones.fh1=50;
%     condiciones.fh2=15; % 40
%     condiciones.diff=1;
%     condiciones.dfn=1/3;
%     condiciones.ro=.75;
%     condiciones.T=2;
%     condiciones.mideltat=0.1;
%     condiciones.dh_hist=0.2;
%     condiciones.oi=1;
%
%condiciones has also other parameters for other functions. In this
%function only, condiciones.L, condiciones.R, condiciones.w and
%condiciones.nfft are use.
%
%df_toolbox
%

if nargin < 4
    %condiciones standard
    condiciones.w = 'hanning';
    condiciones.L = length(egm); %ordinary periodogram
    condiciones.R = 0.5;
    condiciones.nfft = fs/0.1; %frequency resolution 0.1 Hz
    condiciones.fl1 = 40; %high-pass filter between [40-250] Hz
    condiciones.fh1 = 250;
    condiciones.fh2=20; %low-pass filter fcut 20 Hz;
    condiciones.diff=0;
    condiciones.dfn=1/3;
    condiciones.ro=.75;
    condiciones.T=2;
    condiciones.mideltat=0.1;
    condiciones.dh_hist=0.2;
    condiciones.oi=1;
end

nwindow = condiciones.L;  
noverlap = round(nwindow*condiciones.R);
nfft = condiciones.nfft;

[P,f] = pwelch(egm-mean(egm),...
    feval(condiciones.w,nwindow),noverlap,nfft,fs);

if norm_flag
    % P=P/sum(P);
    P = P/max(P);
end