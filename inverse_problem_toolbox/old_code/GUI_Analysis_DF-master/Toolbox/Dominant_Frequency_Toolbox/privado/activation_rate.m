function [ar_sample_dect,ar_ms_mean,ar_ms_std] = activation_rate(egm,fs)
%
%Function that computes the time activation rate from an EGM.
%It uses the preprocessed EGM from the botterom algorithm.
%The it computes the maximum using a qrs detector approach


%%botterom preprocessing Without botterom preprocessing works well
% [~,~,~,z] = df_Ng(egm,fs);

%%maxima detection
[~,r_detect,~] = qrs_detection(egm,fs);

%%conversiion
ar_sample_dect = r_detect;
ar_ms = diff(ar_sample_dect)/fs;
ar_ms_mean = mean(ar_ms);
ar_ms_std = std(ar_ms);

