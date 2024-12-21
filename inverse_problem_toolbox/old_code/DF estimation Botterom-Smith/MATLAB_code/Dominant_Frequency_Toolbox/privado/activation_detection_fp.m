function [ind_d] = activation_detection_fp(x,refractory_period,fs)
%
%Function that detects the activation detections on an emg in atrial
%fibrillation.
%
%Inputs parameters:
%   1) x => egm
%   2) refractory_period => in msecs
%   3) fs => sampling frequency in Hertzs
%
%Output parameters:
%   1) ind_d => activation detectoins indices

%Refractory period conversion to samples

r_p_s = floor(refractory_period/1000 * fs);


[~,ind_d] = findpeaks(x,'MINPEAKDISTANCE',r_p_s);