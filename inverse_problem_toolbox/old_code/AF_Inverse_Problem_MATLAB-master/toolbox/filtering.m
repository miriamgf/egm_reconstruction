function [y_noise, Cn] = filtering (y, SNR, fs, model, f_low, f_high)
% function which adds noise, applies pre-filter and narrow-band filter to torso potentials. 
%
% This routine by Víctor Suárez Gutiérrez (victor.suarez.gutierrez@urjc.es)
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUT:
% y: torso potentials.
% SNR: Signal to Noise Ratio
% fs: sampling frequency.
% model: place where rotor occurs {'SR', 'SAF', 'CAF'}
% f_low: low cutoff frequency for filtering.
% f_high: high cutoff frequency for filtering.
%
%OUTPUT:
% y_noise: filtered torso potentials.
% Cn: noise covariance matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng('default') % for reproducibility 

% % Add noise to y
[y_noise, Cn] = addwhitenoise(y, SNR); 

% Filter
y_noise = preprocessFA (y_noise, fs, f_low, f_high, model);

end   
        


