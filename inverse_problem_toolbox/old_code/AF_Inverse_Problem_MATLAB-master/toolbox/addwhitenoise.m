function [ outSignal, Cn] = addwhitenoise( inSignal, SNR)
% Add gaussian white noise.

% We assume constant noise power in all electrodes, then different SNR for
% each electrodes. We consider 'SNR' the mean SNR value (in dB). 

% inSignal dimensions should be [M electrodes x T times];

PowerInSig   = mean( mean(abs(inSignal).^2,2) );   
PowerInSigdB = 10*log10(PowerInSig);  % dB units.

sigma     = sqrt(10^((PowerInSigdB-SNR) / 10 ));
noise = sigma*randn(size(inSignal));
outSignal = inSignal + noise;
Cn = (sigma^2)*eye(length(outSignal(:,1))); % noise covariance matrix.

end