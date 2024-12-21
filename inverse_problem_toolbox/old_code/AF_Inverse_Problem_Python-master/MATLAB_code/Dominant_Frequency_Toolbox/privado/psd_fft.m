function [y_esp,f_esp] = psd_fft(y,Fs)
L = length(y);
y = y - mean(y);
%NFFT = 2^nextpow2(L); % Next power of 2 from length of y
NFFT = L;
y_esp = fft(y,NFFT)/L;
f_esp = Fs/2*linspace(0,1,NFFT/2+1);

y_esp = 2*abs(y_esp(1:NFFT/2+1));

% Plot single-sided amplitude spectrum.
% plot(f,2*abs(Y(1:NFFT/2+1))) 
% title('Single-Sided Amplitude Spectrum of y(t)')
% xlabel('Frequency (Hz)')
% ylabel('|Y(f)|')

end