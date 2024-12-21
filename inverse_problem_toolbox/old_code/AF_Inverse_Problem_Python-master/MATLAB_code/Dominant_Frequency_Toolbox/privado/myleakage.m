function leakage = myleakage(egm,fs,f0)
%
%Function that computes the leakage from Jekova03
%
%leakage = myleakage(egm,fs,f0)
%
%df_toolbox

% Jekova03
% Preprocesado: filtro paso banda 1-30 Hz
N = 2;                 % Orden del filtro
wci = 1/(fs/2);        % Frecuencia corte inferior normalizada
wcs = 16/(fs/2);       % Frecuencia de corte superior normalizada

[B,A] = butter(N,[wci wcs]);
xfilt = filtfilt(B,A,egm);

vi = xfilt(1:end-1);
vi_1 = xfilt(2:end);
    
T2 = round(fs/f0);
vi_T2= vi(T2:end);
vi = vi(1:end-T2+1);
leakage = sum( abs( vi+vi_T2 ) ) / sum( abs(vi)+abs(vi_T2) );