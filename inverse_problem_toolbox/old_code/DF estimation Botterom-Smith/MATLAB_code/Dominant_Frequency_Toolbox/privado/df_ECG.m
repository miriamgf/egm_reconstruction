function [ff,indff,bwfromf0,z,Pz,fz] = df_ECG(ecg,fs,drawflag,normflag,condiciones,byHandFlag,parfv_anterior)
%
%Function that computes the dominant frequency in an ECG fd using the algorithm
%proposed by Ng in the articles:
% 1) Ng, J., & Goldberger, J. J. (2007). Understanding and Interpreting 
% Dominant Frequency Analysis of AF Electrograms. Journal of Cardiovascular
% Electrophysiology, 18(6), 680?685.
%
% 2) Ng, J., Kadish, A. H., & Goldberger, J. J. (2006). Effect of 
% electrogram characteristics on the relationship of dominant frequency to
% atrial activation rate in atrial fibrillation. 
% Heart rhythm, 3(11), 1295?1305.%
% 3) Ng, J., Kadish, A. H., & Goldberger, J. J. (2007). Technical 
%Considerations for Dominant Frequency Analysis. Journal of Cardiovascular 
%Electrophysiology, 18(7), 757?764.
%
%[ff,indff,bwfromf0,z] = df_Ng(ecg,fs,drawflag,normflag,condiciones,byHandFlag,parfv_anterior)

%One of its inputs is the condiciones struct with the following fields:
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
%df_toolbox



% Calculo de la frecuencia fundamental
if nargin == 2, 
    drawflag = 1;
    normflag = 1;
    byHandFlag = 0;
    %condiciones standard
    condiciones.w = 'hamming';
    condiciones.L = length(ecg); %ordinary periodogram
    condiciones.R = 0.5;
    condiciones.nfft = fs/0.1; %frequency resolution 0.1 Hz
    condiciones.fh2=20; %low-pass filter fcut 20 Hz;
    condiciones.diff=0;
    condiciones.dfn=1/3;
    condiciones.ro=.75;
    condiciones.T=2;
    condiciones.mideltat=0.1;
    condiciones.dh_hist=0.2;
    condiciones.oi=1;
    
end

%This allows to correct the df manually to estimate  the f0 instead of
%dominant frequency approach.
if byHandFlag == 1
    [Pecg,f] = miespectro(ecg,fs,normflag,condiciones);
    figure(3)
    plot(Pecg,'.-.'); axis tight; ejes = axis; ejes(2) = ejes(2)/16; axis(ejes);
    %title(['Introduce posici�n de la frecuencia fundamental. Fuente tipo ',parfv_anterior.source])
    [x,y] = ginput(1);
    x = round(x);
    [kk,m_pos] = max(Pecg(x-3:x+3));
    indff = x-3+m_pos-1;
    ff = f(indff);
    [bwfromf0] = anchoBanda(Pecg,f,ff,condiciones.ro);
    return    
elseif byHandFlag == 2
    Pecg=parfv_anterior.Pecg;
    ff=parfv_anterior.f0;
    f=parfv_anterior.f;
    indff=find(f==ff);
    [bwfromf0] = anchoBanda(Pecg,f,ff,condiciones.ro);
    Pecg=parfv_anterior.Pecg;
end


fn = fs/2;

%Rectification
%yabs = abs((ecg));

% Filtro de Butterworth de orden 2 pasabaja de 20 Hz
[b2,a2] = butter(5,condiciones.fh2/fn,'low');

z = filtfilt(b2,a2,detrend((ecg)));

if 0%condiciones.diff I THINK THAT USING THE CORRECT CUT OFF FREQUENCIES WITH
    %THE CORRECT NORMALIZATION (FN = FS/2), THERE IS NO NEED OF THIS DIFF
    z = detrend(gradient(z));
else
    z = detrend(z);
end
% Calculamos la DEP

if byHandFlag == 0
[Pecg,f] = spectralEstimation(z,fs,normflag,condiciones);

% Calculo de la frecuencia fundamental
[v,indaux] = min(abs(f-2));
[M,indff]=max(Pecg(indaux:end));

    indff = indff+indaux-1;
    ff = f(indff);
end

[bwfromf0,flow,fup] = bandWidth(Pecg,f,ff,condiciones.ro);

% Calculamos el ancho de banda en la se�al de caracteristicas

Pz = Pecg;
fz = f;

if drawflag==1,
    N = length(ecg);
    % Datos en dominio de tiempo
    figure(2);          subplot(2,1,1);    
    ejex = (0:N-1)/fs;  plot(ejex,ecg); axis tight;
    % Se�al filtrada
    hold on,     plot(ejex,z,'r');   axis tight;
    % fft en dominio de frecuencias entre 0 y 40 HZ
    subplot(2,1,2);
    indf = f <= 20;
    plot (f(indf),Pecg(indf));  axis tight;
    hold on, m = max(Pecg(indf));
    plot([flow,flow,NaN,fup,fup],[0 m NaN 0 m],'g:');
    hold off
    %title(['Fuente tipo ',parfv_anterior.source])
end