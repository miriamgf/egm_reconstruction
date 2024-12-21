function [ff,indff,bwfromf0,z,Pz,fz] = df_Ng(egm,fs,flag_harm_corr,drawflag,normflag,condiciones,byHandFlag,parfv_anterior)
%
%Function that computes the dominant frequency fd using the algorithm
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
%[ff,indff,bwfromf0,z] = df_Ng(egm,fs,drawflag,normflag,condiciones,byHandFlag,parfv_anterior)

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
if nargin == 3, 
    drawflag = 0;
    normflag = 1;
    byHandFlag = 0;
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

%This allows to correct the df manually to estimate  the f0 instead of
%dominant frequency approach.
if byHandFlag == 1
    [Pegm,f] = miespectro(egm,fs,normflag,condiciones);
    figure(3)
    plot(Pegm,'.-.'); axis tight; ejes = axis; ejes(2) = ejes(2)/16; axis(ejes);
    %title(['Introduce posici�n de la frecuencia fundamental. Fuente tipo ',parfv_anterior.source])
    [x,y] = ginput(1);
    x = round(x);
    [kk,m_pos] = max(Pegm(x-3:x+3));
    indff = x-3+m_pos-1;
    ff = f(indff);
    [bwfromf0] = anchoBanda(Pegm,f,ff,condiciones.ro);
    return    
elseif byHandFlag == 2
    Pegm=parfv_anterior.Pegm;
    ff=parfv_anterior.f0;
    f=parfv_anterior.f;
    indff=find(f==ff);
    [bwfromf0] = anchoBanda(Pegm,f,ff,condiciones.ro);
    Pegm=parfv_anterior.Pegm;
end

% Filtro de Butterworth entre 40 y 250 Hz y 
% rectificar la se�al
fn = fs/2;
%fn = fs;
wn = [condiciones.fl1/fn condiciones.fh1/fn];

%%Verify that wn(2) is not higher than 1
if wn(2)>=1
    wn(2) = 1 -10*eps;
end

[b1,a1] = butter(5,wn);
y=filtfilt(b1,a1,egm);
yabs = abs((y));
% Filtro de Butterworth de orden 2 pasabaja de 20 Hz
[b2,a2] = butter(5,condiciones.fh2/fn,'low');
z = filtfilt(b2,a2,detrend((yabs)));

if 0%condiciones.diff I THINK THAT USING THE CORRECT CUT OFF FREQUENCIES WITH
    %THE CORRECT NORMALIZATION (FN = FS/2), THERE IS NO NEED OF THIS DIFF
    z = detrend(gradient(z));
else
    z = detrend(z);
end
% Calculamos la DEP

if byHandFlag == 0
[Pegm,f] = spectralEstimation(z,fs,normflag,condiciones);

% Calculo de la frecuencia fundamental
[v,indaux] = min(abs(f-2));
[M,indff]=max(Pegm(indaux:end));

    indff = indff+indaux-1;
    ff = f(indff);
end

[bwfromf0,flow,fup] = bandWidth(Pegm,f,ff,condiciones.ro);

% Calculamos el ancho de banda en la se�al de caracteristicas

Pz = Pegm;
fz = f;

%% Harmonic correction
if flag_harm_corr
[~,idx] = max(Pz);
% Multiple of 3:
[~,pf] = min(abs(f-f(idx)/3));
thmin = f(pf)-0.5;
thmax = f(pf)+0.5;
[~,idxmin1] = min(abs(f-thmin));
[~,idxmax1] = min(abs(f-thmax));
% Multiple of 2:
[~,pf] = min(abs(f-f(idx)/2));
thmin = f(pf)-0.5;
thmax = f(pf)+0.5;
[~,idxmin2] = min(abs(f-thmin));
[~,idxmax2] = min(abs(f-thmax));
if ~isempty(find(Pz(idxmin1:idxmax1)>0.1*max(Pz)))
    [~,pos] = max(Pz(idxmin1:idxmax1));
    pos = pos+idxmin1-1;
elseif ~isempty(find(Pz(idxmin2:idxmax2)>0.1*max(Pz)))
    [~,pos] = max(Pz(idxmin2:idxmax2));
    pos = pos+idxmin2-1;
else
    [~, pos] = max(Pz);
end

ff = f(pos);
end

%%
if drawflag==1,
    N = length(egm);
    % Datos en dominio de tiempo
    figure(2);          subplot(3,1,1);    
    ejex = (0:N-1)/fs;  plot(ejex,egm); axis tight;
    % Se�al filtrada
    subplot(3,1,2);     plot(ejex,z);   axis tight;
    % fft en dominio de frecuencias entre 0 y 40 HZ
    subplot(3,1,3);
    aux = min(find(f>30));
    freq = f(1:aux);
    power = Pegm(1:aux);
    plot (freq,power);  axis tight;
    hold on, m = max(power);
    plot([flow,flow,NaN,fup,fup],[0 m NaN 0 m],'g:');
    hold off
    %title(['Fuente tipo ',parfv_anterior.source])
end