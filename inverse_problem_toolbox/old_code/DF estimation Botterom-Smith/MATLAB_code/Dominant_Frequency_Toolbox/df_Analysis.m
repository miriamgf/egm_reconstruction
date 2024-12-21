function fvpar = df_Analysis(egm,fs,condiciones,byHandFlag,parfv_anterior,mensaje)
%fvpar = df_Analysis(egm,fs,condiciones,byHandFlag,parfv_anterior,mensaje)
%----------------------------------------------------------------
% Calcula ï¿½ndices de un episodio de FV.
% Recibe un segmento de un episodio de FV, frecuencia de muestreo
%   y parï¿½metros de los algoritmos de cï¿½lculo.
% Devuelve struct con los ï¿½ndices calculados.

%!!!!!!!!!!!!!!!!!!
%HAY QUE VERIFICAR LOS CÁLCULOS DE RI OI. 
%HASTA AHORA oi == ri;
%oi_mod = oi; !!!!!!!!!!!!!!!!!!
%----------------------------------------------------------------
% Mapa de figuras:
% figura 1: FV con y sin tendencia
% figure 2: frecuencia fundamental
% figure 3: espectro e ï¿½ndices espectrales asociados
% figure 4: histograma N-alfa 
%----------------------------------------------------------------

if nargin>=3;
    if isempty(condiciones)
        condiciones.w='hamming';
        condiciones.L=256;
        condiciones.R=0.5;
        condiciones.nfft=1024;
        condiciones.fl1=40; %It was previously 30 by JL but I think it should be 40
        condiciones.fh1=250; %It was previously 50 by JL but I think it should be 250 
        condiciones.fh2=20; %%It was previously 15 by JL but I think it should be 20
        condiciones.diff=1;
        condiciones.dfn=1/3;
        condiciones.ro=.75;
        condiciones.T=2;
        condiciones.mideltat=0.1;
        condiciones.dh_hist=0.2;
        condiciones.oi=1;
    end
end
if nargin == 2
    condiciones.w='hamming';
    condiciones.L=256;
    condiciones.R=0.5;
    condiciones.nfft=4096; %1024;
    condiciones.fl1=40; %It was previously 30 by JL but I think it should be 40
    condiciones.fh1=250; %It was previously 50 by JL but I think it should be 250
    condiciones.fh2=20; %%It was previously 15 by JL but I think it should be 20
    condiciones.diff=1;
    condiciones.dfn=1/3;
    condiciones.ro=.75;
    condiciones.T=2;
    condiciones.mideltat=0.1;
    condiciones.dh_hist=0.2;
    condiciones.oi=1;
end
if nargin<4
    byHandFlag = 0;
    mensaje = [];
    parfv_anterior = [];
end


%----------------------------------
% Inicializaciones
%----------------------------------

drawflag1 = 0;       % representaciones grï¿½ficas de cada figure
drawflag2 = 0;       
drawflag3 = 0;
drawflag4 = 0;
normflag=1;         % normalizaciï¿½n del espectro por la potencia total
N = length(egm);    % nï¿½mero de muestras del egm original
t = 1/fs*(0:N-1);   % vector tiempo  


%----------------------------------
% Eliminar tendencias con spline
%----------------------------------

[egm,s_pp] = detrendSpline(egm,fs,drawflag1,mensaje);

%----------------------------------
% egm y vector de tiempos
%----------------------------------

fvpar.egm=egm;
fvpar.s_pp = s_pp;
fvpar.t=t;



%----------------------------------
% representaciï¿½n espectral
%----------------------------------

% Representacion espectral
[fvpar.Pegm,fvpar.f] = spectralEstimation(egm,fs,normflag,condiciones);

%------------------------------
% parï¿½metros espectrales
%------------------------------


% Frecuencia dominante
[fvpar.Pdom,indfdom] = max(fvpar.Pegm);
fvpar.fdom = fvpar.f(indfdom);

% Frecuencia fundamental
[fvpar.f0,indff,bwfromf0,fvpar.z] = df_Ng(egm,fs,drawflag2,normflag,condiciones,byHandFlag,parfv_anterior);
fvpar.Pf0 = fvpar.Pegm(indff);
% [kk,aux] = min(abs(fvpar.f-fvpar.f0));
% fvpar.Pf0 = max(fvpar.Pegm(max([1 aux-5]):aux+5));

% Armï¿½nicos superiores
[fvpar.P2nd,fvpar.P3rd,fvpar.P4th,fvpar.P5th,fvpar.f2nd,fvpar.f3rd,fvpar.f4th,fvpar.f5th] = ...
    higherHarmonic(fvpar.Pegm,fvpar.f,fvpar.f0,condiciones);

% Anchos de banda de f0 y fdom
[fvpar.bw_f0,fvpar.f0_low,fvpar.f0_up] = bandWidth(fvpar.Pegm,fvpar.f,fvpar.f0,condiciones.ro);
[fvpar.bw_fdom,fvpar.fdom_low,fvpar.fdom_up] = bandWidth(fvpar.Pegm,fvpar.f,fvpar.fdom,condiciones.ro);

% Indice de organizacion
[fvpar.oi,fvpar.ejex_oi,fvpar.ejey_oi]=organizationIndex(fvpar.Pegm,fvpar.f,fvpar.fdom,condiciones.oi);

% Frecuencia mediana
fvpar.fmedian = sum(fvpar.Pegm.*fvpar.f)/sum(fvpar.Pegm);

% Leakage
fvpar.leakage = myleakage(egm,fs,fvpar.f0);

% Complementarias espectrales
% Ancho de banda de f0 obtenido en la seï¿½al de caracterï¿½sticas
fvpar.bw_f0_mod = bwfromf0;
% Regularity index corregido
[fvpar.oi_mod,fvpar.ejex_oi_mod,fvpar.ejey_oi_mod,fvpar.ejexbase_oi_mod] = ...
    riHC(egm,fvpar.f0,fvpar.Pegm,fvpar.f);

%-----------------------------------
% Medidas basadas en Teorï¿½a del Caos
%-----------------------------------

% Histograma N-alpha
fvpar.Nalpha_hist = miNalphahist(egm,fs,drawflag4,condiciones,parfv_anterior);


%----------------------------------
% Representaciï¿½n grï¿½fica
%----------------------------------
%drawflag3 = 0;
if drawflag3    
    figure(3),clf        
    m = max(fvpar.Pegm);
    plot(fvpar.f,fvpar.Pegm,'b',fvpar.fdom,fvpar.Pdom,'xr',...
        fvpar.f0,fvpar.Pf0,'or',fvpar.fmedian,0,'*r',...
        fvpar.f2nd,fvpar.P2nd,'ok',fvpar.f3rd,fvpar.P3rd,'ok',...
        fvpar.f4th,fvpar.P4th,'ok',fvpar.f5th,fvpar.P5th,'ok',...
        [fvpar.f0_low,fvpar.f0_low, NaN, fvpar.f0_up, fvpar.f0_up],[0 m NaN 0 m],'g-.',... %[fvpar.Pf0,fvpar.Pf0],'g',...
        [fvpar.fdom_low,fvpar.fdom_low, NaN, fvpar.fdom_up, fvpar.fdom_up],[0 m NaN 0 m],'r:');%[fvpar.Pdom,fvpar.Pdom],'r:');  
%    title(['Fuente tipo ',parfv_anterior.source])
    ejes = axis;
    ejes(2) = 35;
    axis(ejes);
    legend('PSD','f_{dom}','f_0','f_m','Armï¿½nicos','','','','B_{w} f_0','B_{w} f_{dom}')
    hold on, plot(fvpar.ejex_oi,fvpar.ejey_oi,'.:m',...
        fvpar.ejex_oi_mod,fvpar.ejey_oi_mod,'.-.c',...
        fvpar.ejexbase_oi_mod,.1*m*ones(size(fvpar.ejexbase_oi_mod)),':c'); hold off;
    %disp('Pulse cualquier tecla para continuar...');
    %pause %keyboard
end