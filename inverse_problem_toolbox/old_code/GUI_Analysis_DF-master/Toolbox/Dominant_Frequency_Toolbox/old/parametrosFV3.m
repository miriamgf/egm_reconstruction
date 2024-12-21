function fvpar = parametrosFV2(egm,fs,condiciones,byHandFlag,parfv_anterior,mensaje)

%----------------------------------------------------------------
% Calcula �ndices de un episodio de FV.
% Recibe un segmento de un episodio de FV, frecuencia de muestreo
%   y par�metros de los algoritmos de c�lculo.
% Devuelve struct con los �ndices calculados.
%----------------------------------------------------------------
% Mapa de figuras:
% figura 1: FV con y sin tendencia
% figure 2: frecuencia fundamental
% figure 3: espectro e �ndices espectrales asociados
% figure 4: histograma N-alfa 
%----------------------------------------------------------------

if nargin>=3;
    if isempty(condiciones)
        condiciones.w='hamming';
        condiciones.L=256;
        condiciones.R=0.5;
        condiciones.nfft=1024;
        condiciones.fl1=30;
        condiciones.fh1=50;
        condiciones.fh2=15; % 40
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
    condiciones.fl1=30;
    condiciones.fh1=50;
    condiciones.fh2=15; % 40
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

drawflag1 = 0;       % representaciones gr�ficas de cada figure
drawflag2 = 0;       
drawflag3 = 0;
drawflag4 = 0;
normflag=1;         % normalizaci�n del espectro por la potencia total
N = length(egm);    % n�mero de muestras del egm original
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
% representaci�n espectral
%----------------------------------

% Representacion espectral
[fvpar.Pegm,fvpar.f] = miespectro(egm,fs,normflag,condiciones);

%------------------------------
% par�metros espectrales
%------------------------------


% Frecuencia dominante
[fvpar.Pdom,indfdom] = max(fvpar.Pegm);
fvpar.fdom = fvpar.f(indfdom);

% Frecuencia fundamental
[fvpar.f0,indff,bwfromf0,fvpar.z] = freq_fundamental(egm,fs,drawflag2,normflag,condiciones,byHandFlag,parfv_anterior);
fvpar.Pf0 = fvpar.Pegm(indff);
% [kk,aux] = min(abs(fvpar.f-fvpar.f0));
% fvpar.Pf0 = max(fvpar.Pegm(max([1 aux-5]):aux+5));

% Arm�nicos superiores
[fvpar.P2nd,fvpar.P3rd,fvpar.P4th,fvpar.P5th,fvpar.f2nd,fvpar.f3rd,fvpar.f4th,fvpar.f5th] = ...
    higherHarmonic(fvpar.Pegm,fvpar.f,fvpar.f0,condiciones);

% Anchos de banda de f0 y fdom
[fvpar.bw_f0,fvpar.f0_low,fvpar.f0_up] = anchoBanda(fvpar.Pegm,fvpar.f,fvpar.f0,condiciones.ro);
[fvpar.bw_fdom,fvpar.fdom_low,fvpar.fdom_up] = anchoBanda(fvpar.Pegm,fvpar.f,fvpar.fdom,condiciones.ro);

% Indice de organizacion
[fvpar.oi,fvpar.ejex_oi,fvpar.ejey_oi]=organizationIndex(fvpar,condiciones.oi);

% Frecuencia mediana
fvpar.fmedian = sum(fvpar.Pegm.*fvpar.f)/sum(fvpar.Pegm);

% Leakage
fvpar.leakage = mileakage(egm,fs,fvpar.f0);

% Complementarias espectrales
% Ancho de banda de f0 obtenido en la se�al de caracter�sticas
fvpar.bw_f0_mod = bwfromf0;
% Regularity index corregido
[fvpar.oi_mod,fvpar.ejex_oi_mod,fvpar.ejey_oi_mod,fvpar.ejexbase_oi_mod] = ...
    myriHC(egm,fvpar.f0,fs,fvpar.Pegm,fvpar.f);

%-----------------------------------
% Medidas basadas en Teor�a del Caos
%-----------------------------------

% Histograma N-alpha
fvpar.Nalpha_hist = miNalphahist(egm,fs,drawflag4,condiciones,parfv_anterior);


%----------------------------------
% Representaci�n gr�fica
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
    legend('PSD','f_{dom}','f_0','f_m','Arm�nicos','','','','B_{w} f_0','B_{w} f_{dom}')
    hold on, plot(fvpar.ejex_oi,fvpar.ejey_oi,'.:m',...
        fvpar.ejex_oi_mod,fvpar.ejey_oi_mod,'.-.c',...
        fvpar.ejexbase_oi_mod,.1*m*ones(size(fvpar.ejexbase_oi_mod)),':c'); hold off;
    %disp('Pulse cualquier tecla para continuar...');
    %pause %keyboard
end



% ##################################################
% ########### FUNCIONES AUXILIARES #################
% ##################################################


function [P,f] = miespectro(egm,fs,norm_flag,condiciones)

nwindow = condiciones.L;  
noverlap = round(nwindow*condiciones.R);
nfft = condiciones.nfft;

[P,f] = pwelch(egm-mean(egm),...
    feval(condiciones.w,nwindow),noverlap,nfft,fs);

if norm_flag
    P=P/sum(P);
end
%--------------------------------------------------------------------------



function [ff,indff,bwfromf0,z] = freq_fundamental(egm,fs,drawflag,normflag,condiciones,byHandFlag,parfv_anterior)

% Calculo de la frecuencia fundamental
if nargin == 2, drawflag=0; end;

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
wn = [condiciones.fl1/fs condiciones.fh1/fs];    
[b1,a1] = butter(2,wn);
y=filtfilt(b1,a1,egm);
yabs = abs((y));
% Filtro de Butterworth de orden 2 pasabaja de 20 Hz
[b2,a2] = butter(2,condiciones.fh2/fs);
z = filtfilt(b2,a2,detrend((yabs)));
if condiciones.diff
    z = detrend(gradient(z));
else
    z = detrend(z);
end
% Calculamos la DEP

if byHandFlag == 0
[Pegm,f] = miespectro(z,fs,normflag,condiciones);

% Calculo de la frecuencia fundamental
[v,indaux] = min(abs(f-2));
[M,indff]=max(Pegm(indaux:end));

    indff = indff+indaux-1;
    ff = f(indff);
end

[bwfromf0,flow,fup] = anchoBanda(Pegm,f,ff,condiciones.ro);

% Calculamos el ancho de banda en la se�al de caracteristicas


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



%--------------------------------------------------------------------------



function [P2nd,P3rd,P4th,P5th,f2nd,f3rd,f4th,f5th] = higherHarmonic(Pegm,f,f0,condiciones)

% Entrada: DEP, eje de frecuencias, f0
% Salida: P del segundo y tercer arm�nico

radio = f0*condiciones.dfn;

[kk,ini] = min(abs(f-(2*f0-radio)));
[kk,fin] = min(abs(f-(2*f0+radio)));
[P2nd,ind] = max(Pegm(ini:fin));
f2nd = f(ind+ini-1);

[kk,ini] = min(abs(f-(3*f0-radio)));
[kk,fin] = min(abs(f-(3*f0+radio)));
[P3rd,ind] = max(Pegm(ini:fin));
f3rd = f(ind+ini-1);

[kk,ini] = min(abs(f-(4*f0-radio)));
[kk,fin] = min(abs(f-(4*f0+radio)));
[P4th,ind] = max(Pegm(ini:fin));
f4th = f(ind+ini-1);

[kk,ini] = min(abs(f-(5*f0-radio)));
[kk,fin] = min(abs(f-(5*f0+radio)));
[P5th,ind] = max(Pegm(ini:fin));
f5th = f(ind+ini-1);
%--------------------------------------------------------------------------



function [bw,flow,fup] = anchoBanda(P,f,f0,level)

[kk,ind0] = min(abs(f-f0));
indinf = ind0;
if nargin==3,
    level = .75;
end
while 1
    indinf = indinf-1;
    if indinf==1,break,end;
    if P(indinf)<level*P(ind0)
        break
    end
end
indsup = ind0;
while 1
    indsup = indsup+1;
    if indsup==length(P),break,end;
    if P(indsup)<level*P(ind0)
        break
    end
end
flow = f(indinf);  fup = f(indsup);
bw = fup - flow;
%--------------------------------------------------------------------------



function [oi,ejexplot,ejeyplot] = organizationIndex(fvpar,ancho)

Pegm=fvpar.Pegm;
f=fvpar.f;
radio=ancho/2;
ew=0;


[kk,ini] = min(abs(f-(fvpar.fdom-radio)));
[kk,fin] = min(abs(f-(fvpar.fdom+radio)));
ew=sum(Pegm(ini:fin))+ew;
ejexplot = [f(ini:fin)', NaN]; 
ejeyplot = [Pegm(ini:fin)', NaN];


%[kk,ini] = min(abs(f-(fvpar.f2nd-radio)));
%[kk,fin] = min(abs(f-(fvpar.f2nd+radio)));
%ew=sum(Pegm(ini:fin))+ew;
%ejexplot = [ejexplot, f(ini:fin)', NaN]; 
%ejeyplot = [ejeyplot, Pegm(ini:fin)', NaN];

%[kk,ini] = min(abs(f-(fvpar.f3rd-radio)));
%[kk,fin] = min(abs(f-(fvpar.f3rd+radio)));
%ew=sum(Pegm(ini:fin))+ew;
%ejexplot = [ejexplot, f(ini:fin)', NaN]; 
%ejeyplot = [ejeyplot, Pegm(ini:fin)', NaN];

%[kk,ini] = min(abs(f-(fvpar.f4th-radio)));
%[kk,fin] = min(abs(f-(fvpar.f4th+radio)));
%ew=sum(Pegm(ini:fin))+ew;
%ejexplot = [ejexplot, f(ini:fin)', NaN]; 
%ejeyplot = [ejeyplot, Pegm(ini:fin)', NaN];

% [kk,ini] = min(abs(f-(fvpar.f5th-radio)));
% [kk,fin] = min(abs(f-(fvpar.f5th+radio)));
% ew=sum(Pegm(ini:fin))+ew;
% ejexplot = [ejexplot, f(ini:fin)']; 
% ejeyplot = [ejeyplot, Pegm(ini:fin)'];

%[kk,ini] = min(abs(f-2.5));
[kk,ini] = min(abs(f-1));
%[kk,fin] = min(abs(f-(fvpar.f5th-radio)));
[kk,fin] = min(abs(f-35));

oi=ew/sum(Pegm(ini:fin));


%--------------------------------------------------------------------------



function leakage = mileakage(egm,fs,f0)

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
%--------------------------------------------------------------------------



function Nalpha_hist = miNalphahist(egm,fs,plot_flag,condiciones,parfv_anterior)

%mideltat = .1:.01:.15;   % CAMBIO: .1:.1:.15!!!
mideltat = round(fs*condiciones.mideltat); % Retardos en muestras
d = condiciones.T+1;
egm = egm(:)';
horiz = 0:condiciones.dh_hist:15; %CAMBIO 0:.2:15; ahora dh_hist=0.02!!!
Y = zeros(size(horiz));

for dd = 1:length(mideltat)    
    L_v=[];
    for num_v=1:d
        L_v=[L_v (length(egm)-(d-num_v)*mideltat(dd)-(num_v-1)*mideltat(dd))];
    end    
    L_V=min(L_v);
    V=zeros(d,L_V);
    for num_v=1:d
        V(num_v,:)=egm(1 + (num_v-1)*mideltat(dd):1 + (num_v-1)*mideltat(dd)+(L_V-1)); %ojo esto en el antiguo est� cambiao y los vectores empiezan en N*mideltat en lugra de N*mideltat+1
    end
    r2 = ceil(L_V/100*2);
    r5 = ceil(L_V/100*5);
    alphas = zeros(1,L_V);    
    for i=1:L_V
        distancias = sum((V-repmat(V(:,i),1,L_V)).^2,1);
        distancias = sort(distancias);
        bint = regress([log10(cumsum(r2:r5))'],...
            [log10(distancias(r2:r5))',ones(r5-r2+1,1)]);
        alphas(i) = bint(1);
    end
    Y = Y +  hist(alphas,horiz);
end
Y = Y/length(mideltat);


% Parametro de salida
cumY = cumsum(Y);
area = sum(Y);
lowalpha = horiz(min(find(cumY>.25*area)));
upalpha = horiz(min(find(cumY>.75*area)));
width = upalpha - lowalpha;
start = horiz(min(find(cumY>.05*area)));

Nalpha_hist.startbywith = start/width;
Nalpha_hist.width = width;

if plot_flag
    figure(4)    
    plot(horiz,Y), axis tight
    xlabel('\alpha')
    hold on, m = max(Y);
    plot([start, start, NaN, start+width,start+width],...
        [0 m NaN 0 m],':r');
    hold off
    title(['Fuente tipo ',parfv_anterior.source])
end

%--------------------------------------------------------------------------