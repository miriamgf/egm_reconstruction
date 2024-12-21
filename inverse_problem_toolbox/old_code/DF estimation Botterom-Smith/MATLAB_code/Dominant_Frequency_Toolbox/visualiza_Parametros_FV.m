function visualiza_Parametros_FV(corrEntr)

%funcion que obtiene los parametros espectrales de un grupo de pacientes
%Permite representar tambi�n los resultados utilizando corrEntr
%load('/home/ojki/Documentos/Oscar/Investigacion/InvestigacionAnalisisEspectralEGM/BaseDatos/Serce_Para_Paper/BaseDatos_Paper/FV/FVPoblacion/patFV.mat');
[n,path] = uigetfile('../../');
load([path,n]);

close all

%Booleanos de control de paneles
cont_pat = 21;
Es_primero = 1;
Es_ultimo = 0;
salir = 0;

while salir == 0
    
    
    if Es_primero
       ex = menu(['Pat :',num2str(cont_pat),' de ',num2str(length(patFV))],'Siguiente pat','salir');
       if ex == 1
           
           Es_primero = 0;
           pinta_patFV(patFV(cont_pat),cont_pat,corrEntr);
           cont_pat = cont_pat+1;
       elseif ex == 2
           salir = 1;
           break;
       end
       
    end

    if Es_ultimo
       ex = menu(['Pat :',num2str(cont_pat),' de ',num2str(length(patFV))],'Anterior pat','salir');
       if ex == 1
           
           Es_ultimo = 0;
           pinta_patFV(patFV(cont_pat),cont_pat,corrEntr);
           cont_pat = cont_pat -1;
       elseif ex == 2
           salir = 1;
           break;
       end
       
    end

    ex = menu(['Pat :',num2str(cont_pat),' de ',num2str(length(patFV))],'Siguiente pat','Anterior pat','salir');
    
    if ex == 1
        
         pinta_patFV(patFV(cont_pat),cont_pat,corrEntr);
         cont_pat = cont_pat + 1;
        if cont_pat == length(patFV)
            Es_ultimo = 1;
        end
    elseif ex == 2
        
         pinta_patFV(patFV(cont_pat),cont_pat,corrEntr);
         cont_pat = cont_pat - 1;
        if cont_pat == 1
            Es_primero = 1;
        end
    elseif ex == 3
        salir = 1;
        break;
       
    end
    
    
     
    
    end
    
end

%% funciones auxiliares

function plot_egm_PSD(pat,fig,tipo_egm)

switch tipo_egm
    case 'egm'
        Pegm = pat.DFA.fvpar_egm.Pegm;
        f = pat.DFA.fvpar_egm.f;
        f0 = pat.DFA.fvpar_egm.f0;
        fdom = pat.DFA.fvpar_egm.fdom;
        Pdom = pat.DFA.fvpar_egm.Pdom;
        Pf0 = pat.DFA.fvpar_egm.Pf0;
        fmedian = pat.DFA.fvpar_egm.fmedian;
        f2nd = pat.DFA.fvpar_egm.f2nd;
        P2nd = pat.DFA.fvpar_egm.P2nd;
        f3rd = pat.DFA.fvpar_egm.f3rd;
        P3rd = pat.DFA.fvpar_egm.P3rd;
        f4th = pat.DFA.fvpar_egm.f4th;
        P4th = pat.DFA.fvpar_egm.P4th;
        f5th = pat.DFA.fvpar_egm.f5th;
        P5th = pat.DFA.fvpar_egm.P5th;
        m = max(pat.DFA.fvpar_egm.Pegm);
        ejex_oi = pat.DFA.fvpar_egm.ejex_oi;
        ejey_oi = pat.DFA.fvpar_egm.ejey_oi;
        ejex_oi_mod = pat.DFA.fvpar_egm.ejex_oi_mod;
        ejey_oi_mod = pat.DFA.fvpar_egm.ejey_oi_mod;
        ejexbase_oi_mod = pat.DFA.fvpar_egm.ejexbase_oi_mod;
        f0_low = pat.DFA.fvpar_egm.f0_low;
        f0_up = pat.DFA.fvpar_egm.f0_up;
        fdom_low = pat.DFA.fvpar_egm.fdom_low;
        fdom_up = pat.DFA.fvpar_egm.fdom_up;
        tit = ['Egm principal ',pat.source];
        
    case 'egm_aux'
        
        Pegm = pat.DFA.fvpar_egm_aux.Pegm;
        f = pat.DFA.fvpar_egm_aux.f;
        f0 = pat.DFA.fvpar_egm_aux.f0;
        fdom = pat.DFA.fvpar_egm_aux.fdom;
        Pdom = pat.DFA.fvpar_egm_aux.Pdom;
        Pf0 = pat.DFA.fvpar_egm_aux.Pf0;
        fmedian = pat.DFA.fvpar_egm_aux.fmedian;
        f2nd = pat.DFA.fvpar_egm_aux.f2nd;
        P2nd = pat.DFA.fvpar_egm_aux.P2nd;
        f3rd = pat.DFA.fvpar_egm_aux.f3rd;
        P3rd = pat.DFA.fvpar_egm_aux.P3rd;
        f4th = pat.DFA.fvpar_egm_aux.f4th;
        P4th = pat.DFA.fvpar_egm_aux.P4th;
        f5th = pat.DFA.fvpar_egm_aux.f5th;
        P5th = pat.DFA.fvpar_egm_aux.P5th;
        m = max(pat.DFA.fvpar_egm_aux.Pegm);
        ejex_oi = pat.DFA.fvpar_egm_aux.ejex_oi;
        ejey_oi = pat.DFA.fvpar_egm_aux.ejey_oi;
        ejex_oi_mod = pat.DFA.fvpar_egm_aux.ejex_oi_mod;
        ejey_oi_mod = pat.DFA.fvpar_egm_aux.ejey_oi_mod;
        ejexbase_oi_mod = pat.DFA.fvpar_egm_aux.ejexbase_oi_mod;
        f0_low = pat.DFA.fvpar_egm_aux.f0_low;
        f0_up = pat.DFA.fvpar_egm_aux.f0_up;
        fdom_low = pat.DFA.fvpar_egm_aux.fdom_low;
        fdom_up = pat.DFA.fvpar_egm_aux.fdom_up;
        tit = ['Egm auxiliar ',pat.source_aux];
end

figure(fig),clf
set(fig,'Name',tit)
%    m = max(fvpar.Pegm);

plot(ejex_oi,ejey_oi,'.:m',...
    ejex_oi_mod,ejey_oi_mod,'.-.c',...
    ejexbase_oi_mod,.1*m*ones(size(ejexbase_oi_mod)),':c'); hold on;
a = plot(f,Pegm,'b',fdom,Pdom,'^r',...
    f0,Pf0,'or',fmedian,0,'*r',...
    f2nd,P2nd,'ok',f3rd,P3rd,'ok',...
    f4th,P4th,'ok',f5th,P5th,'ok',...
    [f0_low,f0_low, NaN, f0_up, f0_up],[0 m NaN 0 m],'g-.',... %[fvpar.Pf0,fvpar.Pf0],'g',...
    [fdom_low,fdom_low, NaN, fdom_up, fdom_up],[0 m NaN 0 m],'r:');
legend(a,'PSD','f_{dom}','f_0','f_m','Arm�nicos','','','','B_{w} f_0','B_{w} f_{dom}')%[fvpar.Pdom,fvpar.Pdom],'r:');
%    title(['Fuente tipo ',parfv_anterior.source])
ejes = axis;
ejes(2) = 35;
axis(ejes);
%disp('Pulse cualquier tecla para continuar...');
%pause %keyboard
end

function plot_egm_Tiempo(pat,fig,tipo_egm)

switch tipo_egm
    case 'egm'
        t = pat.DFA.fvpar_egm.t;
        s = pat.egm;
        s_pp = pat.DFA.fvpar_egm.s_pp;
        s_det =pat.DFA.fvpar_egm.egm;
        tit = ['Egm principal ',pat.source];
        
    case 'egm_aux'
        t = pat.DFA.fvpar_egm_aux.t;
        s = pat.egm_aux;
        s_pp = pat.DFA.fvpar_egm_aux.s_pp;
        s_det =pat.DFA.fvpar_egm_aux.egm;
        tit = ['Egm principal ',pat.source_aux];
        
end


figure(fig),
set(fig,'Name',tit),
clf
subplot(211)
plot(t,s), hold on,  plot(t,s_pp,'r-.'), axis tight;
subplot(212), plot(t,s_det), axis tight
end

function plot_egm_Tiempo_Freq(pat,fig,tipo_egm,simus)

switch tipo_egm
    case 'egm'
        N = length(pat.DFA.fvpar_egm.egm);
        fs = pat.fs;
        egm = pat.DFA.fvpar_egm.egm;
        z = pat.DFA.fvpar_egm.z;
        if simus == 1
        [Pz,f] = pwelch(z-mean(z),boxcar(2046),[],4*4096,1600); %Espectro de la señal de características
        else
            [Pz,f] = pwelch(z-mean(z),hamming(256),[],4096,128);
        end
        Pz = Pz/sum(Pz);
        f0_low = pat.DFA.fvpar_egm.f0_low;
        f0_up = pat.DFA.fvpar_egm.f0_up;
        fdom_low = pat.DFA.fvpar_egm.fdom_low;
        fdom_up = pat.DFA.fvpar_egm.fdom_up;
        tit = ['Egm principal ',pat.source];
        
    case 'egm_aux'
        N = length(pat.DFA.fvpar_egm_aux.egm);
        fs = pat.fs;
        egm = pat.DFA.fvpar_egm_aux.egm;
        z = pat.DFA.fvpar_egm_aux.z;
        if simus == 1
        [Pz,f] = pwelch(z-mean(z),boxcar(2046),[],4*4096,1600); %Espectro de la señal de características
        else
            [Pz,f] = pwelch(z-mean(z),hamming(256),[],4096,128);
        end
        Pz = Pz/sum(Pz);
        f0_low = pat.DFA.fvpar_egm_aux.f0_low;
        f0_up = pat.DFA.fvpar_egm_aux.f0_up;
        fdom_low = pat.DFA.fvpar_egm_aux.fdom_low;
        fdom_up = pat.DFA.fvpar_egm_aux.fdom_up;
        tit = ['Egm auxiliar ',pat.source_aux];
        
        
end


figure(fig)
set(fig,'Name',tit)
N = length(egm);

% Datos en dominio de tiempo
subplot(3,1,1);
ejex = (0:N-1)/fs;  plot(ejex,egm); axis tight;
% Se�al filtrada
subplot(3,1,2);     plot(ejex,z);   axis tight;
% fft en dominio de frecuencias entre 0 y 40 HZ
% Calculo de la frecuencia fundamental
[v,indaux] = min(abs(f-2));
[M,indff]=max(Pz(indaux:end));
indff = indff+indaux-1;
 
 
subplot(3,1,3);
aux = min(find(f>30));
freq = f(1:aux);
power = Pz(1:aux);
plot (freq,power);  
axis tight;
hold on
plot(f(indaux:aux),Pz(indaux:aux),'k')
plot(f(indff),Pz(indff),'r*','MarkerSize',3)
hold on, m = max(power);
plot([f0_low,f0_low,NaN,f0_up,f0_up],[0 m NaN 0 m],'g:');
hold off
%title(['Fuente tipo ',parfv_anterior.source])
end

function writeD_pat_data(pat,p,e,corrEntr)

data = zeros(2,5)*-100;
data_b = zeros(2,5)*-100;

if corrEntr
    data = zeros(3,5)*-100;
    data_b = zeros(3,5)*-100;
end

%DFA%%%%%%%%
%f_0
data(1,1) = pat.DFA.fvpar_egm.f0;
%ri
data(1,2) = pat.DFA.fvpar_egm.oi;
%oi
data(1,3) = pat.DFA.fvpar_egm.oi_mod;
%p1
data(1,4) = pat.DFA.Modelo.Egm.probs(1);
%pe
data(1,5) = pat.DFA.Modelo.Egm.probs(2);

%FOA%%%%%%%%
%f_0
data(2,1) = pat.FOA.Egm.f0;
%p_1
data(2,4) = pat.FOA.Modelo.Egm.probs(1);
%p_e
data(2,5) = pat.FOA.Modelo.Egm.probs(2);

%CorrEntr
if corrEntr
    data(3,1) = pat.CorrEntropyX10.f0_u;
end


%Bipolar
%DFA%%%%%%%%
%f_0
data_b(1,1) = pat.DFA.fvpar_egm_aux.f0;
%ri
data_b(1,2) = pat.DFA.fvpar_egm_aux.oi;
%oi
data_b(1,3) = pat.DFA.fvpar_egm_aux.oi_mod;
%p1
data_b(1,4) = pat.DFA.Modelo.Egm_aux.probs(1);
%pe
data_b(1,5) = pat.DFA.Modelo.Egm_aux.probs(2);

%FOA%%%%%%%%
%f_0
data_b(2,1) = pat.FOA.Egm_aux.f0;
%p_1
data_b(2,4) = pat.FOA.Modelo.Egm_aux.probs(1);
%p_e
data_b(2,5) = pat.FOA.Modelo.Egm_aux.probs(2);

%CorrEntr
if corrEntr
    data_b(3,1) = pat.CorrEntropyX10.f0_b;
end

cnames = {'f_0','ri','oi','p_1','p_e'};
rnames = {'DFA','FOA'};

%CorrEntr
if corrEntr
    rnames = {'DFA','FOA','CrrEntr'};
end

f = figure('Position',[100 600 1000 300]);
t = uitable('Data',data,'ColumnName',cnames,'RowName',rnames,'Position',[20 150 418 90]);
uicontrol('Style','text','String',['Monopolar; pat ',num2str(p),' epi ',num2str(e)],'Position',[130 240 180 20]);

%f = figure('Position',[100 600 1000 300]);
t2 = uitable('Data',data_b,'ColumnName',cnames,'RowName',rnames,'Position',[430+20 150 418 90]);
uicontrol('Style','text','String',['Bipolar; pat ',num2str(p),' epi ',num2str(e)],'Position',[430+130 240 120 20]);
%keyboard;


end


%%%Preprocesado FV

function [egm,egm_aux] = preProcesadoFV(pat)

%%Acondicionamiento señales de FV

egm = pat.FOA.fvpar_egm.egm;
egm_aux = pat.FOA.fvpar_egm_aux.egm;
t = pat.FOA.t;
t_aux = pat.FOA.t_aux;

%Suavizado
% [indmax,indmin] = maxmin(egm);
% [indmax_a,indmin_a] = maxmin(egm_aux);
% [envUpper,envLower] = envelope(indmax,indmin,egm,t);
% [envUpper_a,envLower_a] = envelope(indmax_a,indmin_a,egm_aux,t_aux);
% egm = (envLower + envUpper)/2;
% egm_aux = (envLower_a + envUpper_a)/2;

%Filtrado
b = fir1(6,20/pat.fs);
egm = filter(b,1,egm);
egm_aux = filter(b,1,egm_aux);
end

%%%%%%%%%%%% Pinta pat

function pinta_patFV(pat,cont_pat,corrEntr)
close all
%Representacion de episodio
salir = 0;

for m = 1:length(pat.epi)
    s{m} = ['Episodio ',num2str(m)];
end

s{m+1} = 'Salir';

while salir == 0
   
    n_epi = menu('Seleccione episodio',s);
    
    if n_epi == length(s)
        salir = 1;
    else
   %Representaciones%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 close all
                %%Tiempo
                f1 = figure; %Representacion temporal egm
                f2 = figure; %Representacion temporal egm_aux
                plot_egm_Tiempo(pat.epi(n_epi),f1,'egm')
                plot_egm_Tiempo(pat.epi(n_epi),f2,'egm_aux')
                %%Tiempo-freq
                f3 = figure;
                f4 = figure;
                plot_egm_Tiempo_Freq(pat.epi(n_epi),f3,'egm',0)
                plot_egm_Tiempo_Freq(pat.epi(n_epi),f4,'egm_aux',0)
                %%Espectro
                f5 = figure;
                f6 = figure;
                plot_egm_PSD(pat.epi(n_epi),f5,'egm')
                plot_egm_PSD(pat.epi(n_epi),f6,'egm_aux')
                %%Representacion de los datos           
                figure(f5)
                l_f0 = line([pat.epi(n_epi).FOA.Egm.f0 pat.epi(n_epi).FOA.Egm.f0],[0 1],'LineStyle','--','Color','r');
                if corrEntr
                    f0_crr = pat.epi(n_epi).CorrEntropyX10.f0_u;
                    set(l_f0,'Xdata',f0_crr*[1 1])
                end
                figure(f6)
                l_f0_2 = line([pat.epi(n_epi).FOA.Egm_aux.f0 pat.epi(n_epi).FOA.Egm_aux.f0],[0 1],'LineStyle','--','Color','k');
                if corrEntr
                    f0_crr = pat.epi(n_epi).CorrEntropyX10.f0_b;
                    set(l_f0_2,'Xdata',f0_crr*[1 1])
                end
                
                f7 = figure;
                plot(pat.epi(n_epi).DFA.fvpar_egm.t,pat.epi(n_epi).DFA.fvpar_egm.egm,...
                    pat.epi(n_epi).DFA.fvpar_egm.t,pat.epi(n_epi).DFA.Modelo.Egm.y_est,'r',...
                    pat.epi(n_epi).DFA.fvpar_egm.t,pat.epi(n_epi).FOA.Modelo.Egm.y_est,'k')
                legend('Señal original','Señal reconstruida con f0','Señal reconstruida con BA modelo')
                set(f7,'Name',['Señal princial ',pat.epi(n_epi).source])
                f8 = figure;
                plot(pat.epi(n_epi).DFA.fvpar_egm_aux.t,pat.epi(n_epi).DFA.fvpar_egm_aux.egm,...
                    pat.epi(n_epi).DFA.fvpar_egm_aux.t,pat.epi(n_epi).DFA.Modelo.Egm_aux.y_est,'r',...
                    pat.epi(n_epi).DFA.fvpar_egm_aux.t,pat.epi(n_epi).FOA.Modelo.Egm_aux.y_est,'k')
                legend('Señal original','Señal reconstruida con f0','Señal reconstruida con BA modelo')
                set(f8,'Name',['Señal auxiliar ',pat.epi(n_epi).source_aux])
                             
                
                [P_modelo,f] = pwelch(pat.epi(n_epi).FOA.Modelo.Egm.y_est-mean(pat.epi(n_epi).FOA.Modelo.Egm.y_est)...
                    ,hamming(256),[],4096,128);
                [P_modelo_aux,f] = pwelch(pat.epi(n_epi).FOA.Modelo.Egm_aux.y_est-mean(pat.epi(n_epi).FOA.Modelo.Egm_aux.y_est)...
                    ,hamming(256),[],4096,128);
 
                
                P_modelo = P_modelo/sum(P_modelo);
                P_modelo_aux = P_modelo_aux/sum(P_modelo_aux);
                
                f9 = figure;
                f10 = figure;
                figure(f9)
                plot(pat.epi(n_epi).DFA.fvpar_egm.f,pat.epi(n_epi).DFA.fvpar_egm.Pegm,f,P_modelo,'k')
                legend('Señal original','Modelo Estimado')
                set(f9,'Name',['Epi ',num2str(n_epi),'. Tipo: ',pat.epi(n_epi).source])
                ejes = axis;
                ejes(2) = 64;
                axis(ejes)
                
                figure(f10)
                plot(pat.epi(n_epi).DFA.fvpar_egm_aux.f,pat.epi(n_epi).DFA.fvpar_egm_aux.Pegm,f,P_modelo_aux,'k')
                legend('Señal original','Modelo Estimado')
                set(f10,'Name',['Epi ',num2str(n_epi),'. Tipo: ',pat.epi(n_epi).source_aux])
                ejes = axis;
                ejes(2) = 64;
                axis(ejes)
                
                if corrEntr 
                    f11 = figure;
                    subplot(211)
                    tt =(0:length(pat.epi(n_epi).CorrEntropyX10.crr_u)-1)/pat.epi(n_epi).fs;
                    plot(tt,pat.epi(n_epi).CorrEntropyX10.crr_u);
                    xlabel('Time (s)')
                    ylabel('CorrEntropyX10 Function')
                    title('Unipolar')
                    
                    subplot(212)
                    ind  = pat.epi(n_epi).CorrEntropyX10.f_crr_u < 25;
                    plot(pat.epi(n_epi).CorrEntropyX10.f_crr_u(ind),pat.epi(n_epi).CorrEntropyX10.p_crr_u(ind));
                    xlabel('Freq (Hz)')
                    ylabel('Power Spectrum CorrEntropyX10 Function')
                    title('Unipolar')
                    
                    f12 = figure;
                    subplot(211)
                    tt =(0:length(pat.epi(n_epi).CorrEntropyX10.crr_b)-1)/pat.epi(n_epi).fs;
                    plot(tt,pat.epi(n_epi).CorrEntropyX10.crr_b);
                    xlabel('Time (s)')
                    ylabel('CorrEntropyX10 Function')
                    title('Bipolar')
                    
                    subplot(212)
                    ind  = pat.epi(n_epi).CorrEntropyX10.f_crr_b < 25;
                    plot(pat.epi(n_epi).CorrEntropyX10.f_crr_b(ind),pat.epi(n_epi).CorrEntropyX10.p_crr_b(ind));
                    xlabel('Freq (Hz)')
                    ylabel('Power Spectrum CorrEntropyX10 Function')
                    title('Bipolar')
                    
                    delete(f10); delete(f9); delete(f8); delete(f7)
                    
                end
                writeD_pat_data(pat.epi(n_epi),cont_pat,n_epi,corrEntr);

         
    end
end
end