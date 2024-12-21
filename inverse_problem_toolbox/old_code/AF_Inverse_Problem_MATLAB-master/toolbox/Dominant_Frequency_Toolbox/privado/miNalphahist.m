function Nalpha_hist = miNalphahist(egm,fs,plot_flag,condiciones,parfv_anterior)
%
%Function that computes (what?!!) Nalpha_hist
%Nalpha_hist = miNalphahist(egm,fs,plot_flag,condiciones,parfv_anterior)
%
%It uses the condiciones struct with the following fields
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
%TO_DO CHECK THE REFERENCES, AND THE COMPUTATIONS

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
   % title(['Fuente tipo ',parfv_anterior.source])
end
