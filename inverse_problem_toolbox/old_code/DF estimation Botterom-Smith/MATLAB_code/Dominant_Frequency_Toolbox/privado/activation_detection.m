function ind_detected = activation_detection(egm,fs)
%
%Detects activation waveforms in intracardiac signal in atrial fibrillation
%
%The activation detection is perfomed on the botteron-preprocessed signal


%########################
% Input parameter control
%########################




%#######################
% Botteron Preprocessing
%#######################

egm_botteron = botteron_prepro(egm,fs);
egm_botteron = detrend(egm_botteron);

%#######################
% Activation detection
%#######################

theta = zeros(size(egm));
alpha = 0.55;
beta = 0.6;
period_eye_closing = 50e-3*fs;% 50ms
%theta(1) = alpha * max(egm_botteron); % Threshold
theta(1) = alpha * median(egm_botteron); % Threshold
mu_i = theta(1);
ind_detected = [];
eye_closing = 0;
count_mu = 2; %ind for mu
mu(1) = mu_i;

for m = 2:length(egm)
    
    %detection
    if (egm_botteron(m) > theta(m-1)) && (m >= eye_closing)
        
        detect_ind = m;
        
        %find the first local maximum following the detection time
        x_aux = m:length(egm_botteron);
        [iM] = maxmin(egm_botteron(x_aux));
        detect_ind = x_aux(iM(1));
        ind_detected = [ind_detected detect_ind];
        
        %update eye_closing
        eye_closing = detect_ind + period_eye_closing;
        
        %update threshold
        z_n_i = egm_botteron(detect_ind); %last peak detected
        mu_i_1 = mu_i; %update exponential average
        mu_i = mu_i_1 + beta*(z_n_i - mu_i_1); %new value of the exponential average
        mu(count_mu) = mu_i;
        theta(m) = alpha*mu_i; %new threshold
        count_mu = count_mu + 1;
        
    else %no detection
        theta(m) = theta(m-1); %keep threshold
    end
    
end

length_ini = length(ind_detected);
%#####################
% Look-back detection
%#####################
% Look-back detection using mean time difference between detected
% activation times

pseudo_rr_int = diff(ind_detected)/fs;

m_rr_int = mean(pseudo_rr_int);

%find pseudo_rr_int > x*m_rr_int

%for those which find is 1 recompute the threshold for those activations
%waveforms with alpha = 0.35

%idx to extract mu values to recalcualte the new thresholds
gamma = 1.2;
alpha_new = 0.2;
idx = find(pseudo_rr_int > gamma * m_rr_int);

%construct a vector of new indexes referred to the original, to give
%support to the new theta values

new_x_vector = [];
theta_new = [];
ct = 1;


%for over idx to recalculate new mu
for m = 1:length(idx)
    
    %mu original
    mu_ori = mu(idx(m)+1);
    
    %find_ind previous correct detection
    ind_previous_detec = ind_detected(idx(m));
    count_x = ind_detected(idx(1));
    
    %%eye closing period
    eye_closing = count_x + period_eye_closing;
    
    new_x_vector = [new_x_vector ind_previous_detec];
    
    %new theta value
    theta_new(ct) = alpha_new * mu_ori;
    
    plot(egm_botteron)
    hold on
    plot(ind_detected(1:length_ini),egm_botteron(ind_detected(1:length_ini)),'r*')
   
    while (egm_botteron(count_x) < theta_new(ct)) || (count_x < eye_closing)
        pause(.3)
        %counters
        count_x = count_x + 1;
        ct = ct + 1;
        
        %threshold
        theta_new(ct) = theta_new(ct-1);
        
        %x_support
        new_x_vector =[new_x_vector count_x];
        
        hold on
        plot(count_x,egm_botteron(count_x),'k.','MarkerSize',4)
        plot(count_x,theta(count_x),'g.','MarkerSize',4)
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%detection in look_back mode%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    detect_ind = count_x;
    
    %find the first local maximum following the detection time
    x_aux = count_x:length(egm_botteron);
    [iM] = maxmin(egm_botteron(x_aux));
    detect_ind = x_aux(iM(1));
    ind_detected = [ind_detected detect_ind];
    
end

keyboard

%
end


%###############################
%###############################
%AUXILIARY FUNCTIONS
%###############################
%###############################

function egm_prepro = botteron_prepro(egm,fs)
%Performs botteron preprocessing

fn = fs/2; %Nyquist frequency

% Bandpass filterinfg between 40 y 250 Hz
fl1 = 40; %Hz
fh1 = 250; %Hz
wn = [fl1/fn fh1/fn];
[b1,a1] = butter(5,wn);
y=filtfilt(b1,a1,egm);

% Rectification
yabs = abs(y);

% Low pass filtering <20 Hz
fh2 = 20; %Hz
[b2,a2] = butter(5,fh2/fn,'low');
egm_prepro = filtfilt(b2,a2,detrend((yabs)));

end

function [indMax,indMin]=maxmin(x)
%[indMax,indMin]=maxmin(x)
%
%Función que busca los m·ximos y minimos locales de los datos de entrada

%No se consideran máximos ni mínimos el primer y último punto de los datos
%de entrada

%ComprobaciÛn de datos
if(size(x,2) == 1)
    x = x';
end

%Primera derivada
xdiff1 = diff(x);

%El signo de xdiff1
xs = sign(xdiff1);

%Segunda derivada para identificar los puntos en los que se cambia de signo
xdiff2 = diff(xs);

%Indices máximos
auxindMax = [0,xdiff2<0,0];

indMax = find(auxindMax);

%Indices mÌnimos
auxindMin = [0,xdiff2>0,0];
indMin = find(auxindMin);
end