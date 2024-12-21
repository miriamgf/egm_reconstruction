Fs=1000; %Hz
Fmin=3;
Fmax=30;

N_min=round(Fs/Fmax);
N_max=round(Fs/Fmin);
index=1;
min_std=10000;
for window_length=N_min:N_max,
%     disp(['Length ' num2str(window_length) ' of ' num2str(N_max)])
    SzX=window_length;
    SzY(index)=floor(length(Electrogram)/window_length);
    A=zeros(SzY(index),SzX);
    for i=1:SzY(index),
        A(i,:)=(Electrogram((i-1)*SzX+1:i*SzX))/(eps+max(Electrogram((i-1)*SzX+1:i*SzX)));
    end
    [dump rot_i1]=max(A(1,:));
    for i=2:SzY(index),
        [dump rot_i]=max(A(i,:));
        A(i,:)=circshift(A(i,:)',rot_i1-rot_i);
    end

    Lambda=svd(A);
    Norm_coeff(index)=std(sum(A'))*sqrt(window_length);
    SVR(index)=Lambda(1)/Lambda(2)/Norm_coeff(index);
    if std(sum(A'))<min_std
        min_std=std(sum(A'));
    end
    index=index+1;
end
%SVR=SVR*min_std/std(Electrogram);
SVR=SVR/max(SVR);
freq_range=Fs./[N_min:N_max];   

[Vt DF]=max(SVR);
Freq_low_limit=max(freq_range(DF)-0.75,Fmin);
Freq_high_limit=min(freq_range(DF)+0.75,Fmax);
Freq_high_idx=min(find(freq_range<=Freq_low_limit));
Freq_low_idx=max(find(freq_range>=Freq_high_limit));
DF_energy=sum(SVR(1:min(Freq_high_idx+1,length(SVR))))-sum(SVR(1:Freq_low_idx-1));
Total_energy=sum(SVR);
RI=DF_energy/Total_energy;


% Calculating PSD and its DF and RI

Hs=spectrum.periodogram;
sig_fft=psd(Hs,Electrogram,'Fs',Fs);
Fmin_fft=min(find(sig_fft.Frequencies>=Fmin));
Fmax_fft=max(find(sig_fft.Frequencies<=Fmax));

freq_range_fft=sig_fft.Frequencies(Fmin_fft:Fmax_fft);
PSD_fft=sig_fft.Data(Fmin_fft:Fmax_fft);
PSD_fft=PSD_fft/max(PSD_fft);
[v i]=max(PSD_fft);
DF_fft=freq_range_fft(i);

Freq_low_limit_fft=max(DF_fft-0.75,Fmin);
Freq_high_limit_fft=min(DF_fft+0.75,Fmax);

Freq_low_idx_fft=min(find(freq_range_fft>=Freq_low_limit_fft));
Freq_high_idx_fft=max(find(freq_range_fft<=Freq_high_limit_fft));
RI_fft=(sum(PSD_fft(1:min(Freq_high_idx_fft+1,length(PSD_fft))))-sum(PSD_fft(1:Freq_low_idx_fft-1)))/(sum(PSD_fft));

PSD_fft_interp=interp1(freq_range_fft,PSD_fft,fliplr(freq_range));

SVRPSD=PSD_fft_interp.*fliplr(SVR);
SVRPSD(isnan(SVRPSD))=0;
SVRPSD=SVRPSD/max(SVRPSD);
[v i]=max(SVRPSD);
freqrange_f=fliplr(freq_range);
DF_SVRPSD=freqrange_f(i);
Freq_low_limit_SVRPSD=max(DF_SVRPSD-0.75,Fmin);
Freq_high_limit_SVRPSD=min(DF_SVRPSD+0.75,Fmax);
Freq_low_idx_SVRPSD=max(find(freqrange_f<=Freq_low_limit_SVRPSD));
Freq_high_idx_SVRPSD=min(find(freqrange_f>=Freq_high_limit_SVRPSD));
DF_energy_SVRPSD=sum(SVRPSD(1:Freq_high_idx_SVRPSD+1))-sum(SVRPSD(1:Freq_low_idx_SVRPSD-1));
RI_SVRPSD=DF_energy_SVRPSD/sum(SVRPSD);

figure;
subplot('position',[0.1 0.66 0.8 0.25])
time_scale=(1:length(Electrogram))*1/Fs*1000;
plot(time_scale(end-500:end),Electrogram(end-500:end));
xlabel('Time [ms]')
%title(['X=' num2str(frame_x) ' Y=' num2str(frame_y) ' trans membrane'])
V=axis;
axis([time_scale(end-500) time_scale(end) V(3) V(4)]);
set(gca,'ytick',[]')

subplot('position',[0.1 0.1 0.25 0.4]);
plot(sig_fft.Frequencies(Fmin_fft:Fmax_fft),sig_fft.Data(Fmin_fft:Fmax_fft))
xlabel('Freq [Hz]')
ylabel('Power Density [V^2/Hz]')
title(['PSD, DF= ' num2str(DF_fft) 'Hz RI= ' num2str(RI_fft)]);
V=axis;
axis([Fmin Fmax V(3) V(4)]);
set(gca,'ytick',[]')
hold on
Hs=spectrum.welch;
Hs.SegmentLength=ceil(1000/DF_fft);
Hs.WindowName='hamming';
sig_fft_envelope=psd(Hs,Electrogram,'Fs',Fs);
Gain=max(sig_fft.Data(Fmin_fft:Fmax_fft))/max(sig_fft_envelope.Data(Fmin_fft:Fmax_fft));
plot(sig_fft_envelope.Frequencies(Fmin_fft:Fmax_fft),Gain*sig_fft_envelope.Data(Fmin_fft:Fmax_fft),'r')



subplot('position',[0.4 0.1 0.25 0.4]);
plot(freq_range,SVR)
ylabel('Normalized sigma1/sigma2')
xlabel('Freq [Hz]')
title(['SVR, DF= ' num2str(freq_range(DF)) 'Hz RI= ' num2str(RI)])
V=axis;
axis([Fmin Fmax V(3) V(4)]);
set(gca,'ytick',[]')

subplot('position',[0.7 0.1 0.25 0.4]);
plot(fliplr(freq_range),SVRPSD)
title(['SVR x PSD, DF= ' num2str(DF_SVRPSD) 'Hz RI= ' num2str(RI_SVRPSD)]);
xlabel('Freq [Hz]')
ylabel('Enhanced Power Density [V^2/Hz]')
V=axis;
axis([Fmin Fmax V(3) V(4)]);
set(gca,'ytick',[]')

