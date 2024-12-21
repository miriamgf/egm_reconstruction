%[Psvd,fsvd] = mysvd(Electrogram,Fs);

%

Fmin=3;
Fmax=30;

N_min=round(Fs/Fmax);
N_max=round(Fs/Fmin);

index=1;
min_std=10000;
for window_length=N_min:N_max,
    disp(['Length ' num2str(window_length) ' of ' num2str(N_max)])
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
SVR=SVR*min_std/std(Electrogram);
SVR=SVR/max(SVR);
freq_range=Fs./[N_min:N_max];   

[Vt DF]=max(SVR);
Freq_low_limit=max(freq_range(DF)-0.75,Fmin);
Freq_high_limit=min(freq_range(DF)+0.75,Fmax);
Freq_high_idx=min(find(freq_range<=Freq_low_limit));
Freq_low_idx=max(find(freq_range>=Freq_high_limit));
DF_energy=sum(SVR(1:Freq_high_idx+1))-sum(SVR(1:Freq_low_idx-1));
Total_energy=sum(SVR);
RI=DF_energy/Total_energy;
keyboard


