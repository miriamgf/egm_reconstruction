function [RI, OI, metrics] = RI_OI_EGMs (EGM, fs, DF_egm)

[N,~] = size(EGM);
L = 1024;
j=1;
while L > length(EGM(1,:))
    L = 2^nextpow2(N-j);
    j=j+1;
end

for i=1:size(EGM,1)
    egm=EGM(i,:);
    egm = egm-mean(egm);
    [Pxx,fxx] = pwelch(egm,hamming(L),[],L,fs);
    [bw,~,~] = bandWidth(Pxx,fxx,DF_egm(i));
    [OI(i),~,~] = organizationIndex(Pxx,fxx,DF_egm(i),bw);
    [RI(i),~,~,~] = riHC(egm,DF_egm(i),Pxx,fxx);
end
metrics.mRI=mean(RI);
metrics.stdRI=std(RI);

metrics.mOI=mean(OI);
metrics.stdOI=std(OI);
end
