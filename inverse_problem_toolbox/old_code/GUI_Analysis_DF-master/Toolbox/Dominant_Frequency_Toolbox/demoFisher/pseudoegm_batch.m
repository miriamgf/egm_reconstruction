frequency=3:0.2:28;
%d_pert=0:5:45;
% a_pert=1:-0.1:0.5;
a_pert=0.7;
d_pert=35;
for repetition=1:1,
    for f=1:length(frequency),
        for d=1:length(d_pert),
            for a=1:length(a_pert),
                disp(['Repetition ' num2str(repetition) ' Frequency ' num2str(frequency(f)) ' D ' num2str(d_pert(d)) ' A ' num2str(a_pert(a))]);
                Electrogram=pseudoegm(frequency(f),d_pert(d),a_pert(a));
                Electrogram=awgn(Electrogram(1:1024),100);
                SVD_pseudoegm
                pause
                close
%                 DF_PSD(repetition,f,d,a)=DF_fft;
%                 RI_PSD(repetition,f,d,a)=RI_fft;
%                 DF_SVR(repetition,f,d,a)=DF;
%                 RI_SVR(repetition,f,d,a)=RI;
%                 DF_PSDSVR(repetition,f,d,a)=DF_SVRPSD;
%                 RI_PSDSVR(repetition,f,d,a)=RI_SVRPSD;
            end
        end
    end
end