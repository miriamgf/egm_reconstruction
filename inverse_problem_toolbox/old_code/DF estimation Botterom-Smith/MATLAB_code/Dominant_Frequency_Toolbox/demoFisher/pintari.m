function pintari

%

figure(1)
load resultDemowithD
subplot(2,2,1),
surf(D,f0,RItot), axis tight, shading interp
ejes = axis; ejes(5:6) = [0 1]; axis(ejes)
xlabel('D (ms)');  
ylabel('f_0 (Hz)');  zlabel('RI')

load resultDemowithDrationcorrection
subplot(2,2,2),
surf(D,f0,RItot), axis tight, shading interp
ejes = axis;
ejes(5:6) = [0 1.3]; axis(ejes)
xlabel('D (ms)')
ylabel('f_0 (Hz)')
zlabel('RI')
%title('Theoretical ratio correction')

load resultDemowithDHC
subplot(2,2,3),
surf(D,f0,RItot), axis tight, shading interp
ejes = axis;
ejes(5:6) = [0 1]; axis(ejes)
xlabel('D (ms)')
ylabel('f_0 (Hz)')
zlabel('RI')
%title('Harmonic correction')

load resultDemoSingleHarmonic;
subplot(2,2,4),
surf(D,f0,RItot), axis tight, shading interp
ejes = axis;
ejes(5:6) = [0 1]; axis(ejes)
xlabel('D (ms)')
ylabel('f_0 (Hz)')
zlabel('RI')
%title('Single Harmonic correction')
