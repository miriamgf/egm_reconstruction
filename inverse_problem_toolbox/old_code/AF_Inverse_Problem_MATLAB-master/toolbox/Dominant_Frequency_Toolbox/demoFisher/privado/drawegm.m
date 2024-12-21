
function drawegm(egm,t);

%

fs = 1000;
[P,f] = psd(egm-mean(egm),8*length(egm),fs);

%keyboard
subplot(211)
plot(t,egm), axis tight
xlabel('t (secs)'), ylabel('EGM(t)');

subplot(212)
plot(f,P), axis tight
xlabel('f (Hz)'), ylabel('PSD(f)');
hold on
ejes = axis;
ejes(2) = 150;
axis(ejes);
m = ejes(4)/2;
plot([ejes(1), 3, 3, 15, 15, ejes(2)],[0 0 m m 0 0],':r')
hold off
