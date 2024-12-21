
f0 = 4:1:26;
n = length(f0);
Fs = 1000;

egm = pseudoegm(5,0,.8);
[freq_range,SVR] = mysvd(egm,Fs);
m = length(freq_range);

aux1 = zeros(n,m);

for nn = 1:n
    disp([num2str(nn), ' of ',num2str(n)]);
    egm = pseudoegm(f0(nn),15,.8);
    [freq_range,SVR] = mysvd(abs(egm),Fs);
    aux1(nn,:) = SVR;   % SVD
    drawnow
end

figure(1),clf
surf(freq_range,f0,aux1)
xlabel('f axis'); ylabel('component at f0')
axis tight; shading interp

