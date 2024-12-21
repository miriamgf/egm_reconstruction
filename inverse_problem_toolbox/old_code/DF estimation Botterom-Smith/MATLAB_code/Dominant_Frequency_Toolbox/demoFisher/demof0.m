function demof0;

%

f0 = 3:.5:15;
D = 0:5:45;

n = length(f0);
m = length(D);
n_realizations = 1000;

f0tot1 = zeros(n,m);    % Just the spectral maximum
f0tot2 = zeros(n,m);    % With zero padding
F0tot1 = zeros(n,m,n_realizations);
F0tot2 = zeros(n,m,n_realizations);

for k = 1:n_realizations
    disp([num2str(k),' of ',num2str(n_realizations)]);
    RI = zeros(n,m);
    for i=1:n
        for j=1:m
            [egm,t] = pseudoegm(7,f0(i),D(j));
            egm = egm(1:1024);
            [f0tot1(i,j),f0tot2(i,j)] = myF0(abs(egm));
        end
    end
    F0tot1(:,:,k) = f0tot1;
    F0tot2(:,:,k) = f0tot2;
end
F0tot1m = mean(F0tot1,3);
F0tot2m = mean(F0tot2,3);
F0tot1std = std(F0tot1,[],3);
F0tot2std = std(F0tot2,[],3);

save resultF0

figure(1),clf
subplot(221)
surf(D,f0,F0tot1m), axis tight, shading interp
xlabel('Uncertainty D (ms)')
ylabel('true f0 (Hz)')
zlabel('detected f0 (Hz) ')
subplot(222)
surf(D,f0,F0tot2m), axis tight, shading interp
xlabel('Uncertainty D (ms)')
ylabel('true f0 (Hz)')
zlabel('detected f0 (Hz) ')
subplot(223)
surf(D,f0,F0tot1std), axis tight, shading interp
xlabel('Uncertainty D (ms)')
ylabel('true f0 (Hz)')
zlabel('detected f0 (Hz) ')
subplot(224)
surf(D,f0,F0tot2std), axis tight, shading interp
xlabel('Uncertainty D (ms)')
ylabel('true f0 (Hz)')
zlabel('detected f0 (Hz) ')
