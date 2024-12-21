function demoRIwithD;

%

f0 = 3:.1:15;
D = 0:5:45; % ms

n = length(f0);
m = length(D);

RItot = zeros(n,m);

n_realizations = 1000;
for k = 1:n_realizations
    disp([num2str(k),' of ',num2str(n_realizations)]);
    RI = zeros(n,m);
    for i=1:n
        for j=1:m
            [egm,t] = pseudoegm(f0(i),D(j));
            RI(i,j) = myri(abs(egm),f0(i));
        end
    end
    
    RItot = RItot + RI;
end
RItot = RItot/n_realizations;

save resultDemowithD 

figure(3),clf
surf(D,f0,RItot), axis tight, shading interp
ejes = axis;
ejes(5:6) = [0 1];
axis(ejes)
xlabel('Uncertainty D (ms)')
ylabel('f0 (Hz)')
zlabel('RI')
