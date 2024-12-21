function demoRIwithD;

%

f0 = 3:.1:30;
D = 0;
A = .3:.05:1; % ms

n = length(f0);
m = length(A);

RItot = zeros(n,m);

n_realizations = 10;
for k = 1:n_realizations
    disp([num2str(k),' of ',num2str(n_realizations)]);
    RI = zeros(n,m);
    for i=1:n
        for j=1:m
            [egm,t] = pseudoegm(f0(i),D,A(j));
            RI(i,j) = myriSHC(abs(egm),f0(i));
        end
    end
    
    RItot = RItot + RI;
end
RItot = RItot/n_realizations;

save resultDemoSingleHarmonicA
% load resultDemoSingleHarmonicA
figure(6),clf
surf(A,f0,RItot), axis tight, shading interp
ejes = axis;
ejes(5:6) = [0 1]; axis(ejes)
xlabel('Uncertainty A')
ylabel('f0 (Hz)')
zlabel('RI')
