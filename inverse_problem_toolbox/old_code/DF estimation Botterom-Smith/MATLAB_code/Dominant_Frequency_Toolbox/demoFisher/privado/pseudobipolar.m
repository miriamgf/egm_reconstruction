function [egm,tr1,tr2] = pseudobipolar;

%

fs = 1000;
ts = 1/fs;
mytau = 0.005; % 5 ms

tr1 = 2*triang(2*floor(mytau/ts)+1);
tr2 = -triang(4*floor(mytau/ts)+1);

egm = tr2;
n = floor(length(egm)/2);
p = ceil(length(tr1)/2);
egm(n-p+2:n+p) = egm(n-p+2:n+p) + tr1;