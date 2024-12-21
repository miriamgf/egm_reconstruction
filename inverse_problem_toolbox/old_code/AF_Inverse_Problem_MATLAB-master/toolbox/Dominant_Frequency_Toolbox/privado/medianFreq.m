function fm = medianFreq(Pegm,f)
%
%Function that computes the median frequency
%
%fm = medianFreq(Pegm,f)
%
%fd_toolbox

% Frecuencia mediana
fm = sum(Pegm.*f)/sum(Pegm);
