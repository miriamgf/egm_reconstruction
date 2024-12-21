
function [P2nd,P3rd,P4th,P5th,f2nd,f3rd,f4th,f5th] = higherHarmonic(Pegm,f,f0,condiciones)
%
%Function that computes the higherHarmonics from the Pegm up to the 5th
%harmonic
%
% [P2nd,P3rd,P4th,P5th,f2nd,f3rd,f4th,f5th] = higherHarmonic(Pegm,f,f0,condiciones)
%
%It uses condiciones struct, wich has the following fields:
%condicions has the following fields:
%
%     condiciones.w='hamming';
%     condiciones.L=256;
%     condiciones.R=0.5;
%     condiciones.nfft=4096; %1024;
%     condiciones.fl1=30;
%     condiciones.fh1=50;
%     condiciones.fh2=15; % 40
%     condiciones.diff=1;
%     condiciones.dfn=1/3;
%     condiciones.ro=.75;
%     condiciones.T=2;
%     condiciones.mideltat=0.1;
%     condiciones.dh_hist=0.2;
%     condiciones.oi=1;
%
% Entrada: DEP, eje de frecuencias, f0
% Salida: P del segundo y tercer arm�nico
%
%df_toolbox
radio = f0*condiciones.dfn;

[kk,ini] = min(abs(f-(2*f0-radio)));
[kk,fin] = min(abs(f-(2*f0+radio)));
[P2nd,ind] = max(Pegm(ini:fin));
f2nd = f(ind+ini-1);

[kk,ini] = min(abs(f-(3*f0-radio)));
[kk,fin] = min(abs(f-(3*f0+radio)));
[P3rd,ind] = max(Pegm(ini:fin));
f3rd = f(ind+ini-1);

[kk,ini] = min(abs(f-(4*f0-radio)));
[kk,fin] = min(abs(f-(4*f0+radio)));
[P4th,ind] = max(Pegm(ini:fin));
f4th = f(ind+ini-1);

[kk,ini] = min(abs(f-(5*f0-radio)));
[kk,fin] = min(abs(f-(5*f0+radio)));
[P5th,ind] = max(Pegm(ini:fin));
f5th = f(ind+ini-1);