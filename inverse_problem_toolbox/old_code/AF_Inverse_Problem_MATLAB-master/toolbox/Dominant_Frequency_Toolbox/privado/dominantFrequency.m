function [Pfd,fd] = dominantFrequency(Pegm,f)
%
%Function that computes the fd from the psd estimated directly from the egm
%without any preprocessing.
%
%[Pfd,fd] = dominantFrequency(Pegm,f)
%
%df_toolbox

[Pfd,indfdom] = max(Pegm);
fd = f(indfdom);