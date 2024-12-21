function [bw,flow,fup] = bandWidth(P,f,f0,level)
%Function that computes the bandwith of the components indicated by the
%parameter f0, and using the level value from level parameter.
%
%[bw,flow,fup] = bandWidth(P,f,f0,level)

[kk,ind0] = min(abs(f-f0));
indinf = ind0;
if nargin==3,
    level = .75;
end
while 1
    indinf = indinf-1;
    if indinf==1,break,end;
    if P(indinf)<level*P(ind0)
        break
    end
end
indsup = ind0;
while 1
    indsup = indsup+1;
    if indsup==length(P),break,end;
    if P(indsup)<level*P(ind0)
        break
    end
end
flow = f(indinf);  fup = f(indsup);
bw = fup - flow;