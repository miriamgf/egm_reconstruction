function [oi,ejexplot,ejeyplot] = organizationIndex(Pegm,f,fdom,ancho)
%
%Function that computes the organization index (or it is "ri") CHECK THAT!
%
%[oi,ejexplot,ejeyplot] = organizationIndex(Pegm,f,ancho)
%
%TO_DO CHECK THAT THE COMPUTATION IS DOING ACCORDINGLY TO THE PAPERS
%
%df_toolbox
%

%Pegm=fvpar.Pegm;
%f=fvpar.f;
radio=ancho/2;
ew=0;


[kk,ini] = min(abs(f-(fdom-radio)));
[kk,fin] = min(abs(f-(fdom+radio)));
ew=sum(Pegm(ini:fin))+ew;
ejexplot = [f(ini:fin)', NaN]; 
ejeyplot = [Pegm(ini:fin)', NaN];


%[kk,ini] = min(abs(f-(fvpar.f2nd-radio)));
%[kk,fin] = min(abs(f-(fvpar.f2nd+radio)));
%ew=sum(Pegm(ini:fin))+ew;
%ejexplot = [ejexplot, f(ini:fin)', NaN]; 
%ejeyplot = [ejeyplot, Pegm(ini:fin)', NaN];

%[kk,ini] = min(abs(f-(fvpar.f3rd-radio)));
%[kk,fin] = min(abs(f-(fvpar.f3rd+radio)));
%ew=sum(Pegm(ini:fin))+ew;
%ejexplot = [ejexplot, f(ini:fin)', NaN]; 
%ejeyplot = [ejeyplot, Pegm(ini:fin)', NaN];

%[kk,ini] = min(abs(f-(fvpar.f4th-radio)));
%[kk,fin] = min(abs(f-(fvpar.f4th+radio)));
%ew=sum(Pegm(ini:fin))+ew;
%ejexplot = [ejexplot, f(ini:fin)', NaN]; 
%ejeyplot = [ejeyplot, Pegm(ini:fin)', NaN];

% [kk,ini] = min(abs(f-(fvpar.f5th-radio)));
% [kk,fin] = min(abs(f-(fvpar.f5th+radio)));
% ew=sum(Pegm(ini:fin))+ew;
% ejexplot = [ejexplot, f(ini:fin)']; 
% ejeyplot = [ejeyplot, Pegm(ini:fin)'];

%[kk,ini] = min(abs(f-2.5));
[kk,ini] = min(abs(f-1));
%[kk,fin] = min(abs(f-(fvpar.f5th-radio)));
[kk,fin] = min(abs(f-35));

oi=ew/sum(Pegm(ini:fin));