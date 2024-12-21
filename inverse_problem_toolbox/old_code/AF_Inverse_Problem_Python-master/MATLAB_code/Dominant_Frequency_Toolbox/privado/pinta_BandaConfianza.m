function [f] = pinta_BandaConfianza(media,des_stan,color,t,x)

if nargin<4
   t = 1;
end

%Funcion que pinta bandas de confianza
if color == 1;
   c = [132/255 132/255 132/255];
elseif color == 0
   c = [239/255 233/255 233/255]; 
end


superior = media+des_stan;

%df = des_stan - media;

inferior = media-des_stan;
if nargin < 5
x = 1:length(media);
end

f = fill([x fliplr(x)],[superior fliplr(inferior)],c);

set(f,'FaceAlpha',t,'EdgeAlpha',t);%set edge color

hold on
plot(x,media,'r','LineWidth',1)
