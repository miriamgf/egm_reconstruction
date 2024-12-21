function [xo,vo] = definelinea2puntos(FV,Mod_XL, zona)
t2=FV.faces;
p=FV.vertices;

patch('Faces',t2,'Vertices',p,'facecolor','c','edgecolor','k') 
hold on;    axis equal;
light('Style','infinite');
 if nargin>2
     hold on; plot3(Mod_XL.vertices(zona,1),Mod_XL.vertices(zona,2),Mod_XL.vertices(zona,3),'.r')
 end
 
pause;    [az,el]=view;
CamViAn=get(gca,'CameraViewAngle');
CamPos=get(gca,'CameraPosition');
CamTar=get(gca,'CameraTarget');

     
faces_Linea=[];
Nodos_Linea=[];


punts=[];
punts2=[];
% Recta Lesió
for i=1:2
    ginput(1);
    punts(i)=aundreucallbackClickA3DPoint(FV.vertices');
    close all
    %Dibuja la superficie!
    patch('Faces',t2,'Vertices',p,'facecolor','c','edgecolor','k') 

    hold on
    axis equal;
    light('Style','infinite');
    view(az,el)
    set(gca,'CameraViewAngle',CamViAn);
    set(gca,'CameraPosition',CamPos);
    set(gca,'CameraTarget',CamTar);
 
    plot3(FV.vertices(punts,1),FV.vertices(punts,2),FV.vertices(punts,3),'or')

end
%     Dibuja recta
    t=-0.1:0.05:1.1;
    xo=FV.vertices(punts(1),:);
    vo=FV.vertices(punts(2),:)-FV.vertices(punts(1),:);
    rect=[];
    for i=1:3
        rect(:,i)= xo(i)+vo(i).*t;
    end
    plot3(rect(:,1),rect(:,2),rect(:,3),'ob')

end