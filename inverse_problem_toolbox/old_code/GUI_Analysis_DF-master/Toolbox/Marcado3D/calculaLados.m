
function [FV]=calculaLados(FV)

FV.lados=zeros(length(FV.vertices),15);

for i=1:15
    
    difx=(FV.vertices(:,1)-FV.vertices(FV.veins(:,i),1));
    dify=(FV.vertices(:,2)-FV.vertices(FV.veins(:,i),2));
    difz=(FV.vertices(:,3)-FV.vertices(FV.veins(:,i),3));
    FV.lados(:,i)=sqrt(difx.^2+dify.^2+difz.^2);

end 

FV.lados=FV.lados.*FV.plantilla;
end
