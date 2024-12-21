% function [facesout,nodos_out]=pincel(t2,p,tam)
function [nodos_out]=pincel(FV,tam,figure_used)
%elijo el punto de inicio de la propagación
% pointCloudIndex=aundreucallbackClickA3DPoint(p');
pointCloudIndex=aundreucallbackClickA3DPoint(FV.vertices',figure_used,[]);


p1=pointCloudIndex;


f=[];
nodos=p1;
nodos_prev=[];
% nodos_1=[];
facesout=[];

%while (reply=='Y' && (length(nodos)~=length(nodos_1)))
if(tam>0)
for i=1:tam    
%     nodos_1=nodos;
    jep=length(nodos);
%     caras=[];
    
    for j=1:jep
        nodos_prev=[nodos_prev,FV.vecinos(nodos(j),:)]; % averigua que faces estan en los nodos.
    end
    nodos=sort(unique(nodos_prev));
    nodos(1)=[];
    
%     for j=1:jep
%         [caras2,c]=find(t2 == nodos(j)); % averigua que faces estan en los nodos.
%         caras=[caras;caras2];
%     end
%     
%     facesout=[facesout;caras];
%     facesout=sort(unique(facesout));
    
%     for j=1:length(caras)
%          nodos=[nodos t2(caras(j),:)];
%     end
%         nodos=unique(nodos);
end
end
        nodos_out=nodos;
  
end

