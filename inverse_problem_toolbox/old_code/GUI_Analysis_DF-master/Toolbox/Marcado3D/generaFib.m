
function [FV]=generaFib(FV,nodFibr,nombre)

    newCond= zeros(length(nodFibr),15);
    FV.conductancias(nodFibr,:)=newCond;
%     FV.vecinos(nodFibr,:)=newCond;
    
    for i=1:length(nodFibr)
        
        if mod(i,100)==0; disp([num2str(i/length(nodFibr)*100) ' %']); end 
        
        [fila,col]=find(FV.vecinos==nodFibr(i));
        for j=1:length(fila)
%              fil= FV.conductancias(fila(j),:);
%              fil(col(j))=[];
%              fil=[fil 0];
%              FV.conductancias(fila(j),:)=fil;

             FV.conductancias(fila(j),col(j))=0;
             
%              fil= FV.vecinos(fila(j),:);
%              fil(col(j))=[];
%              fil=[fil 0];
%              FV.vecinos(fila(j),:)=fil;
        end
        
    end

%     guardaConductancias(FV,nombre);
%     guardaVecinos(FV,nombre);
end