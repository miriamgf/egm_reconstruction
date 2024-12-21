function  mod=generaVecinos( mod )

mod.vecinos=zeros(length(mod.vertices),15);
mod.plantilla=zeros(length(mod.vertices),15);

for h=1:length(mod.vertices) % Todos los nodos.

        [f,c]=find(mod.faces == h); % averigua que faces tiene cada nodo. 
        vector1=[];
       %Para Triangulos
        for i=1:length(f)
            vector1=[vector1 mod.faces(f(i),:)];
        end
        vector1=unique(vector1);
        quita=find(vector1==h);  
        vector1=[vector1(1:quita-1) vector1(quita+1:end)];
  
        mod.numvec(h)=length(vector1);
        
        
        if length(vector1)<=15
           plant=[ones(1,length(vector1)) zeros(1,15-length(vector1))];
           vector1= [vector1 length(mod.vertices)*ones(1,15-length(vector1))];
        else
           plant=[ones(1,15)]; 
           vector1= vector1(1:15);
        end
        
        mod.vecinos(h,:)=vector1;
        mod.plantilla(h,:)=plant;
end
    mod.veins=mod.vecinos;
    mod.vecinos=mod.veins.*mod.plantilla;
end

