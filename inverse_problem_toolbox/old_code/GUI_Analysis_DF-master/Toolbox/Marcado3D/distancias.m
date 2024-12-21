function [c,comp]=distancias(FV,Ne,distmax)
comp=0;   
tam=size(Ne);
c=Ne; %inicializa c
    for i=1:tam(2)
       a=Ne(1,i);
       a=a{1};
       long=0;
       for j=1:length(a)
        
        difx=(FV.vertices(i,1)-FV.vertices(a(j),1));
        dify=(FV.vertices(i,2)-FV.vertices(a(j),2));
        difz=(FV.vertices(i,3)-FV.vertices(a(j),3));
        dist=sqrt(difx.^2+dify.^2+difz.^2);
        comp=comp|dist>distmax;
        long(j)=round(dist); %se redondea a entero 
       end 
    c{i}=long;
    end

end