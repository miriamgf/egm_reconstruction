function  guardaConductancias( mod,nombreProyecto )
nombreArchivo=[nombreProyecto '_cond.txt'];
fid=fopen(nombreArchivo,'w');

for i=1:length(mod.conductancias)             %% filas
%     if (isempty(find(mod.vecinos(i,:)==0)))
%         num = 14;
%     else
%         num=min(find(mod.vecinos(i,:)==0))-2 ;
%     end
    
    for j=1:14   
       fwrite(fid,num2str (mod.conductancias(i,j)));
       fwrite(fid,[' ']);
    end
    fwrite(fid,num2str (mod.conductancias(i,j+1)));
    fwrite(fid,char(10));
end
fclose(fid);

end


