function  guardaVecinos( mod,nombreProyecto )
nombreArchivo=[nombreProyecto '_vec.txt'];
fid=fopen(nombreArchivo,'w');

for i=1:length(mod.vecinos)             %% filas
    if (isempty(find(mod.vecinos(i,:)==0)))
        num = 14;
    else
        num=min(find(mod.vecinos(i,:)==0))-2 ;
    end
    
    for j=1:num       
       fwrite(fid,num2str (mod.vecinos(i,j)));
       fwrite(fid,[' ']);
    end
    fwrite(fid,num2str (mod.vecinos(i,j+1)));
    fwrite(fid,char(10));
end
fclose(fid);

end

