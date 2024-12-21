function  guardaRem( Rem,nombreProyecto )
nombreArchivo=[nombreProyecto '_rem.txt'];
fid=fopen(nombreArchivo,'w');

for i=1:length(Rem)             %% filas

    fwrite(fid,num2str (Rem(i)));
    fwrite(fid,char(10));
end
fclose(fid);

end


