function  guardaStim(S1,S2,nombreProyecto)
nombreArchivo=[nombreProyecto '_S1.txt'];
fid=fopen(nombreArchivo,'w');

for i=1:length(S1)             %% filas
    fwrite(fid,num2str (S1(i)));
    fwrite(fid,char(10));
end
fclose(fid);

nombreArchivo=[nombreProyecto '_S2.txt'];
fid=fopen(nombreArchivo,'w');

for i=1:length(S2)             %% filas
    fwrite(fid,num2str (S2(i)));
    fwrite(fid,char(10));
end
fclose(fid);
end

