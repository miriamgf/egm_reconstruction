function [MTransfer]=BEM_TransfMat(Modelo, cond)

% Reconstruccion de Modelo
[f,c]=size(cond); % fila1 cond, fila2 numcapa externa
Nlmod=length(Modelo);

 for i=1:length(Modelo)
 Modelo(i).node=Modelo(i).vertices';
 Modelo(i).face=Modelo(i).faces';
 end

if f==1
    for i=1:c-1
        cond(2,i)=cond(1, i+1);
    end
end


display('Conductividades:')
display(['IN : ' num2str(cond(1,:)) ])
display(['OUT: ' num2str(cond(2,:)) ])
display('----------')
pause(2)

% Añado conductividades al nuevo modelo.
for i=1:Nlmod
    Mod_.surface{i}=Modelo(i);
    Mod_.surface{i}.sigma=[ cond(1,i) cond(2, i) ];
end

% invierto las capas. Ahora la ultima capa es el corazon.
% for i=1:Nlmod
%     Modelo2.surface{i}=Mod.surface{Nlmod-i+1};
% end

MTransfer= bemMatrixPP(Mod_); % Aqui se procesan las matrices de influencia de potenciales y de gradiente.

% Utorso=MTransfer*Uh;

end