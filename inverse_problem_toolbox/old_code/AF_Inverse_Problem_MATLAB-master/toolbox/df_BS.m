function [xDF,z_egm] = df_BS (EGM, fs)

% Preparamos el parpool para paralelizar.
if isempty(gcp('nocreate'))
    parpool;
end

% Inicializamos la matriz de salida.
[N,T] = size(EGM);
xDF = zeros(N,1);
z_egm = zeros(N,T);

% Paralelizo con bloques de 100 en 100 y evitar memory overflow.
index_step=1:100:size(EGM,1);
if isempty(find(index_step==size(EGM,1), 1))
    index_step(end+1)=size(EGM,1);
end

% Calculo la DF por partes
for i=1:length(index_step)-1
    parfor j=index_step(i):index_step(i+1)
        fprintf('DF (Botterom-Smith). Node %d \n',j);
        %[xDF(j),~,~,~,~,~] = df_Ng(EGM(j,:),fs,0);
        [z_egm(j,:),~,~, xDF(j)] = botterom_smith_df(EGM(j,:),fs,10,50,20);
    end
end