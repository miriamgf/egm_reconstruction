function res = sampen_egms(EGM,fs)
% Function that process EGMs to calculate Sample Entropy

%% Downsampling of input to 50Hz.
% dec_rate=fs/50;
% for i=1:size(EGM,1)
%     egm_resampled(i,:) = decimate(EGM(i,:),dec_rate);
% end

% Preparamos el parpool para paralelizar.
if isempty(gcp('nocreate'))
    parpool;
end

% Paralelizo con bloques de 100 en 100 y evitar memory overflow.
index_step=1:100:size(EGM,1);
if isempty(find(index_step==size(EGM,1), 1))
    index_step(end+1)=size(EGM,1);
end

% Inicializo la matriz de SampEn
res=zeros(1,size(EGM,1));

% Calculo el SampEn por partes
for i=1:length(index_step)-1
    parfor j=index_step(i):index_step(i+1)
        fprintf('SampEn. Node %d \n',j);
        r = 0.2*std(EGM(j,:));
        m = 3;
        res(j) = SampEn(EGM(j,:),r,m); 
    end
end