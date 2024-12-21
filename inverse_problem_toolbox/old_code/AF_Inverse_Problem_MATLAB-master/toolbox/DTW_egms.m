function [MatchingCost, metrics_MC]=DTW_egms(x_ref,xhat_test,selected_nodes)

% Preparamos el parpool para paralelizar.
if isempty(gcp('nocreate'))
    parpool;
end

% Paralelizo con bloques de 100 en 100 y evitar memory overflow.
index_step=1:100:2048;
if isempty(find(index_step==size(x_ref,1), 1))
    index_step(end+1)=size(x_ref,1);
end

% Inicializo la matriz de MatchingCost
MatchingCost=zeros(1,size(x_ref,1));

% Calculo el MatchingCost por partes
for i=1:length(index_step)-1
    parfor j=index_step(i):index_step(i+1)
        fprintf('DTW. Node %d \n',j);
        [MatchingCost(j),~,~]=DTWSakoe(x_ref(j,:),xhat_test(j,:),0);
    end
end

% Métricas DTW
metrics_MC.mDTW=mean(MatchingCost(selected_nodes));
metrics_MC.stdDTW=std(MatchingCost(selected_nodes));