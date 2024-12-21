function [res] = SampEn(X,r,m)
%""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
% PROPÓSITO:
% Función que calcula la estimación de la entropía muestral SampEn de una
% señal.
% FORMA DE USO:
%	[res] = SampEn(X,r,m)
%
% ARGUMENTOS...
% ...DE ENTRADA:
%       .-X ---> señal de la que se pretende estimar la SampEn.
%       .-r ---> filtro de ruido.
%       .-m ---> longitud de los vectores a ser comparados.
% ...DE  SALIDA:
%       .-sampen  ---> valor de la SampEn calculada.
%
% COMENTARIOS:
%Lake, D. E., J. S. Richman, et al. (2002).
%"Sample entropy anallysis of neonatal heart rate variability."
%Am. J. Physiol. Heart. Circ. Physiol. 283: 789-797.
%
%
%"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
%
% FECHA: 27-09-2007
%
% VERSION 2.0
%
%""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
%
%

N=length(X);

%convierte en vector columna
X=X(:);


B_m_i = zeros(1,N-m);
A_m_i = zeros(1,N-m);

%se crea una matriz que contendra todos los vectores a ser comparados
%entre si.
for n=1:2
    M=zeros(N-m,m+n-1);
    [f,c]= size(M);
    for i=1:f
        M(i,:)=X(i:i+m+n-2);
    end
    %calculo de la medida de correlacion.
    
    for i=1:f
        %se construye una matriz cuyas filas contien el vector a ser comparado con el resto de los
        %vectores, replica la matriz dada con dimensiones fx1.
        Mi=repmat(M(i,:),f,1);
        %para cada fila de la matrix el maximo entre las columnas de la matriz de
        %diferencias
        dist = max(abs(Mi-M),[],2);
        %para eliminar las autocomparaciones
        dist(i,:)=[];
        if n ==1
            B_m_i(i)=length(find(dist<=r))/(N-m-1);
        else
            A_m_i(i)=length(find(dist<=r))/(N-m-1);
        end
    end
end
B_m = mean(B_m_i);
A_m = mean(A_m_i);
res= log(B_m) - log(A_m);