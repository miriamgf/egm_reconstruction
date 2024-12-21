%% Constrained Tikhonov (3-terms)
function [x_hat, lambda_opt, metrics] = Constrained_Tikhonov_L3 (A, AA, LL, y, x, lambda1, lambda2, D, lambda3)
%Tikhonov global method reconstruction.
%
% This routine by Víctor Suárez Gutiérrez (victor.suarez.gutierrez@urjc.es)
% using L-curve approach to Tikhonov Regularization. IEEE trans on biomed eng.
% 2000. 47(9):1293-1296.
%
% The analytical solution of the inverse problem in terms of Tikhonov
% regularization is
%
% phi_E_hat = inv(A'*A + lambda*L'*L)*A'*phi_T
%
%
% To reduce computational requirements we define
% AA = A'*A and LL = L'*L;

if length(lambda1)==length(lambda2)==length(lambda3)==1
    inv_term=(AA+lambda1*LL+lambda2*(D'*D)+lambda3*eye(length(D)));
    for t=1:size(x,2)
        if(t>1)
            x_hat(:,t) = inv_term\((A'*y(:,t))+(lambda2*D*x(:,t))+lambda3*x_hat(:,t-1));
        else
            x_hat(:,t) = inv_term\((A'*y(:,t))+(lambda2*D*x(:,t)));
        end
    end
else
    lambda_opt{1}=lambda1;
    lambda_opt{2}=lambda2;
    lambda_opt{3}=lambda3;
    % Saco múltiples estimaciones de x_hat para evaluar parámetros.
    for i=1:length(lambda1)
        fprintf('Constrained Tikhonov method, computing estimation (lambda1 %i/%i).\n', i,length(lambda1));
        for j=1:length(lambda2)
            for k=1:length(lambda3)
                inv_term=(AA+lambda1(i)*LL+lambda2(j)*(D'*D)+lambda3(k)*eye(length(D)));
                for t=1:size(x,2)
                    if(t>1)
                        x_hat(:,t) = inv_term\((A'*y(:,t))+(lambda2(j)*D*x(:,t))+lambda3(k)*x_hat(:,t-1));
                    else
                        x_hat(:,t) = inv_term\((A'*y(:,t))+(lambda2(j)*D*x(:,t)));
                    end
                end
                metrics{i,j,k} = timemetrics_mod (x(1:2039,:)', x_hat(1:2039,:)');
            end
        end
    end
end
end