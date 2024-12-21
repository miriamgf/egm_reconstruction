%% Classical Tikhonov
function [x_hat, lambda_opt] = Classical_Tikhonov_Mod (A, AA, L, LL, y, xrt, lambda, opt_calc, n_iterations)
%Tikhonov global method reconstruction.
%
% This routine by Víctor Suárez Gutiérrez (victor.suarez.gutierrez@urjc.es)
% using L-curve approach to Tikhonov Regularization. IEEE trans on biomed eng.
% 2000. 47(9):1293-1296.
%
% Modified routine by Miguel Ángel Cámara (miguelangel.camara@urjc.es)
%
% The analytical solution of the inverse problem in terms of Tikhonov
% regularization is
%
% phi_E_hat = inv(A'*A + lambda*L'*L)*A'*phi_T
%
%
% To reduce computational requirements we define
% AA = A'*A and LL = L'*L;

% Preparamos el parpool para paralelizar.
if isempty(gcp('nocreate'))
    parpool;
end

if length(lambda)==1
    invA = (AA+lambda*LL)\A.';
    x_hat = invA*y;
    lambda_opt = lambda;
else
    % we look for the opt value of Lambda
    % using the l-curve
    if opt_calc==1  % L-Curve
        tic
        lambda=logspace(-1,-9,4);
        for iter=1:n_iterations
            fprintf('Classical Tikhonov method. Iteration %i\n',iter);
            
            magnitude_term = zeros(size(lambda));
            error_term =  zeros(size(lambda));
            
            parfor j=1:length(lambda)
                %fprintf('Classical Tikhonov method, computing estimation, L-Curve method (lambda %i/%i).\n', j,length(lambda));
                invA = (AA+lambda(j)*LL)\A.';
                x_hat = invA*y;
                error_term(j)    = norm(A*x_hat-y,'fro')^2;
                magnitude_term(j)= norm(L*x_hat,'fro')^2;
            end
            
            x = log10(error_term);
            z = log10(magnitude_term);
            dx = gradient(x,2); % uses h (x,h) as the spacing between points in each direction.
            dz = gradient(z,2);
            
            % optimal lambda vector:
            ddx = gradient(dx,2);
            ddz = gradient(dz,2);
            curva =(dx.*ddz-ddx.*dz)./((dx.^2+dz.^2).^(3/2));
            
            abscurva=abs(curva);
            [~,maxcurva]=max(abscurva(:));
            
            %maxcurva  = find(abs(curva) == max(abs(curva)), 1, 'first');
            
            lambda_opt = lambda(maxcurva);
            
            I=maxcurva;
            if I-1<=0
                lambda_inf=log10(lambda(I));
                lambda_sup=log10(lambda(I+1));
            elseif I+1>length(lambda)
                lambda_inf=log10(lambda(I-1));
                lambda_sup=log10(lambda(I));
            else
                lambda_inf=log10(lambda(I-1));
                lambda_sup=log10(lambda(I+1));
            end
            lambda=logspace(lambda_inf,lambda_sup,4);
            
            if lambda(1)-lambda(2)<=1e-12
                break;
            end
        end
        
        fprintf('Optimal parameters (Classical Tikhonov). Lambda=%i\n',lambda_opt);
        x_hat = ((AA+lambda_opt*LL)\A.')*y;
        toc
        
        if 0     % change 0 for 1 for L-curve representation.
            figure(),
            plot(x,z,'ko-');
            xlabel('log ||Ax-y||^2');
            ylabel('log ||Lx||^2');
            hold on;
            plot(x(maxcurva), z(maxcurva),'*r');
            hold on;
            plot(x(maxcurva+2), z(maxcurva+2),'*g');
            hold on;
            plot(x(maxcurva+1), z(maxcurva+1),'*');
            title('L-Curve function for Tikhonov method'),
            fprintf('Optimal lambda value (L-Curve method): %i.\n',lambda_opt);
        end
        
    elseif opt_calc==2  % CRESO
        for i=1:length(lambda)
            fprintf('Classical Tikhonov method, computing estimation, CRESO method (lambda %i/%i).\n', i,length(lambda));
            x_hat = (AA+lambda(i)*LL)\(A'*y);
            magnitude_term = lambda(i)^2*norm(L*x_hat,'fro')^2;
            error_term= norm(A*x_hat-y,'fro')^2;
            B(i)= magnitude_term-error_term;
        end
        
        C_lambda = gradient(B);
        C_lambda_interp = interp1(lambda,C_lambda,0:0.001:max(lambda),'spline');
        [~,locs] = findpeaks(C_lambda_interp,0:0.001:max(lambda));
        lambda_opt=locs(1);
        invA = (AA+lambda_opt*LL)\A.';
        x_hat = invA*y;
        
        if 0    % change 0 for 1 for CRESO representation.
            figure(),
            subplot(211),plot(lambda,C_lambda),title('C(\lambda) function'),
            xlabel('\lambda values');
            ylabel('C(\lambda)');
            xlim([min(lambda) max(lambda)]),
            subplot(212),plot(0:0.001:max(lambda),C_lambda_interp),title('C(\lambda) function (interpolated)'),
            xlabel('\lambda values');
            ylabel('C(\lambda)');
            xlim([min(lambda) max(lambda)]),
            fprintf('Optimal lambda value (CRESO method): %i.\n',lambda_opt);
        end
    end
end
end