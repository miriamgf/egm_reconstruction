%% Constrained Tikhonov (2-terms)
function [x_hat, lambda_opt] = Constrained_Tikhonov_L2_Mod (A, AA, L, LL, y, x, lambda1, lambda2, D, x_cons, opt_calc, n_iterations)
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

% Preparamos el parpool para paralelizar.
if isempty(gcp('nocreate'))
    parpool;
end

if length(lambda1)==1 && length(lambda2)==1
    x_hat = (AA+lambda1*LL+lambda2*(D'*D))\((A'*y)+(lambda2*D*x_cons));
    lambda_opt{1}=lambda1;
    lambda_opt{2}=lambda2;
else
    if opt_calc==1  % L-Hypersurface
        tic
        lambda1=logspace(-1,-9,4);
        lambda2=logspace(-1,-9,4);
        for iter=1:n_iterations
            fprintf('Constrained Tikhonov method. Iteration %i\n',iter);
            
            for i=1:length(lambda1)
                %fprintf('Constrained Tikhonov method, computing estimation, L-Hypersurface method (lambda1 %i/%i).\n', i,length(lambda1));
                parfor j=1:length(lambda2)
                    x_hat = (AA+lambda1(i)*LL+lambda2(j)*(D'*D))\((A'*y)+(lambda2(j)*D*x_cons));
                    magnitude_term (i,j) = norm(L*x_hat,'fro')^2;
                    error_term_1 (i,j)= norm(A*x_hat-y,'fro')^2;
                    error_term_2 (i,j) = norm(D*x_hat-x_cons,'fro')^2;
                end
            end
            
            xterm = log10(error_term_1);
            yterm = log10(error_term_2);
            zterm = log10(magnitude_term);
            
            [dxx,dxy] = gradient(xterm,2); % uses h (x,h) as the spacing between points in each direction.
            dx=sqrt(dxx.^2+dxy.^2);
            [dyx,dyy] = gradient(yterm,2);
            dy=sqrt(dyx.^2+dyy.^2);
            [dzx,dzy] = gradient(zterm,2);
            dz=sqrt(dzx.^2+dzy.^2);
            
            [ddxx,ddxy] = gradient(dx,2);
            ddx=sqrt(ddxx.^2+ddxy.^2);
            [ddyx,ddyy] = gradient(dy,2);
            ddy=sqrt(ddyx.^2+ddyy.^2);
            [ddzx,ddzy] = gradient(dz,2);
            ddz=sqrt(ddzx.^2+ddzy.^2);
            
            curva = sqrt((ddz.*dy-ddy.*dz).^2+(ddx.*dz-ddz.*dx).^2+(ddy.*dx-ddx.*dy).^2)./((dx.^2+dy.^2+dz.^2).^(3/2));
            
            %maxcurva  = find(abs(curva) == max(abs(curva)), 1, 'first');
            abscurva=abs(curva);
            [~,index_maxcurva]=max(abscurva(:));
            [I,J] = ind2sub(size(curva),index_maxcurva);
            lambda_opt(1) = lambda1(I);
            lambda_opt(2) = lambda2(J);
            
            if I-1<=0
                lambda1_inf=log10(lambda1(I));
                lambda1_sup=log10(lambda1(I+1));
            elseif I+1>length(lambda1)
                lambda1_inf=log10(lambda1(I-1));
                lambda1_sup=log10(lambda1(I));
            else
                lambda1_inf=log10(lambda1(I-1));
                lambda1_sup=log10(lambda1(I+1));
            end
            lambda1=logspace(lambda1_inf,lambda1_sup,4);
            
            if J-1<=0
                lambda2_inf=log10(lambda2(J));
                lambda2_sup=log10(lambda2(J+1));
            elseif J+1>length(lambda2)
                lambda2_inf=log10(lambda2(J-1));
                lambda2_sup=log10(lambda2(J));
            else
                lambda2_inf=log10(lambda2(J-1));
                lambda2_sup=log10(lambda2(J+1));
            end
            lambda2=logspace(lambda2_inf,lambda2_sup,4);  
            
            if lambda1(1)-lambda1(2)<=1e-12 && lambda2(1)-lambda2(2)<=1e-12
                break;
            end
        end  
        
        fprintf('Optimal parameters. Lambda1=%i; Lambda2=%i\n',lambda_opt(1),lambda_opt(2));
        x_hat = (AA+lambda_opt(1)*LL+lambda_opt(2)*(D'*D))\((A'*y)+(lambda_opt(2)*D*x_cons));
        toc
        if 0    % change 0 for 1 for L-Hypersurface representation.
            figure(),
            surf(xterm,yterm,zterm),title('L-Hypersurface for one-term Constrained Tikhonov'),
            hold on;
            plot3(xterm(index_maxcurva), yterm(index_maxcurva),zterm(index_maxcurva),'*r');
            plot3(xterm(index_maxcurva-1), yterm(index_maxcurva-1),zterm(index_maxcurva-1),'*g');
            plot3(xterm(index_maxcurva+1), yterm(index_maxcurva+1),zterm(index_maxcurva+1),'*');
            hold off
            xlabel('log ||Ax-y||^2');
            ylabel('log ||Dx-x_{rt}||^2');
            zlabel('log ||Lx||^2');
            fprintf('Optimal lambda values (L-Hypersurface method): lambda1=%i. lambda2=%i. \n',lambda_opt(1),lambda_opt(2));
        end
        
    elseif opt_calc==2  % CRESO
        for i=1:length(lambda1)
            fprintf('Constrained Tikhonov method, computing estimation, CRESO method (lambda1 %i/%i).\n', i,length(lambda1));
            for j=1:length(lambda2)
                x_hat = (AA+lambda1(i)*LL+lambda2(j)*(D'*D))\((A'*y)+(lambda2(j)*D*x_cons));
                magnitude_term = lambda1(i)^2*norm(L*x_hat,'fro')^2;
                error_term_1 = norm(A*x_hat-y,'fro')^2;
                error_term_2 = lambda2(i)^2*norm(D*x_hat-x_cons,'fro')^2;
                B(i,j)= magnitude_term-error_term_1-error_term_2;
            end
        end
        [Cx, Cy] = gradient(B);
        C_lambda=sqrt(Cx.^2+Cy.^2);
        
        figure(),
        surf(lambda1,lambda2,C_lambda);
        [~,locs] = findpeaks(data);
        [I,J] = ind2sub(size(C_lambda),locs);
        pause;
    else
        lambda_opt{1}=lambda1;
        lambda_opt{2}=lambda2;
        % Saco múltiples estimaciones de x_hat para evaluar parámetros.
        for i=1:length(lambda1)
            fprintf('Constrained Tikhonov method, computing estimation (lambda1 %i/%i).\n', i,length(lambda1));
            for j=1:length(lambda2)
                x_hat = (AA+lambda1(i)*LL+lambda2(j)*(D'*D))\((A'*y)+(lambda2(j)*D*x_cons));
            end
        end
    end
end
end