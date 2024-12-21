function [x_hat,lambda_opt] = constrained_tikhonov (A, AA, L, LL, y, x, lambda1, opt_calc0, lambda2, D, x_cons, opt_calc1, lambda3, opt_calc2)
% Tikhonov reconstruction method.
%
% This routine by Víctor Suárez Gutiérrez (victor.suarez.gutierrez@urjc.es)
% Modified by Miguel Ángel Cámara (miguelangel.camara@urjc.es)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUT:
% A: transfer matrix.
% L: matrix which takes part in regularization term of Tikhonov approach.
% AA: A'*A
% LL: L'*L
% y: torso potentials.
% lambda: regularization parameter.
% order: Tikhonov and TSVD order (0, 1 or 2).
% SNR: signal noise rate (dB).
% reg_param_method: global (g) or by instant (i) calculation.
% compute_params: 1 if reg_params need to be calculated. Otherwise 0.

%
%OUTPUT:
% x_hat: estimated epicardial potentials.
% lambda_opt: optimal regularization parameters (one if global or an array by instant).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
if nargin==8
    [x_hat,lambda_opt] = Classical_Tikhonov_Mod (A, AA, L, LL, y, x, lambda1,opt_calc0,100);
elseif nargin==12
    %[x_hat,lambda_opt] = Constrained_Tikhonov_L2 (A, AA, L, LL, y, x, lambda1, lambda2, D, x_cons, opt_calc1);
    [x_hat,lambda_opt] = Constrained_Tikhonov_L2_Mod (A, AA, L, LL, y, x, lambda1, lambda2, D, x_cons, opt_calc1,100);
elseif nargin==13
    [x_hat,lambda_opt] = Constrained_Tikhonov_L3 (A, AA, LL, y, x, lambda1, lambda2, D, x_cons, lambda3, opt_calc2);
end

end

