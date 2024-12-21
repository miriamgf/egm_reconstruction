function metrics = timemetrics_mod (x,x_hat,selected_nodes)
% This function calculates different metrics to evaluate the difference
% between real values (x) and estimated ones (x_hat)
%
% Input parameters:
%   - x: Ground Truth. Expected Dimensions N x T, 
%        being N the number of leads and T the number of time instants.
%   - x_hat: Estimatd signal. Expected dimensions N x T, 
%            being N the number of leads and T the number of time instants.
%
% This function by Felipe Alonso-Atienza (felipe.alonso@urjc.es)
% Modified by Miguel Ángel Cámara (miguelangel.camara@urjc.es)

% parsing
if nargin < 4
    verbose=0;
end

%% Norms
norm_x     = sqrt(sum(x.^2));
norm_x_hat = sqrt(sum(x_hat.^2));

%% Root mean square error
error = (x - x_hat);    % this is a matrix
RMSEt = sqrt(sum(error.^2))./norm_x; % sum over rows.

%% Correlation coefficient
CCt=sum(x.*x_hat)./(norm_x.*norm_x_hat);

%% Relative Difference Measure Star (RDMS)
% Definition taken from: INVERSE ELECTROCARDIOGRAPHY USING REDUCED 
% LEAD- SET BY TTLS AND LTTLS REGULARIZATION ALGORITHMS
%(http://www.wseas.us/e-library/conferences/2014/Istanbul/ELECT/ELECT-19.pdf)

% normalize signals by its power for each time instant
x_norm     = bsxfun(@rdivide,x,norm_x);         
x_norm_hat = bsxfun(@rdivide,x_hat,norm_x_hat);

% calculate the norm of the difference of signals (for each time instant)
difference = x_norm - x_norm_hat;
RDMSt       = sqrt(sum(difference.^2));

%% mean values over time
% metrics.MRMSE = mean(RMSEt(selected_nodes));
% metrics.stdRMSE = std(RMSEt(selected_nodes));
metrics.MCC   = mean(CCt(selected_nodes));
metrics.stdCC   = std(CCt(selected_nodes));
metrics.MRDMS = mean(RDMSt(selected_nodes));
metrics.stdRDMS = std(RDMSt(selected_nodes));
metrics.RDMSt = RDMSt;
metrics.CCt = CCt;
% metrics.RMSEt = RMSEt;

%% drawing for debugging
if verbose
    n = 0:numel(RMSE)-1;
    
    % RMSE and RDMS
    figure;
    plot(n, RMSE); 
    hold on;
    plot(n, RDMS, 'r'); 
    plot(n, ones(1, numel(n))*mean(RMSE), 'Linewidth', 4);
    plot(n, ones(1,numel(n))*mean(RDMS), 'r', 'Linewidth', 4);
    hold off;
    title('SNR 10 dB', 'Fontsize', 25);
    xlabel('n [sample]', 'Fontsize', 18);   
    ylabel('Normalized Error','Interpreter','Latex','Fontsize',18)
    legend('RMSE','RDMS')

    % CC
    figure;
    plot(n,CC); 
    hold on;
    plot(n,ones(1,numel(n))*mean(CC), 'Linewidth', 4);
    hold off;
    title('SNR 10 dB', 'Fontsize', 25);
    xlabel('n [sample]', 'Fontsize', 18);   
    ylabel('CC','Interpreter','Latex','Fontsize',18)
end
end