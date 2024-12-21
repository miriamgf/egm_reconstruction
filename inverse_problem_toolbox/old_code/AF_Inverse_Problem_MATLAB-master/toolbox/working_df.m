function xDF = working_df(x, fs)
% Assessment of dominant frequency for each node.
%
% This function by Víctor Suárez Gutiérrez (victor.suarez.gutierrez@urjc.es)
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUT:
% x: signal.
% fs: sampling frequency.
% model: place where rotor occurs {'SR', 'SAF', 'CAF'}
%
%OUTPUT:
% xDF: dominant frequency for each node of model.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Assessment:
[N,~] = size(x);
xDF = zeros(N,1);

% if ~isempty(strfind(model,'Sinusal'))
%     for i=1:N
%         signal = x(i,:);
%         rx = xcorr(signal,'coeff');
%         [pks,locs] = findpeaks(rx,'MinPeakDistance',200);
%         xDF(i,:) = fs*(round(length(locs)/2)-1)/length(signal(1,:));
%     end
% else
L = 1024;
j=1;
while L > length(x(1,:))
    L = 2^nextpow2(N-j);
    j=j+1;
end
for i=1:N
    egm  = x(i,:);
    egm = egm-mean(egm);
    [Pff,f] = pwelch(egm,hamming(L),[],L,fs);
    [~,idx] = max(Pff);
    
    n_harmonics = 3;
    
    f_harmonics = [];
    
    for n = 2:4%look up to 4*f0
        
        f_to_search = f((f>(n*f(idx)-(f(idx)/4))) &  (f<((n+1)*f(idx)- (f(idx)/4))));
        idx_f_to_search = find(f>=f_to_search(1) & f<=f_to_search(end));
       [p_max, idx_h] = max(Pff(idx_f_to_search));
       f_harmonics = [f_harmonics,f_to_search(idx_h)];
    end
    
    xDF = f(idx);
    
    f0_harmonics = mean(diff(f_harmonics));
    if abs(f(idx)-f0_harmonics) > 1
        xDF = f0_harmonics;
    end
    
    
    %     end
end