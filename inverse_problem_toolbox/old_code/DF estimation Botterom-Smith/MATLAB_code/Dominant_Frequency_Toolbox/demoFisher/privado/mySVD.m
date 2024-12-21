function [freq_range,SVR,auxeig,v] = mysvd(Electrogram,Fs,method);

%
plotflag = 1;
if nargin == 2
method = 'svd';
%method = 'pca';
%method = 'ica';
end

rand('seed',0); rand('seed',0);
Electrogram = awgn(Electrogram,80);

%b = fir1(8,100/1000); a = 1;
%[N,wn] = buttord(30/1000,40/1000,1,10);
%[b,a] = butter(N,wn);
%Electrogram = filtfilt(b,a,Electrogram);
%Electrogram = cumsum(Electrogram-mean(Electrogram));

Fmin=3;     Fmax=30;
N_min=round(Fs/Fmax);
N_max=round(Fs/Fmin);
L = length(Electrogram);
Electrogram = Electrogram(:)';

index=1;
auxeig = zeros(N_max,N_max);
for window_length=N_min:N_max,

    %if window_length == 99,keyboard,end;
    
    auxegm = Electrogram;
    B = buffer(auxegm,window_length);
    B = B';
    % Normalization
    [m,n] = size(B);
    for i=1:m
        %B(i,:) = B(i,:) - min(B(i,:));
        B(i,:) = B(i,:)/(eps+max((B(i,:))));
        %B(i,:) = B(i,:)/(eps+norm(B(i,:)));
        %B(i,:) = B(i,:)/std(B(i,:));
    end
    %[dump rot_i1]=max(B(1,:));
    rot_i1 = round(window_length/2);
    for i=1:m;
        [dump rot_i]=max(B(i,:));
        B(i,:)=circshift(B(i,:)',rot_i1-rot_i);
    end
    B = B(1:end-1,:);
    %Norm_coeff(index)=std(sum(B'))*sqrt(window_length);
    Norm_coeff(index) = 1;
    switch method
        case 'svd'            
            [U,Lambda,V] = svd(B);
            SVR(index) = Lambda(1,1)/(eps+Lambda(2,2))...
                /Norm_coeff(index);
            pp = diag(Lambda);
            auxeig(index,1:length(pp)) = pp;
        case 'pca';
            C = cov(B,1);
            [VV,Lambda] = eig(C);
            pp = diag(Lambda);
            auxeig(index,1:length(pp)) = pp(end:-1:1);
            SVR(index) = pp(end)/(pp(end-1)+eps)/Norm_coeff(index);
        case 'ica'
            nn = round(m/2);
            auxsig = [mean(B(1:nn,:),1);...
                mean(B(nn+1:end,:),1)];
            [icasig,A,W] = fastica (auxsig,'numOfIC',4);
            l1 = sum((W(1,:)));
            if length(W(:,1))>1
                l2 = sum((W(2,:)));
            else
                l2 = eps;
            end
            SVR(index) = abs(l1/l2);
    end
    index=index+1;
end
SVR=SVR/max(SVR);
freq_range=Fs./[N_min:N_max];

% ************************************
%           Find the eigenvector  
% ************************************
[forget,ind] = max(SVR);
auxegm = Electrogram;
B = buffer(auxegm,N_min+ind(1));
B = B';
% Normalization
[m,n] = size(B);
for i=1:m
    B(i,:) = B(i,:)/(eps+max((B(i,:))));
end
rot_i1 = round((N_min+ind(1))/2);
for i=1:m;
    [dump rot_i]=max(B(i,:));
    B(i,:)=circshift(B(i,:)',rot_i1-rot_i);
end
B = B(1:end-1,:);
switch method
    case 'svd'
        [U,Lambda,V] = svd(B);
        v = V(:,1);
    case 'pca';
        C = cov(B,1);
        [VV,Lambda] = eig(C);
        v = VV(:,end);
end

% ************************************
%           Some plots
% ************************************
if plotflag
    figure(1)
    subplot(211),
    plot(freq_range,SVR);
    subplot(212),
    stem(N_min:N_max,SVR);

    figure(2)
    plot(v), axis tight;
end
keyboard