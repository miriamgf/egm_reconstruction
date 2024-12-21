function [q_detect,r_detect,s_detect] = qrs_detection(ecg,fs)

% QRS Detection Example
% shows the effect of each filter according to Pan-Tompkins algorithm.
% Note that, the decision  algorithm is different then the mentioned algorithm.



x1 = ecg;
%x1 = load('ecg3.dat'); % load the ECG signal from the file
%fs = 200;              % Sampling rate
N = length (x1);       % Signal length
t = [0:N-1]/fs;        % time index


% figure(1)
% subplot(2,1,1)
% plot(t,x1)
% xlabel('second');ylabel('Volts');title('Input ECG Signal')
% subplot(2,1,2)
% plot(t(200:600),x1(200:600))
% xlabel('second');ylabel('Volts');title('Input ECG Signal 1-3 second')
% xlim([1 3])

% Cancellation DC drift and normalization

x1 = x1 - mean (x1 );    % cancel DC conponents
x1 = x1/ max( abs(x1 )); % normalize to one

% figure(2)
% subplot(2,1,1)
% plot(t,x1)
% xlabel('second');ylabel('Volts');title(' ECG Signal after cancellation DC drift and normalization')
% subplot(2,1,2)
% plot(t(200:600),x1(200:600))
% xlabel('second');ylabel('Volts');title(' ECG Signal 1-3 second')
% xlim([1 3])

% Low Pass Filtering

% LPF (1-z^-6)^2/(1-z^-1)^2
b=[1 0 0 0 0 0 -2 0 0 0 0 0 1];
a=[1 -2 1];


h_LP=filter(b,a,[1 zeros(1,12)]); % transfer function of LPF

x2 = conv (x1 ,h_LP);
%x2 = x2 (6+[1: N]); %cancle delay
x2 = x2/ max( abs(x2 )); % normalize , for convenience .

% figure(3)
% subplot(2,1,1)
% plot([0:length(x2)-1]/fs,x2)
% xlabel('second');ylabel('Volts');title(' ECG Signal after LPF')
% xlim([0 max(t)])
% subplot(2,1,2)
% plot(t(200:600),x2(200:600))
% xlabel('second');ylabel('Volts');title(' ECG Signal 1-3 second')
% xlim([1 3])

% High Pass Filtering

% HPF = Allpass-(Lowpass) = z^-16-[(1-z^-32)/(1-z^-1)]
b = [-1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 -32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1];
a = [1 -1];

h_HP=filter(b,a,[1 zeros(1,32)]); % impulse response iof HPF

x3 = conv (x2 ,h_HP);
% x3 = x3 (16+[1: N]); %cancle delay
x3 = x3/ max( abs(x3 ));

% figure(4)
% subplot(2,1,1)
% plot([0:length(x3)-1]/fs,x3)
% xlabel('second');ylabel('Volts');title(' ECG Signal after HPF')
% xlim([0 max(t)])
% subplot(2,1,2)
% plot(t(200:600),x3(200:600))
% xlabel('second');ylabel('Volts');title(' ECG Signal 1-3 second')
% xlim([1 3])
% 
% Derivative Filter

% Make impulse response
h = [-1 -2 0 2 1]/8;
% Apply filter
x4 = conv (x3 ,h);
x4 = x4 (2+[1: N]);
x4 = x4/ max( abs(x4 ));

% figure(5)
% subplot(2,1,1)
% plot([0:length(x4)-1]/fs,x4)
% xlabel('second');ylabel('Volts');title(' ECG Signal after Derivative')
% subplot(2,1,2)
% plot(t(200:600),x4(200:600))
% xlabel('second');ylabel('Volts');title(' ECG Signal 1-3 second')
% xlim([1 3])
% 
% Squaring

x5 = x4 .^2;
x5 = x5/ max( abs(x5 ));
% figure(6)
% subplot(2,1,1)
% plot([0:length(x5)-1]/fs,x5)
% xlabel('second');ylabel('Volts');title(' ECG Signal Squarting')
% subplot(2,1,2)
% plot(t(200:600),x5(200:600))
% xlabel('second');ylabel('Volts');title(' ECG Signal 1-3 second')
% xlim([1 3])

% Moving Window Integration

% Make impulse response
h = ones (1 ,31)/31;
Delay = 15; % Delay in samples

% Apply filter
x6 = conv (x5 ,h);
x6 = x6(15+[1: N]);
x6 = x6/max( abs(x6 ));

% figure(7)
% subplot(2,1,1)
% plot([0:length(x6)-1]/fs,x6)
% xlabel('second');ylabel('Volts');title(' ECG Signal after Averaging')
% subplot(2,1,2)
% plot(t(200:600),x6(200:600))
% xlabel('second');ylabel('Volts');title(' ECG Signal 1-3 second')
% xlim([1 3])

% Find QRS Points Which it is different than Pan-Tompkins algorithm

% figure(7)
% subplot(2,1,1)
max_h = max(x6);
% thresh = mean(x6);
thresh = median(x6); %works better for EGM
poss_reg =(x6>thresh*max_h)';

% figure (8)
% subplot(2,1,1)
% hold on
% plot (t(200:600),x1(200:600)/max(x1))
% box on
% xlabel('second');ylabel('Integrated')
% xlim([1 3])

% subplot(2,1,2)
% plot (t(200:600),x6(200:600)/max(x6))
% xlabel('second');ylabel('Integrated')
% xlim([1 3])

left = find(diff([0 poss_reg])==1);
right = find(diff([poss_reg 0])==-1);

left=left-(6+16);  % cancle delay because of LP and HP
right=right-(6+16);% cancle delay because of LP and HP

for i=1:length(left)
    [R_value(i) R_loc(i)] = max( x1(left(i):right(i)) );
    R_loc(i) = R_loc(i)-1+left(i); % add offset

    [Q_value(i) Q_loc(i)] = min( x1(left(i):R_loc(i)) );
    Q_loc(i) = Q_loc(i)-1+left(i); % add offset

    [S_value(i) S_loc(i)] = min( x1(left(i):right(i)) );
    S_loc(i) = S_loc(i)-1+left(i); % add offset

end

% there is no selective wave
Q_loc=Q_loc(find(Q_loc~=0));
R_loc=R_loc(find(R_loc~=0));
S_loc=S_loc(find(S_loc~=0));

q_detect = Q_loc;
r_detect = R_loc;
s_detect = S_loc;


% figure
% subplot(2,1,1)
% title('ECG Signal with R points');
% plot (t,x1/max(x1) , t(R_loc) ,R_value , 'r^', t(S_loc) ,S_value, '*',t(Q_loc) , Q_value, 'o');
% legend('ECG','R','S','Q');
% subplot(2,1,2)
% plot (t,x1/max(x1) , t(R_loc) ,R_value , 'r^', t(S_loc) ,S_value, '*',t(Q_loc) , Q_value, 'o');
% xlim([1 3])
 