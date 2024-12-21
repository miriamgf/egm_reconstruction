function [signals_k, DFs]=kuklik_transform_IP(signals, Fs);

% signals: [N x T]

lowf = 1.5; 
highf = 25; % those setting work but different numbers may work
period_min = 130e-3; 
period_max = 280e-3; 

%%%%%%%%%%%%%

% signals=signals(indices_const,:);

%%%%%%%%%%

Lecg=signals;

siglen = size(Lecg,2); 
n_signals = size(Lecg,1);  

sig = zeros(n_signals,siglen); 
rsig = zeros(n_signals,siglen);

[b,a] = butter(4, [lowf highf]/(Fs/2), 'bandpass'); %prep bandpass filter
 
for i = 1:n_signals
    sig(i,:)=filtfilt(b,a,Lecg(i,:));
end

%save array of dV/dt --> first derivative
diffv = diff(sig')';

%determine pd (Welch Method FFT)
pd = nan(n_signals,1); tvec = 1/Fs:1/Fs:siglen/Fs; %time at each sample in seconds
seglength = 2*Fs; noverlap = Fs; %We are using 2000 ms segments (not 2000 points)
nFFT = 200^2; %zero pad FFT to get finer resolution (effectively interpolates spectrum) 
for i = 1:n_signals
  %find largest continuous chunk of signal for each channel
  isig = sig(i,:); sigtimes = find(~isnan(isig)); %non-saturated times
  if ~isempty(sigtimes) %make sure there is some signal here
      nvals = [0 cumsum(diff(sigtimes)~=1)]; %mark consecutive regions
      sigseg = isig(sigtimes(nvals==mode(nvals))); %find longest consecutive region

      %find dominant frequency of this chunk
      if numel(sigseg)>=seglength %ensure we have enough signal
        [pxx,f] = pwelch(sigseg,seglength,noverlap,nFFT,Fs); %plot(1./f,10*log10(pxx));
        invf = 1./f; pxx = pxx(invf>=period_min & invf<=period_max); 
        invf(invf<period_min | invf>period_max) = []; %ensure period is between 130-280ms
        pdval = invf(find(pxx==max(pxx))); [~,mind]=min(abs(tvec-pdval)); 
        pd(i) = mind; %save pd in units of samples NOT seconds
      else
          %if you have < 2000 ms signal, can't get a good spectrum - error
        disp(['Not Enough Signal to Take FFT! Channel ' num2str(i) ' Discarded.']); 
        sig(i,:)=nan;
      end
  end
end
 
 %compute sine recomposition per Kuklik et al. 2015 paper
for i = 1:n_signals
     for t = 1:siglen-1
            if ~isnan(pd(i)) %check that there is some data in this channel
                 halfpd = round(pd(i)/2);
                 dvdt = diffv(i,t); tt = (-halfpd:halfpd);
                 swave = sin(2*pi*tt/pd(i))*abs(dvdt)*((1-sign(dvdt))/2);
                 twave = t + (-halfpd:halfpd); swave(twave<=0 | twave>siglen) = []; 
                 twave(twave<=0)=[]; twave(twave>siglen)=[];
                 rsig(i,twave) = rsig(i,twave) + swave;
            else
                rsig(i,:)=nan; %no data, mark to be interpolated over
            end
     end
end

DFs=Fs./(pd);
signals_k=rsig;
