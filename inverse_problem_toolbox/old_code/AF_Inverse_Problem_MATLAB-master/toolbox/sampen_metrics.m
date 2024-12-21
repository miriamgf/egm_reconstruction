function [x_hat_SampEn,samp_metrics] = sampen_metrics(x_hat,fs)

x_hat_SampEn = sampen_egms(x_hat,fs);
samp_metrics.mSampEn = mean(x_hat_SampEn);
samp_metrics.stdSampEn = std(x_hat_SampEn);