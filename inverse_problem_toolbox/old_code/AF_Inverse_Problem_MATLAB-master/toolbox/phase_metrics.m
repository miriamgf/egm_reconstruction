function [phase, phase_metrics]=phase_metrics(x_hat,DF,model,fs,xphase, selected_nodes,filt_HDF)

phase = instantphase (x_hat, DF, model, fs,filt_HDF);
[~, CC, RDMS] = timemetrics (xphase(1:2039,:)', phase(1:2039,:)');
phase_metrics.RDMSt_phase = RDMS';
phase_metrics.MRDMSt_phase=mean(phase_metrics.RDMSt_phase(selected_nodes));
phase_metrics.std_MRDMSt_phase=std(phase_metrics.RDMSt_phase(selected_nodes));
phase_metrics.CCt_phase = CC';
phase_metrics.MCCt_phase=mean(phase_metrics.CCt_phase(selected_nodes));
phase_metrics.std_MCCt_phase=std(phase_metrics.CCt_phase(selected_nodes));

end