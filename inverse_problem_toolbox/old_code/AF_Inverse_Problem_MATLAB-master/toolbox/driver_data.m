function [driver_metrics]=driver_data(x_hat_tikh, DF, fs, xmodes, atrial_areas, xSMF, atrial_model_filled, atrial_model_nofilled,filt_HDF)

[driver_metrics.driver, driver_metrics.modes, driver_metrics.SMF] = driver_location (atrial_model_filled , x_hat_tikh, DF, fs,filt_HDF);
if ~isempty(driver_metrics.SMF)
    [driver_metrics]=driver_metrics_SMF(driver_metrics, xmodes, atrial_areas, xSMF, atrial_model_nofilled);
end
end

function [driver_metrics]=driver_metrics_SMF(driver_metrics, xmodes, atrial_areas, xSMF, atrial_model_nofilled)
    [driver_metrics.WUI, driver_metrics.WOI, driver_metrics.CCdriver, driver_metrics.MD] = drivermetrics (xmodes, driver_metrics.modes, atrial_areas, xSMF, driver_metrics.SMF, atrial_model_nofilled);
end