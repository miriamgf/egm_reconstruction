#!/bin/bash

log_file="output/logs/plots/general_log_regularization.log"
echo "General logs at: $log_file"

experiment_id="OMAMI_VAE_no_filt_testing_repeated_no_filt_l2"

echo "Saving logs at: output/logs/plots/${experiment_id}__no_filt_2D.log" | tee -a "$log_file"
python -u scripts/visualization/2d_plot.py --algorithm_ID "$experiment_id" > "output/logs/plots/${experiment_id}_no_filt_2D.log" 2>&1 
wait

echo "Saving logs at: output/logs/plots/${experiment_id}__no_filt_3D.log" | tee -a "$log_file"
python -u scripts/visualization/3d_plot_egm_bpsm.py --algorithm_ID "$experiment_id" > "output/logs/plots/${experiment_id}_no_filt_3D.log" 2>&1 


# Segundo bucle: Algoritmos VAE
ID_values=("_shuffle_patient" "_tm" "_SNR20") 

for id in "${ID_values[@]}"; do
    experiment_id="OMAMI_VAE_no_filt_testing_repeated_no_filt_l2${id}"

    echo "Saving logs at: output/logs/plots/${experiment_id}__no_filt_2D.log" | tee -a "$log_file"
    python -u scripts/visualization/2d_plot.py --algorithm_ID "$experiment_id" > "output/logs/plots/${experiment_id}_no_filt_2D.log" 2>&1 
    wait

    echo "Saving logs at: output/logs/plots/${experiment_id}__no_filt_3D.log" | tee -a "$log_file"
    python -u scripts/visualization/3d_plot_egm_bpsm.py --algorithm_ID "$experiment_id" > "output/logs/plots/${experiment_id}_no_filt_3D.log" 2>&1 

done

echo "Experimentos completados."
