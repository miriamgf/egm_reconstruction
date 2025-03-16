
#echo "Saving logs at: output/logs/plots/OMAMI_VAE_no_filt_testing_repeated_2D.log"
#nohup python -u scripts/visualization/2d_plot.py --algorithm_ID 'OMAMI_VAE_no_filt_testing_repeated' > output/logs/plots/OMAMI_VAE_no_filt_testing_repeated_2D.log 2>&1 
#wait




#echo "Saving logs at: output/logs/plots/OMAMI_no_filt_testing2_repeated_2D.log"
#nohup python -u scripts/visualization/2d_plot.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated' > output/logs/plots/OMAMI_no_filt_testing2_repeated_2D.log 2>&1 
#wait

echo "Saving logs at: output/logs/plots/OMAMI_no_filt_testing2_repeated_3D.log"
nohup python -u scripts/visualization/3d_plot_egm_bpsm.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated' > output/logs/plots/OMAMI_no_filt_testing2_repeated_3D.log 2>&1 
wait


#echo "Saving logs at: output/logs/plots/OMAMI_no_filt_Optuna_bs_fs_Optuna_repeated_2D.log" | tee -a "$log_file"
#nohup python -u scripts/visualization/2d_plot.py 2>&1 --algorithm_ID 'OMAMI_no_filt_Optuna_bs_fs_Optuna_repeated' > output/logs/plots/OMAMI_no_filt_Optuna_bs_fs_Optuna_repeated_2D.log 
#wait

#echo "Saving logs at: output/logs/plots/OMAMI_no_filt_Optuna_bs_fs_Optuna_repeated_3D.log" | tee -a "$log_file"
#nohup python -u scripts/visualization/3d_plot_egm_bpsm.py 2>&1 --algorithm_ID 'OMAMI_no_filt_Optuna_bs_fs_Optuna_repeated' > output/logs/plots/OMAMI_no_filt_Optuna_bs_fs_Optuna_repeated_3D.log
#wait

#echo "Saving logs at: output/logs/plots/OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna_repeated_2D.log" | tee -a "$log_file"
#nohup python -u scripts/visualization/2d_plot.py 2>&1 --algorithm_ID 'OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna_repeated' > output/logs/plots/OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna_repeated_2D.log 
#wait

#echo "Saving logs at: output/logs/plots/OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna_repeated_3D.log" | tee -a "$log_file"
#nohup python -u scripts/visualization/3d_plot_egm_bpsm.py 2>&1 --algorithm_ID 'OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna_repeated' > output/logs/plots/OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna_repeated_3D.log 
#wait

