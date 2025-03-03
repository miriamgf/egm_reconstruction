echo "Step: [3/3] Plot 2D..."
echo "Saving logs at: output/logs/plots/OMAMI_no_filt_Optuna_bs_fs_Optuna_2D.log"
nohup python -u scripts/visualization/2d_plot.py --algorithm_ID 'OMAMI_no_filt_Optuna_bs_fs_Optuna' > output/logs/plots/OMAMI_no_filt_Optuna_bs_fs_Optuna_2D.log 2>&1 
wait
echo "Step: [3/3] Plot 3D..."
echo "Saving logs at: output/logs/plots/OMAMI_no_filt_Optuna_bs_fs_Optuna_3D.log"
nohup python -u scripts/visualization/3d_plot_egm_bpsm.py --algorithm_ID 'OMAMI_no_filt_Optuna_bs_fs_Optuna' > output/logs/plots/OMAMI_no_filt_Optuna_bs_fs_Optuna_3D.log 2>&1 
