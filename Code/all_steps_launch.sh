echo "Step: [1/3] Training..."
echo "Saving training log at: output/logs/OMAMI_VAE_Reduced_no_filt_Optuna_long.log"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE_Reduced' --optuna true --filter_EGM false > output/logs/OMAMI_VAE_Reduced_no_filt_Optuna_long.log 2>&1  &
wait

echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna.log"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna' --ev_DL false --ev_TIK true > output/logs/evaluation/OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna.log 2>&1 
wait

echo "Step: [3/3] Plot 3D..."
echo "Saving logs at: output/logs/plots/OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna.png"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna' --ev_DL false --ev_TIK true > output/logs/evaluation/OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna.log 2>&1 

