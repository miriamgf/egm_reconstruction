log_file="output/logs/overfitting_general.log" 
echo "General files at: $log_file"


# 3. SNR 20
#echo "Step: [1/3] Training..."
#echo "Saving training log at: output/logs/OMAMI_no_filt_testing2_repeated_l2_SNR20.log" | tee -a "$log_file"
#nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --optuna false --SNR_white_noise 20 --filter_EGM false > output/logs/OMAMI_no_filt_testing2_repeated_l2_SNR20.log 2>&1  &
#wait

# 4. SNR 20
#echo "Step: [1/3] Training..."
#echo "Saving training log at: output/logs/OMAMI_VAE_no_filt_testing_repeated_l2_SNR20.log" | tee -a "$log_file"
#nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --optuna false --SNR_white_noise 20 --filter_EGM false > output/logs/OMAMI_VAE_no_filt_testing_repeated_l2_SNR20.log 2>&1  &
#wait

# Evaluations
echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_no_filt_testing2_repeated_l2_tm.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated_l2_tm' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_no_filt_testing2_repeated_l2_tm.log 2>&1 
wait

echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_l2_tm.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_no_filt_testing_repeated_l2_tm' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_l2_tm.log 2>&1 
wait

echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_no_filt_testing2_repeated_l2_SNR20.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated_l2_SNR20' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_no_filt_testing2_repeated_l2_SNR20.log 2>&1 
wait

echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_l2_SNR20.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_no_filt_testing_repeated_l2_SNR20' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_l2_SNR20.log 2>&1 

log_file="output/logs/overfitting_general.log" 
echo "General files at: $log_file"

# 1. Only L2 and high dropout - AE
echo "Step: [1/3] Training..."
echo "Saving training log at: output/logs/OMAMI_no_filt_testing2_repeated_l2_attention.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --attention true --optuna false --filter_EGM false > output/logs/OMAMI_no_filt_testing2_repeated_l2_attention.log 2>&1  &
wait

# 2. Only L2 and high dropout - VAE
echo "Step: [1/3] Training..."
echo "Saving training log at: output/logs/OMAMI_VAE_no_filt_testing_repeated_l2_attention.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --attention true --optuna false --filter_EGM false > output/logs/OMAMI_VAE_no_filt_testing_repeated_l2_attention.log 2>&1  &

# --------------------- Evaluations
echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_no_filt_testing2_repeated_l2_attention.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated_l2_attention' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_no_filt_testing2_repeated_l2_attention.log 2>&1 
wait

echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_l2_attention.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_no_filt_testing_repeated_l2_attention' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_l2_attention.log 2>&1 
wait








