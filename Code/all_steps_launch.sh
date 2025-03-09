log_file="output/logs/overfitting_general.log" 
echo "General files at: $log_file"

# 1. Only L2 and high dropout - AE
echo "Step: [1/3] Training..."
echo "Saving training log at: output/logs/OMAMI_no_filt_testing2_repeated_l2.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --optuna false --filter_EGM false > output/logs/OMAMI_no_filt_testing2_repeated_l2.log 2>&1  &
wait

# 2. Only L2 and high dropout - VAE
echo "Step: [1/3] Training..."
echo "Saving training log at: output/logs/OMAMI_VAE_no_filt_testing_repeated_l2.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --optuna false --filter_EGM false > output/logs/OMAMI_VAE_no_filt_testing_repeated_l2.log 2>&1  &
wait

# 3. Shuffeling - AE
echo "Step: [1/3] Training..."
echo "Saving training log at: output/logs/OMAMI_no_filt_testing2_repeated_shuffle.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --optuna false --shuffle_patient true --filter_EGM false > output/logs/OMAMI_no_filt_testing2_repeated_shuffle.log 2>&1  &
wait

# 3. Shuffeling - VAE
echo "Step: [1/3] Training..."
echo "Saving training log at: output/logs/OMAMI_VAE_no_filt_testing_repeated_shuffle.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --optuna false --shuffle_patient true --filter_EGM false > output/logs/OMAMI_VAE_no_filt_testing_repeated_shuffle.log 2>&1  &
wait

# Evaluations
echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_no_filt_testing2_repeated_l2.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated_l2' --ev_DL false --ev_TIK true > output/logs/evaluation/OMAMI_no_filt_testing2_repeated_l2.log 2>&1 
wait

echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_l2.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_no_filt_testing_repeated_l2' --ev_DL false --ev_TIK true > output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_l2.log 2>&1 
wait

echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_no_filt_testing2_repeated_shuffle.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated_shuffle' --ev_DL false --ev_TIK true > output/logs/evaluation/OMAMI_no_filt_testing2_repeated_shuffle.log 2>&1 
wait

echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_shuffle.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_no_filt_testing_repeated_shuffle' --ev_DL false --ev_TIK true > output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_shuffle.log 2>&1 



