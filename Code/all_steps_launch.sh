log_file="output/logs/overfitting_general.log" 
echo "General files at: $log_file"



# --------------------- Evaluations


# Evaluations
echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_no_filt_testing2_repeated_no_filt_l2_attention.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated_no_filt_l2_attention' --ev_DL true --ev_TIK false > output/logs/evaluation/OMAMI_no_filt_testing2_repeated_no_filt_l2_attention.log 2>&1 
wait

echo "Step: [2/3] Evaluating..."
echo "Saving evaluation log at: output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_attention.log" | tee -a "$log_file"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_attention' --ev_DL true --ev_TIK false > output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_attention.log 2>&1 
wait




