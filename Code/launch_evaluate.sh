#echo "Launching evaluate_dl.py"
#nohup python scripts/evaluation/evaluate_dl.py --algorithm_ID 'OMAMI_VAE' > output/logs/evaluation/evaluate_dl.log 2>&1 &

#echo "Launching evaluate_tik.py"
#nohup python scripts/evaluation/evaluate_tik.py --algorithm_ID 'OMAMI_VAE' > output/logs/evaluation/evaluate_tik.log 2>&1 &

# Wait for both background processes to finish
#wait

#echo "All evaluation scripts have completed."

#nohup python scripts/evaluation/evaluate_tik.py --algorithm_ID 'OMAMI_VAE_Optuna' > output/logs/evaluation/evaluate_tik.log 2>&1 &
#wait

#echo "output/logs/evaluation/evaluate_tik_OMAMI_VAE.log"
#nohup python -u  scripts/evaluation/evaluate_tik.py --algorithm_ID 'OMAMI_VAE_Optuna' > output/logs/evaluation/evaluate_tik_OMAMI_VAE.log 2>&1 &
#wait

echo "output/logs/evaluation/evaluate_tik_OMAMI_VAE_no_filt.log"
nohup python -u scripts/evaluation/evaluate_dl.py --algorithm_ID 'OMAMI_VAE_no_filt' > output/logs/evaluation/evaluate_tik_OMAMI_VAE_no_filt.log 2>&1 &
wait
echo "output/logs/evaluation/evaluate_tik_OMAMI_VAE_no_filt.log"
nohup python -u scripts/evaluation/evaluate_tik.py --algorithm_ID 'OMAMI_VAE_no_filt' > output/logs/evaluation/evaluate_tik_OMAMI_VAE_no_filt.log 2>&1 &
echo "All evaluation scripts have completed."
#nohup python visualization/3d_plot_egm_bpsm.py --algorithm_ID 'OMAMI_VAE_Optuna' > output/logs/evaluation/evaluate_tik.log 2>&1 &

#OMAMI_VAE_Optuna  evaluate_tik