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

# Esperar 7 horas (7 * 60 * 60 segundos)

#

#echo "output/logs/evaluation/OMAMI_VAE_no_filt.log"
#nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_no_filt' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_VAE_no_filt.log 2>&1 &
#wait

#echo "output/logs/evaluation/OMAMI_VAE_no_filt_testing.log"
#nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_no_filt_testing_repeated' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_VAE_no_filt_testing.log 2>&1 
#wait

#echo "output/logs/evaluation/OMAMI_no_filt_testing2.log"
#nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_no_filt_testing2.log 2>&1 &
#wait

echo "output/logs/evaluation/OMAMI_no_filt_Optuna_bs_fs_Optuna_repeated.log"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_no_filt_Optuna_bs_fs_Optuna_repeated' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_no_filt_Optuna_bs_fs_Optuna_repeated.log 2>&1 
wait

echo "output/logs/evaluation/OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna_repeated.log"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna_repeated' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_VAE_Reduced_no_filt_Optuna_bs_fs_Optuna.log 2>&1 &
#wait
#echo "output/logs/evaluation/OMAMI_VAE_Optuna_1.log"
#nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_Optuna_1' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_VAE_Optuna_1.log 2>&1 &
#wait
#echo "output/logs/evaluation/OMAMI_repeated.log"
#nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_repeated' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_repeated.log 2>&1 &
#wait
#echo "Finished!!"
