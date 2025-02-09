#!/bin/bash

# Ejecutar comandos de terminal automáticamente

# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log
export CUDA_VISIBLE_DEVICES=0

#echo "python output/logs/evaluation/evaluate_tik_OMAMI_no_filt_tk.log"
#nohup python -u  scripts/evaluation/evaluate_tik.py --algorithm_ID 'OMAMI_no_filt' > output/logs/evaluation/evaluate_tik_OMAMI_no_filt_tk.log 2>&1 &
#wait

echo "output/logs/OMAMI_no_filt_testing_tfdata.log"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --optuna false --filter_EGM false > output/logs/OMAMI_no_filt_testing_tfdata.log 2>&1  

#echo "output/logs/OMAMI_no_filt_Optuna_VAE_testing.log"
#nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --optuna true --filter_EGM false > output/logs/OMAMI_no_filt_optuna_VAE_testing.log 2>&1 &




#echo "output/logs/output_OMAMI_VAE_no_filt_bs_fs.log"
#nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --optuna true --filter_EGM false > output/logs/output_OMAMI_VAE_no_filt_bs_fs.log 2>&1 
#wait



#echo "output/logs/output_OMAMI_VAE_optuna.log"
#nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --n_nodes 2048 --filter_EGM false --fold 3 > output/logs/output_OMAMI_optuna_fold_3.log 2>&1 
#wait

#nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --n_nodes 2048 --filter_EGM false > output/logs/output_OMAMI_VAE_no_filt.log 2>&1 
#wait

#echo "OMAMI ski"
#nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_ski' > output/logs/output_AE_ski_weighted.log 2>&1 
#./simple_launch.sh
#--filter_EGM false