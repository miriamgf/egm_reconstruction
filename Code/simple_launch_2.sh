#!/bin/bash


# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log


echo "OMAMI_VAE 2048, 400"
export CUDA_VISIBLE_DEVICES=0
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --n_nodes 2048 > output/logs/output_VAE_400_2048_norm.log 2>&1 

echo "OMAMI_VAE ski 2048, 400"
export CUDA_VISIBLE_DEVICES=1
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE_ski' --n_nodes 2048  > output/logs/output_VAE_ski_400_2048_norm.log 2>&1 

#./simple_launch_2.sh

nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --n_nodes 2048 --filter_EGM false --fold 3 > output/logs/output_OMAMI_optuna_fold_3.log 2>&1 
wait
echo "output/logs/evaluation/evaluate_tik_fold3.log"
nohup python -u scripts/evaluation/evaluate_tik.py --algorithm_ID 'OMAMI_fold_3_no_filt' > output/logs/evaluation/evaluate_tik_OMAMI_fold_3_no_filt.log 2>&1 &
wait
echo "output/logs/evaluation/evaluate_dl_fold3.log"
nohup python -u scripts/evaluation/evaluate_dl.py --algorithm_ID 'OMAMI_fold_3_no_filt' > output/logs/evaluation/evaluate_dl_OMAMI_fold_3_no_filt.log 2>&1 &
echo "output/logs/output_OMAMI_VAE_optuna.log"


-------
echo "output/logs/evaluation/evaluate_tik_OMAMI_no_filt_tk.log"
nohup python -u scripts/evaluation/evaluate_tik.py --algorithm_ID 'OMAMI_VAE_no_filt' > output/logs/evaluation/evaluate_tik_OMAMI_no_filt_tk.log 2>&1 &
echo "All evaluation scripts have completed."

echo "output/logs/evaluation/evaluate_tik_VAE_OPT.log"
nohup python -u scripts/evaluation/evaluate_tik.py --algorithm_ID 'OMAMI_VAE_Optuna' > output/logs/evaluation/evaluate_tik_VAE_OPT.log 2>&1 &
wait
echo "output/logs/evaluation/evaluate_dl_VAE_OPT.log"
nohup python -u scripts/evaluation/evaluate_dl.py --algorithm_ID 'OMAMI_VAE_Optuna' > output/logs/evaluation/evaluate_dl_VAE_OPT.log 2>&1 &
wait
echo "output/logs/evaluation/evaluate_tik_fold3.log"
nohup python -u scripts/evaluation/evaluate_tik.py --algorithm_ID 'OMAMI_fold_3_no_filt' > output/logs/evaluation/evaluate_tik_OMAMI_fold_3_no_filt.log 2>&1 &
wait
echo "output/logs/evaluation/evaluate_dl_fold3.log"
nohup python -u scripts/evaluation/evaluate_dl.py --algorithm_ID 'OMAMI_fold_3_no_filt' > output/logs/evaluation/evaluate_dl_OMAMI_fold_3_no_filt.log 2>&1 &
echo "output/logs/output_OMAMI_VAE_optuna.log"