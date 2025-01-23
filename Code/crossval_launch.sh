#!/bin/bash

# Ejecutar comandos de terminal automáticamente

# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log
export CUDA_VISIBLE_DEVICES=0
echo "cross val OMAMI 2048, 400"
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --n_nodes 2048 --filter_EGM false --fold 0 > output/logs/output_OMAMI_optuna_fold_0.log 2>&1 
wait $!
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --n_nodes 2048 --filter_EGM false --fold 1 > output/logs/output_OMAMI_optuna_fold_1.log 2>&1 
wait $!
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --n_nodes 2048 --filter_EGM false --fold 2 > output/logs/output_OMAMI_optuna_fold_2.log 2>&1 
wait $!
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --n_nodes 2048 --filter_EGM false --fold 3 > output/logs/output_OMAMI_optuna_fold_3.log 2>&1 

#echo "OMAMI ski"
#nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_ski' > output/logs/output_AE_ski_weighted.log 2>&1 
#./simple_launch.sh
