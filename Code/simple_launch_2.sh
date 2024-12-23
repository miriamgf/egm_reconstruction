#!/bin/bash


# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log
export CUDA_VISIBLE_DEVICES=1
echo "OMAMI_VAE 1024"
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_ski' --n_nodes 1024 > output/logs/output_AE_ski_1024.log 2>&1 

#nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE'  > output/logs/output_VAE_weighted.log 2>&1 
#wait $!
#echo "OMAMI_VAE"
#nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE_ski'  > output/logs/output_VAE_ski_weighted.log 2>&1 
#./simple_launch_2.sh