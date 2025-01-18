#!/bin/bash


# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log


echo "OMAMI_VAE 2048, 400"
export CUDA_VISIBLE_DEVICES=0
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --n_nodes 2048 > output/logs/output_VAE_400_2048_norm.log 2>&1 

echo "OMAMI_VAE ski 2048, 400"
export CUDA_VISIBLE_DEVICES=1
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE_ski' --n_nodes 2048  > output/logs/output_VAE_ski_400_2048_norm.log 2>&1 

#./simple_launch_2.sh

