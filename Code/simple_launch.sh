#!/bin/bash

# Ejecutar comandos de terminal automáticamente

# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log
export CUDA_VISIBLE_DEVICES=0
echo "OMAMI 2048, 400"
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --n_nodes 2048 > output/logs/output_VAE_optuna.log 2>&1 
#wait $!

#echo "OMAMI ski"
#nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_ski' > output/logs/output_AE_ski_weighted.log 2>&1 
#./simple_launch.sh
