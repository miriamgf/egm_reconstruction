#!/bin/bash

# Ejecutar comandos de terminal automáticamente


# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log
echo "OMAMI_VAE_ski Optuna"
export CUDA_VISIBLE_DEVICES=1
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --n_nodes 1024 > output/logs/output_VAE_SKIP_optuna.log 2>&1 &

#./simple_launch_optuna.sh