#!/bin/bash

# Ejecutar comandos de terminal automáticamente

# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log
export CUDA_VISIBLE_DEVICES=0
echo "OMAMI"
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --LSTM 'True' > output/logs/output_AE_ls.log 2>&1 &

#./simple_launch.sh