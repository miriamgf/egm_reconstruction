#!/bin/bash


# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log
export CUDA_VISIBLE_DEVICES=1
echo "OMAMI"
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --LSTM 'True' > output/logs/output_AE_lstm.log 2>&1 &

#./simple_launch.sh