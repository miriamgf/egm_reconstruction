#!/bin/bash

# Ejecutar comandos de terminal automáticamente

# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log
echo "OMAMI_ski"
nohup python scripts/train_multioutput_optuna.py --algorithm 'OMAMI_ski' --LSTM 'True' > output/logs/output_AE_SKI_lstm.log 2>&1 &

#./simple_launch.sh