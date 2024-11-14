#!/bin/bash

# Ejecutar comandos de terminal automáticamente
echo "Iniciando experimento de ruido..."

# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log
echo "E0 -> Execution --SNR_em_noise 100 --SNR_white_noise 20"
nohup python scripts/train_multioutput_optuna.py > output/logs/output_VAE1.log 2>&1 &

#./simple_launch.sh