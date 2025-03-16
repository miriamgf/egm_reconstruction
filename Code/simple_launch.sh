#!/bin/bash

# Ejecutar comandos de terminal automáticamente

# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log

echo "Saving training log at: output/logs/OMAMI_no_filt_testing2_repeated_l2.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --optuna false --filter_EGM false > output/logs/OMAMI_no_filt_testing2_repeated_l2.log 2>&1  &
wait

