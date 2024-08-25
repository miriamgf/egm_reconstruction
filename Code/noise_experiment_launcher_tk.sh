#!/bin/bash

# Ejecutar comandos de terminal automáticamente
echo "Iniciando experimento de ruido..."

# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log
echo "E0 -> Execution --SNR_em_noise 100 --SNR_white_noise 20"
nohup python scripts/Tikhonov/main.py --SNR_em_noise 100 --SNR_white_noise 20 --experiment_number 0 --patches_oclussion "PT" --unfold_code 1 > output/logs/output_CINC_E1.log 2>&1 &
echo "E1 -> Execution --SNR_em_noise 5 --SNR_white_noise 3"
nohup python scripts/Tikhonov/main.py --SNR_em_noise 3 --SNR_white_noise 20 --experiment_number 1 --patches_oclussion "PT" --unfold_code 1 > output/logs/output_CINC_E1.log 2>&1 &
echo "E2 -> Execution --SNR_em_noise 10 --SNR_white_noise 5"
nohup python scripts/Tikhonov/main.py --SNR_em_noise 5 --SNR_white_noise 20 --experiment_number 2 --patches_oclussion "PT" --unfold_code 1 > output/logs/output_CINC_E2.log 2>&1 &
echo "E3 -> Execution --SNR_em_noise 20 --SNR_white_noise 10"
nohup python scripts/Tikhonov/main.py --SNR_em_noise 10 --SNR_white_noise 20 --experiment_number 3 --patches_oclussion "PT" --unfold_code 1 > output/logs/output_CINC_E3.log 2>&1 &
echo "E4 -> Execution --SNR_em_noise 30 --SNR_white_noise 20"
nohup python scripts/Tikhonov/main.py --SNR_em_noise 20 --SNR_white_noise 20 --experiment_number 4 --patches_oclussion "PT" --unfold_code 1 > output/logs/output_CINC_E4.log 2>&1 &

echo "El experimento ha terminado."
##./noise_experiment_launcher_tk.sh
