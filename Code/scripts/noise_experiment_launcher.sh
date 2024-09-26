#!/bin/bash

# Ejecutar comandos de terminal automáticamente
echo "Iniciando experimento de ruido..."

# Ejecutar cada comando en segundo plano y redirigir la salida a archivos de log
echo "E1 -> Execution --SNR_em_noise 5 --SNR_white_noise 3"
python scripts/train_multioutput.py --SNR_em_noise 3 --SNR_white_noise 20 --experiment_number 1 --patches_oclussion "PT" --unfold_code 1 > output/logs/output_CINC_E1.log 2>&1 
echo "E2 -> Execution --SNR_em_noise 10 --SNR_white_noise 5"
python scripts/train_multioutput.py --SNR_em_noise 5 --SNR_white_noise 20 --experiment_number 2 --patches_oclussion "PT" --unfold_code 1 > output/logs/output_CINC_E2.log 2>&1 
echo "E3 -> Execution --SNR_em_noise 20 --SNR_white_noise 10"
python scripts/train_multioutput.py --SNR_em_noise 10 --SNR_white_noise 20 --experiment_number 3 --patches_oclussion "PT" --unfold_code 1 > output/logs/output_CINC_E3.log 2>&1 
echo "E4 -> Execution --SNR_em_noise 30 --SNR_white_noise 20"
python scripts/train_multioutput.py --SNR_em_noise 20 --SNR_white_noise 20 --experiment_number 4 --patches_oclussion "PT" --unfold_code 1 > output/logs/output_CINC_E4.log 2>&1 
echo "E5 -> Execution --SNR_em_noise 20 --SNR_white_noise 20 --Oclussion P1"
python scripts/train_multioutput.py --SNR_em_noise 20 --SNR_white_noise 20 --experiment_number 5 --patches_oclussion "P1" --unfold_code 1 > output/logs/output_CINC_E5.log 2>&1 
echo "E6 -> Execution --SNR_em_noise 20 --SNR_white_noise 20 --Oclussion P2"
python scripts/train_multioutput.py --SNR_em_noise 20 --SNR_white_noise 20 --experiment_number 6 --patches_oclussion "P2" --unfold_code 1 > output/logs/output_CINC_E6.log 2>&1 
echo "E7 -> Execution --SNR_em_noise 20 --SNR_white_noise 20 --Oclussion P3"
python scripts/train_multioutput.py --SNR_em_noise 20 --SNR_white_noise 20 --experiment_number 7 --patches_oclussion "P3" --unfold_code 1 > output/logs/output_CINC_E7.log 2>&1 
echo "E8 -> Execution --SNR_em_noise 20 --SNR_white_noise 20 --Oclussion P4"
python scripts/train_multioutput.py --SNR_em_noise 20 --SNR_white_noise 20 --experiment_number 8 --patches_oclussion "P4" --unfold_code 1 > output/logs/output_CINC_E8.log 2>&1 
echo "E9 -> Execution --SNR_em_noise 20 --SNR_white_noise 20 --Oclussion None --unfold_code 2"
python scripts/train_multioutput.py --SNR_em_noise 20 --SNR_white_noise 20 --experiment_number 9 --patches_oclussion "PT" --unfold_code 2 > output/logs/output_CINC_E9.log 2>&1 

echo "El script ha terminado."
