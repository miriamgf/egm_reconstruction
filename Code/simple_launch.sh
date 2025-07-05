#!/bin/bash

mkdir -p output/logs

# VAE

# 2. 2 clases (E0.1)
echo "Running: OMAMI_VAE - stratified - discard_classes false"
python -u scripts/train_multioutput_optuna.py \
    --algorithm 'OMAMI_VAE' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'deterministic' --oversampling false \
    2>&1 | tee output/logs/OMAMI_VAE_stratified_2clases_det.log

# 2. 2 clases (E0.1)
echo "Running: OMAMI_VAE - stratified - discard_classes false"
python -u scripts/train_multioutput_optuna.py \
    --algorithm 'OMAMI' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'deterministic' --oversampling false \
    2>&1 | tee output/logs/OMAMI_VAE_stratified_2clases_det.log

     
#1 OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_strat_5_class_overs
#2 OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_strat_2_class_overs
#3 OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_tm_strat_2_class_overs_det
#4 OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_tm_strat_2_class_overs


#5 OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_5_class_overs
#6 OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_2_class_overs
#7 OMAMI_no_filt_testing2_repeated_no_filt_l2_tm_strat_2_class_overs_det
#8 OMAMI_no_filt_testing2_repeated_no_filt_l2_tm_strat_2_class_overs

# Lista de algorithm_IDs
ALGORITHMS=(
  "OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_strat_2_class_overs_det" #E2.1 ?
  "OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_2_class_overs_det"
)

# Ejecutar cada evaluación secuencialmente
for ALG_ID in "${ALGORITHMS[@]}"; do
  LOG_FILE="output/logs/evaluation/${ALG_ID}.log"
  echo "Ejecutando evaluación para: $ALG_ID"
  echo "Guardando log en: $LOG_FILE"

  python -u scripts/evaluation/evaluate_main.py \
    --algorithm_ID "$ALG_ID" \
    --ev_DL true \
    --stratified_split true \
    --ev_TIK true \
    | tee "$LOG_FILE"
done
