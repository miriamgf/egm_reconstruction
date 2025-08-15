#!/bin/bash

# Lista de algorithm_IDs
ALGORITHMS=(
  "OMAMI_VAE_no_filt_testing_repeated_no_filt_strat_2_class_overs_REPLICATE",
  "OMAMI_VAE_no_filt_testing_repeated_no_filt_strat_2_class_overs_augmented" 
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
    --ev_TIK false \
    | tee "$LOG_FILE"
done



