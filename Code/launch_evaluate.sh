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


#!/bin/bash

mkdir -p output/logs

# VAE

# 1. 6 clases + strat (E2.1)
echo "Running: OMAMI_VAE - stratified - discard_classes true"
python -u scripts/train_multioutput_optuna.py \
    --algorithm 'OMAMI_VAE' --optuna false --discard_classes false \
    --filter_EGM false --split_mode 'stratified' --oversampling true \
    2>&1 | tee output/logs/OMAMI_VAE_strat_6clases.log

# 2. 2 clases (E0.1)
echo "Running: OMAMI_VAE - stratified - discard_classes false"
python -u scripts/train_multioutput_optuna.py \
    --algorithm 'OMAMI_VAE' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'deterministic' --oversampling false \
    2>&1 | tee output/logs/OMAMI_VAE_stratified_2clases.log

 #3. 2 clases + deterministic + reg (E1.2)
echo "Running: OMAMI_VAE - deterministic - 2 classes + reg"
python -u scripts/train_multioutput_optuna.py \
    --algorithm 'OMAMI_VAE' --time_masking true --optuna false \
    --discard_classes true --filter_EGM false \
    --split_mode 'deterministic' --oversampling false \
    2>&1 | tee output/logs/OMAMI_VAE_deterministic_2clases_reg.log

# 4. 2 clases + stratified + reg (E1.2)
echo "Running: OMAMI_VAE - stratified - 2 classes + reg"
python -u scripts/train_multioutput_optuna.py \
    --algorithm 'OMAMI_VAE' --time_masking true --optuna false \
    --discard_classes true --filter_EGM false \
    --split_mode 'stratified' --oversampling true \
    2>&1 | tee output/logs/OMAMI_VAE_stratified_2clases_reg.log


# AE

# 1. 6 clases + strat (E2.1)
echo "Running: OMAMI - stratified - discard_classes true"
python -u scripts/train_multioutput_optuna.py \
    --algorithm 'OMAMI' --optuna false --discard_classes false \
    --filter_EGM false --split_mode 'stratified' --oversampling true \
    2>&1 | tee output/logs/OMAMI_strat_6clases.log

# 2. 2 clases (E0.1)
echo "Running: OMAMI - stratified - discard_classes false"
python -u scripts/train_multioutput_optuna.py \
    --algorithm 'OMAMI' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'deterministic' --oversampling false \
    2>&1 | tee output/logs/OMAMI_stratified_2clases.log

# 3. 2 clases + deterministic + reg (E1.2)
echo "Running: OMAMI - deterministic - 2 classes + reg"
python -u scripts/train_multioutput_optuna.py \
    --algorithm 'OMAMI' --time_masking true --optuna false \
    --discard_classes true --filter_EGM false \
    --split_mode 'deterministic' --oversampling false \
    2>&1 | tee output/logs/OMAMI_deterministic_2clases_reg.log

# 4. 2 clases + stratified + reg (E1.2)
echo "Running: OMAMI - stratified - 2 classes + reg"
python -u scripts/train_multioutput_optuna.py \
    --algorithm 'OMAMI' --time_masking true --optuna false \
    --discard_classes true --filter_EGM false \
    --split_mode 'stratified' --oversampling true \
    2>&1 | tee output/logs/OMAMI_stratified_2clases_reg.log

