#!/bin/bash

mkdir -p output/logs

# VAE

# 2. 2 clases (E0.1)
echo "Running: OMAMI_VAE opt to replicate, logs at: output/logs/replicate.log"
python -u scripts/train_augmented.py \
    --algorithm 'OMAMI_VAE' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'stratified' --oversampling true \
    2>&1 | tee output/logs/replicate_augmented.log
