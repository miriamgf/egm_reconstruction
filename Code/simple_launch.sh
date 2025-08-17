#!/bin/bash

mkdir -p output/logs

# VAE


echo "Running: OMAMI_VAE opt to replicate, logs at: output/logs/replicate_augmented_test_source.log"
python -u scripts/train_augmented.py \
    --algorithm 'OMAMI_VAE' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'stratified' --oversampling true \
    --data_augmentation false \
    2>&1 | tee output/logs/replicate_augmented_test_source.log