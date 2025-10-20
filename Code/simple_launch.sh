#!/bin/bash

mkdir -p output/logs

# VAE
python -u scripts/train_augmented.py \
    --algorithm 'OMAMI_VAE' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'stratified' --oversampling true \
    --data_augmentation true --perc_augmentation 25 \
    2>&1 | tee output/logs/replicate_augmented_test_source_25_cond.log

echo "Running: OMAMI_VAE opt to replicate, logs at: output/logs/replicate_augmented_test_source_10_cond.log"
python -u scripts/train_augmented.py \
    --algorithm 'OMAMI_VAE' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'stratified' --oversampling true \
    --data_augmentation true --perc_augmentation 10 \
    2>&1 | tee output/logs/replicate_augmented_test_source_10_cond.log

echo "Running: OMAMI_VAE opt to replicate, logs at: output/logs/replicate_augmented_test_source_14_cond.log"
python -u scripts/train_augmented.py \
    --algorithm 'OMAMI_VAE' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'stratified' --oversampling true \
    --data_augmentation true --perc_augmentation 14 \
    2>&1 | tee output/logs/replicate_augmented_test_source_14_cond.log

echo "Running: OMAMI_VAE opt to replicate, logs at: output/logs/replicate_augmented_test_source_18_cond.log"
python -u scripts/train_augmented.py \
    --algorithm 'OMAMI_VAE' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'stratified' --oversampling true \
    --data_augmentation true --perc_augmentation 18 \
    2>&1 | tee output/logs/replicate_augmented_test_source_18_cond.log

echo "Running: OMAMI_VAE opt to replicate, logs at: output/logs/replicate_augmented_test_source_20_cond.log"
python -u scripts/train_augmented.py \
    --algorithm 'OMAMI_VAE' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'stratified' --oversampling true \
    --data_augmentation true --perc_augmentation 20 \
    2>&1 | tee output/logs/replicate_augmented_test_source_20_cond.log

echo "Finished"

