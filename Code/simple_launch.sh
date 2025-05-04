#!/bin/bash



echo "Saving training log at: output/logs/OMAMI_VAE_stratified_2_classes.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --optuna false --discard_classes true --filter_EGM false --split_mode 'stratified' --oversampling true  > output/logs/OMAMI_VAE_stratified_2_classes.log 2>&1  &
wait
