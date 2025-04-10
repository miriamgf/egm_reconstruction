#!/bin/bash


echo "Saving training log at: output/logs/OMAMI_stratified_no_oversampling.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --optuna false --filter_EGM false --split_mode 'stratified' --oversampling false  > output/logs/OMAMI_stratified_no_oversampling.log 2>&1  &
wait

echo "Saving training log at: output/logs/OMAMI_VAE_stratified_no_oversampling.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --optuna false --filter_EGM false --split_mode 'stratified' --oversampling false  > output/logs/OMAMI_VAE_stratified_no_oversampling.log 2>&1  &
wait

echo "Saving training log at: output/logs/OMAMI_stratified_oversampling.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI' --optuna false --filter_EGM false --split_mode 'stratified' --oversampling true  > output/logs/OMAMI_stratified_oversampling.log 2>&1  &
wait

echo "Saving training log at: output/logs/OMAMI_VAE_stratified_oversampling.log" | tee -a "$log_file"
nohup python -u scripts/train_multioutput_optuna.py --algorithm 'OMAMI_VAE' --optuna false --filter_EGM false --split_mode 'stratified' --oversampling true  > output/logs/OMAMI_VAE_stratified_oversampling.log 2>&1  &
wait

echo "Saving training log at: output/logs/OMAMI_VAE_gen.log" | tee -a "$log_file"
nohup python -u scripts/generation_EGM.py --algorithm 'OMAMI_VAE' --optuna false --filter_EGM false > output/logs/OMAMI_VAE_gen.log 2>&1  &
wait