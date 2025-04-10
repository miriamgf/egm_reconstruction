#!/bin/bash

echo "Saving training log at: output/logs/OMAMI_VAE_gen.log" | tee -a "$log_file"
nohup python -u scripts/generation_EGM.py --algorithm 'OMAMI_VAE' --optuna false --filter_EGM false --evaluation false > output/logs/OMAMI_VAE_gen.log 2>&1  &
wait