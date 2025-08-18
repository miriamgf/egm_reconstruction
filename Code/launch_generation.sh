#!/bin/bash

echo "Saving training log at: output/logs/Generation_replicate.log" | tee -a "$log_file"
nohup python -u scripts/generation_EGM.py --optuna false --evaluation true > output/logs/Generation_replicate.log 2>&1  &